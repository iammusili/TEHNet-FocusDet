"""TEHNet: translation-equivariant heatmap network, and heatmap decoding.

Backbone: 5 residual blocks (96 channels, dilations 1/2/4/8/16), each with two
3x3 dilated convs and an inverted-bottleneck FFN (1x1 convs, expansion 4).
Head: two 3x3 convs + 1x1 projection. All convs have stride 1 and no
downsampling is used, so the network is strictly translation-equivariant.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ResBlockFFN(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1, ffn_expand=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_ch)
        hidden = out_ch * ffn_expand
        self.ffn = nn.Sequential(
            nn.Conv2d(out_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ffn(out)
        return self.relu(out + identity)


class HeatmapHead(nn.Module):
    def __init__(self, in_ch=96, mid_ch=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return self.conv3(x)


def build_tehnet(channels=96, dilations=(1, 2, 4, 8, 16)):
    blocks, in_ch = [], 3
    for d in dilations:
        blocks.append(ResBlockFFN(in_ch, channels, dilation=d))
        in_ch = channels
    return nn.Sequential(*blocks), HeatmapHead(channels)


def load_tehnet(ckpt_path, device):
    """The checkpoint holds 'backbone_state_dict' and 'head_state_dict'."""
    backbone, head = build_tehnet()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    backbone.load_state_dict(ckpt['backbone_state_dict'])
    head.load_state_dict(ckpt['head_state_dict'])
    return backbone.to(device).eval(), head.to(device).eval()


@torch.no_grad()
def infer_heatmap(backbone, head, img_rgb01, device):
    """img_rgb01: (H, W, 3) float32 RGB in [0, 1], native resolution.
    Returns the (H, W) heatmap tensor in [0, 1]."""
    x = torch.from_numpy(img_rgb01).permute(2, 0, 1)
    x = (x - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
    logits = head(backbone(x.unsqueeze(0).to(device)))
    return torch.sigmoid(logits[0, 0]).float()


def _distance_nms(points, d_min):
    """Greedy point NMS: keep a point only if it is >= d_min px from every
    higher-scoring kept point. points: (N, 3) [x, y, score]."""
    order = np.argsort(-points[:, 2])
    pts = points[order]
    xy = pts[:, :2]
    kept = []
    for i in range(len(pts)):
        if kept:
            dx = xy[kept, 0] - xy[i, 0]
            dy = xy[kept, 1] - xy[i, 1]
            if (dx * dx + dy * dy).min() < d_min * d_min:
                continue
        kept.append(i)
    return pts[kept]


def extract_peaks(heatmap, nms_kernel=3, score_thresh=0.1, top_k=500, d_min=6.0):
    """Local maxima above score_thresh (top_k by score), then distance NMS.
    Returns (N, 3) numpy [x, y, score]."""
    hm = heatmap[None, None]
    hm_max = F.max_pool2d(hm, nms_kernel, stride=1, padding=nms_kernel // 2)
    keep = ((hm == hm_max) & (hm > score_thresh))[0, 0]
    ys, xs = torch.where(keep)
    scores = heatmap[ys, xs]
    if scores.numel() > top_k:
        scores, idx = torch.topk(scores, top_k)
        ys, xs = ys[idx], xs[idx]
    peaks = torch.stack([xs.float(), ys.float(), scores], dim=1).cpu().numpy()
    if d_min > 0 and len(peaks) > 1:
        peaks = _distance_nms(peaks, d_min)
    return peaks


def peaks_to_boxes(peaks, box_size, img_h, img_w):
    """Square boxes of side box_size centred at the peaks, clipped to the image.
    Returns (N, 4) numpy [x1, y1, x2, y2]."""
    if len(peaks) == 0:
        return np.empty((0, 4))
    half = box_size / 2.0
    x, y = peaks[:, 0], peaks[:, 1]
    return np.column_stack([np.clip(x - half, 0, img_w), np.clip(y - half, 0, img_h),
                            np.clip(x + half, 0, img_w), np.clip(y + half, 0, img_h)])
