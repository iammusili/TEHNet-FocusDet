"""Focus-image construction, inverse coordinate mapping and branch fusion.

Box expansion, greedy merging, packing (PHSPPOG) and the inverse mapping follow
UFPMP-Det; the x2 super-resolution model is the EDSR released by GLSAN.
Shared by the evaluation pipeline and the single-image demo.
"""
import math
import os
import sys

import cv2
import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from third_party.ufp.unified_foreground_packing import (  # noqa: E402
    scale_boxes, ForegroundRegionGeneration, Packing)

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
NUM_CLASSES = 10


def list_images(image_dir):
    return [f for f in sorted(os.listdir(image_dir)) if os.path.splitext(f)[1].lower() in IMG_EXTS]


# ---------------------------------------------------------------- focus image
def merge_regions(boxes, img_w, img_h, sr_scale, box_scale=1.5):
    """Expand boxes by box_scale and greedily merge them (UFPMP-Det).
    Returns integer regions [(x1, y1, x2, y2), ...] in original-image pixels.
    Merging is done in the x sr_scale coordinate space."""
    if len(boxes) == 0:
        return []
    boxes_s = np.asarray(boxes, dtype=np.float64) * sr_scale
    expanded = scale_boxes(boxes_s, box_scale, image_shape=[img_w * sr_scale, img_h * sr_scale])
    regions, _ = ForegroundRegionGeneration(boxes_s, expanded)
    rects = []
    for rx1, ry1, rx2, ry2 in regions / sr_scale:
        x1, y1 = max(0, int(np.floor(rx1))), max(0, int(np.floor(ry1)))
        x2, y2 = min(img_w, int(np.ceil(rx2))), min(img_h, int(np.ceil(ry2)))
        if x2 > x1 and y2 > y1:
            rects.append((x1, y1, x2, y2))
    return rects


def load_edsr(weights, device):
    from third_party.edsr.edsr import EDSR
    model = EDSR()
    model.load_state_dict(torch.load(weights, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def super_resolve(model, img_bgr, device):
    """x2 EDSR on a BGR uint8 image; returns BGR uint8."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = torch.FloatTensor(rgb.copy()).to(device).permute(2, 0, 1).unsqueeze(0)
    out = model(x).squeeze().permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def build_focus_image(img, rects, sr_scale, sr_model=None, device='cpu'):
    """Crop every region, magnify it (EDSR x2 if sr_scale == 2) and pack all
    patches into one canvas without resampling.
    Returns (canvas, rec); each rec row is [x1, y1, w, h, n_x, n_y, 1] with the
    source box in the x sr_scale space and (n_x, n_y) its offset on the canvas."""
    patches = {}
    for x1, y1, x2, y2 in rects:
        crop = img[y1:y2, x1:x2]
        patches[(x1, y1, x2 - x1, y2 - y1)] = super_resolve(sr_model, crop, device) if sr_scale == 2 else crop.copy()
    regions = np.array([[x1 * sr_scale, y1 * sr_scale, x2 * sr_scale, y2 * sr_scale]
                        for x1, y1, x2, y2 in rects], dtype=np.float64)
    rec, canvas_w, canvas_h = Packing(regions, np.array([1] * len(rects)))
    canvas = np.zeros((int(math.ceil(canvas_h)), int(math.ceil(canvas_w)), 3), np.uint8)
    for r in rec:
        rx1, ry1, rw, rh, n_x, n_y, _ = [int(math.floor(v)) for v in r]
        canvas[n_y:n_y + rh, n_x:n_x + rw] = patches[(rx1 // sr_scale, ry1 // sr_scale,
                                                      rw // sr_scale, rh // sr_scale)]
    rec = [[v.item() if isinstance(v, np.generic) else v for v in r] for r in rec]
    return canvas, rec


# ------------------------------------------------------------ detections I/O
def load_yolo_txt(path):
    """YOLO txt (cls cx cy w h conf, normalized) -> (N, 6) array; None if missing."""
    if not os.path.exists(path):
        return None
    if os.path.getsize(path) == 0:
        return np.empty((0, 6))
    d = np.loadtxt(path)
    return d.reshape(1, -1) if d.ndim == 1 else d


def yolo_to_xyxy(row, img_w, img_h):
    cx, cy = float(row[1]) * img_w, float(row[2]) * img_h
    w, h = float(row[3]) * img_w, float(row[4]) * img_h
    x1, y1 = cx - w / 2, cy - h / 2
    return x1, y1, x1 + w, y1 + h


def write_visdrone(path, per_class):
    """per_class[c] = iterable of [x1, y1, x2, y2, score] -> VisDrone result txt."""
    write_visdrone_rows(path, [(c, *d) for c, dets in enumerate(per_class) for d in dets])


def write_visdrone_rows(path, rows):
    """rows: iterable of (cls, x1, y1, x2, y2, score), cls 0-based."""
    with open(path, 'w') as f:
        for c, x1, y1, x2, y2, s in rows:
            f.write('%.6f %.6f %.6f %.6f %.6f %d -1 -1\n' % (x1, y1, x2 - x1, y2 - y1, s, int(c) + 1))


# ------------------------------------------------------- mapping and fusion
def _inside_ratio(box, patch):
    """Fraction of the area of box that lies inside patch."""
    left, top = max(box[0], patch[0]), max(box[1], patch[1])
    right, bottom = min(box[2], patch[2]), min(box[3], patch[3])
    if left >= right or top >= bottom:
        return 0
    return (right - left) * (bottom - top) / ((box[2] - box[0]) * (box[3] - box[1]))


def map_to_original(dets, rec, canvas_w, canvas_h, sr_scale, max_area=None):
    """Assign each focus-image detection to the patch that contains more than 90%
    of its area (others are discarded) and map it back to original-image coordinates.
    dets: (N, 6) YOLO rows on the canvas. Returns per-class lists of [x1, y1, x2, y2, score]."""
    out = [[] for _ in range(NUM_CLASSES)]
    for chip in rec:
        o_x1, o_y1, w, h, n_x, n_y, sf = [math.floor(v) for v in chip]
        chip_box = [n_x, n_y, n_x + w * sf, n_y + h * sf]
        for d in dets:
            x1, y1, x2, y2 = yolo_to_xyxy(d, canvas_w, canvas_h)
            if _inside_ratio([x1, y1, x2, y2], chip_box) > 0.9:
                new_w = (x2 - x1) / sf / sr_scale
                new_h = (y2 - y1) / sf / sr_scale
                new_x = ((x1 - n_x) / sf + o_x1) / sr_scale
                new_y = ((y1 - n_y) / sf + o_y1) / sr_scale
                if max_area is None or new_w * new_h <= max_area:
                    out[int(round(d[0]))].append([new_x, new_y, new_x + new_w, new_y + new_h, d[5]])
    return out


def nms(dets, thresh):
    dets = np.asarray(dets)
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        w = np.maximum(0.0, np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]]) + 1)
        h = np.maximum(0.0, np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]]) + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


def fuse(global_per_class, focus_per_class, iou=0.6):
    """Concatenate both branches per class and apply per-class NMS."""
    fused = []
    for g, f in zip(global_per_class, focus_per_class):
        dets = np.array(g + f)
        fused.append(dets[nms(dets, iou)] if len(dets) else [])
    return fused
