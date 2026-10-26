"""Single-image demo: runs the whole pipeline once, prints per-stage timing and
writes visualizations of each stage.

    python focusdet/demo.py <image> [--sr_scale 1] [--out demo_out]

Detection uses TPH-YOLOv5 detect.py with the evaluation setting (1996 px, TTA,
conf 0.001, NMS IoU 0.6); the reported numbers in the README come from
run_focus_det.sh, not from this demo.
"""
import argparse
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from focusdet import focus as F  # noqa: E402
from focusdet.tehnet import load_tehnet, infer_heatmap, extract_peaks, peaks_to_boxes  # noqa: E402

TPH_DIR = os.path.join(_REPO, 'third_party', 'tph_yolov5')
CLASSES = ('pedestrian', 'people', 'bicycle', 'car', 'van',
           'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
COLORS = [(85, 170, 0), (0, 120, 255), (0, 200, 255), (255, 120, 120), (170, 0, 255),
          (0, 255, 200), (255, 200, 0), (255, 0, 200), (0, 0, 255), (255, 128, 0)]


class Timer:
    def __init__(self):
        self.rows = []

    def __call__(self, stage, fn, note=lambda r: ''):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        self.rows.append((stage, dt, note(r)))
        print('  [{:8.1f} ms] {}  {}'.format(dt * 1000, stage, self.rows[-1][2]))
        return r


def detect(image_path, args, tag):
    """TPH-YOLOv5 detect.py on one image -> (N, 6) YOLO rows (cls cx cy w h conf)."""
    cmd = [sys.executable, 'detect.py', '--source', os.path.abspath(image_path),
           '--weights', os.path.abspath(args.det_weights), '--imgsz', str(args.det_imgsz), '--augment',
           '--conf-thres', '0.001', '--iou-thres', '0.6', '--max-det', '300',
           '--save-txt', '--save-conf', '--nosave', '--device', '0' if args.device == 'cuda' else 'cpu',
           '--project', os.path.abspath(args.out), '--name', tag, '--exist-ok']
    subprocess.run(cmd, cwd=TPH_DIR, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    dets = F.load_yolo_txt(os.path.join(args.out, tag, 'labels', stem + '.txt'))
    return np.empty((0, 6)) if dets is None else dets


def draw(img, per_class, score_th=0.25):
    out = img.copy()
    for c, dets in enumerate(per_class):
        for x1, y1, x2, y2, s in dets:
            if s >= score_th:
                cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[c], 2)
                cv2.putText(out, '{} {:.2f}'.format(CLASSES[c], s), (int(x1), max(12, int(y1) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[c], 1, cv2.LINE_AA)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('image')
    p.add_argument('--out', default='demo_out')
    p.add_argument('--sr_scale', type=int, default=2, choices=[1, 2])
    p.add_argument('--tehnet', default=os.path.join(_REPO, 'weights', 'tehnet_best.pth'))
    p.add_argument('--edsr', default=os.path.join(_REPO, 'weights', 'visdrone_x2.pt'))
    p.add_argument('--det_weights', default=os.path.join(_REPO, 'weights', 'yolov5l-xs-2.pt'))
    p.add_argument('--det_imgsz', type=int, default=1996)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if not torch.cuda.is_available():
        args.device = 'cpu'
    os.makedirs(args.out, exist_ok=True)
    save = lambda name, im: cv2.imwrite(os.path.join(args.out, name), im)  # noqa: E731
    T = Timer()

    img = cv2.imread(args.image)
    assert img is not None, 'cannot read ' + args.image
    H, W = img.shape[:2]
    backbone, head = load_tehnet(args.tehnet, args.device)
    sr_model = F.load_edsr(args.edsr, args.device) if args.sr_scale == 2 else None
    print('{} ({}x{}), sr_scale={}'.format(args.image, W, H, args.sr_scale))

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    hm = T('TEHNet heatmap', lambda: infer_heatmap(backbone, head, rgb, args.device))
    peaks = T('peak decoding', lambda: extract_peaks(hm), lambda r: '{} peaks'.format(len(r)))
    boxes = peaks_to_boxes(peaks, 96, H, W)
    rects = T('region merging', lambda: F.merge_regions(boxes, W, H, args.sr_scale),
              lambda r: '{} boxes -> {} regions'.format(len(boxes), len(r)))

    hm_color = cv2.applyColorMap((np.clip(hm.cpu().numpy(), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    save('01_heatmap.png', cv2.addWeighted(img, 0.55, hm_color, 0.45, 0))
    vis = img.copy()
    for x, y, _ in peaks:
        cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
    save('02_boxes96.png', vis)
    vis = img.copy()
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 128, 255), 3)
    save('03_regions.png', vis)

    g_rows = detect(args.image, args, 'det_global')
    g = [[] for _ in range(F.NUM_CLASSES)]
    for d in g_rows:
        g[int(round(d[0]))].append([*F.yolo_to_xyxy(d, W, H), d[5]])
    focus = [[] for _ in range(F.NUM_CLASSES)]

    if rects:
        canvas, rec = T('SR + packing', lambda: F.build_focus_image(img, rects, args.sr_scale, sr_model, args.device),
                        lambda r: 'canvas {}x{}'.format(r[0].shape[1], r[0].shape[0]))
        focus_path = os.path.join(args.out, '05_focus_image.jpg')  # JPEG, as in run_focus_det.sh
        cv2.imwrite(focus_path, canvas)
        f_rows = detect(focus_path, args, 'det_focus')
        ch, cw = canvas.shape[:2]
        on_canvas = [[] for _ in range(F.NUM_CLASSES)]
        for d in f_rows:
            on_canvas[int(round(d[0]))].append([*F.yolo_to_xyxy(d, cw, ch), d[5]])
        save('06_focus_dets.png', draw(canvas, on_canvas))
        focus = F.map_to_original(f_rows, rec, cw, ch, args.sr_scale, max_area=1024)
        if args.sr_scale == 2:
            x1, y1, x2, y2 = max(rects, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
            crop = cv2.resize(img[y1:y2, x1:x2], (2 * (x2 - x1), 2 * (y2 - y1)), interpolation=cv2.INTER_NEAREST)
            sr = F.super_resolve(sr_model, img[y1:y2, x1:x2], args.device)
            save('04_sr_compare.png', np.hstack([crop, np.full((crop.shape[0], 12, 3), 255, np.uint8), sr]))

    fused = F.fuse(g, focus)
    base, final = draw(img, g), draw(img, fused)
    for im, title in ((base, 'baseline (global branch)'), (final, 'fused')):
        cv2.rectangle(im, (0, 0), (im.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(im, title, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    save('07_compare.png', np.hstack([base, np.full((H, 16, 3), 255, np.uint8), final]))

    with open(os.path.join(args.out, 'timing.md'), 'w') as f:
        f.write('| stage | ms | note |\n|---|---|---|\n')
        for stage, dt, note in T.rows:
            f.write('| {} | {:.1f} | {} |\n'.format(stage, dt * 1000, note))
    print('outputs written to', args.out)


if __name__ == '__main__':
    main()
