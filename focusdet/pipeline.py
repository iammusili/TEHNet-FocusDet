"""Command-line steps of the focus branch (called by run_focus_det.sh).

  boxes   TEHNet heatmap -> peaks -> s x s boxes            (one json per image)
  pack    merge regions -> (EDSR x2) -> pack into focus image (+ rec json)
  global  global-branch YOLO txt -> VisDrone result txt
  focus   focus-branch YOLO txt -> mapped back -> VisDrone result txt
  fuse    global + small focus detections -> per-class NMS -> VisDrone result txt
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from focusdet import focus as F  # noqa: E402
from focusdet.tehnet import load_tehnet, infer_heatmap, extract_peaks, peaks_to_boxes  # noqa: E402


def _device(name):
    return name if name == 'cpu' or torch.cuda.is_available() else 'cpu'


def _stem(name):
    return os.path.splitext(name)[0]


def cmd_boxes(a):
    device = _device(a.device)
    backbone, head = load_tehnet(a.tehnet, device)
    os.makedirs(a.out_dir, exist_ok=True)
    names = F.list_images(a.image_dir)
    for i, name in enumerate(names):
        img = cv2.imread(os.path.join(a.image_dir, name))
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        peaks = extract_peaks(infer_heatmap(backbone, head, rgb, device), a.nms_kernel,
                              a.score_thresh, a.top_k, a.d_min)
        boxes = peaks_to_boxes(peaks, a.box_size, h, w).tolist()
        with open(os.path.join(a.out_dir, _stem(name) + '.json'), 'w') as f:
            json.dump(boxes, f)
        print('[boxes] {}/{} {}: {} boxes'.format(i + 1, len(names), name, len(boxes)))


def cmd_pack(a):
    device = _device(a.device)
    sr_model = F.load_edsr(a.edsr, device) if a.sr_scale == 2 else None
    os.makedirs(a.out_img_dir, exist_ok=True)
    os.makedirs(a.out_rec_dir, exist_ok=True)
    names = F.list_images(a.image_dir)
    for i, name in enumerate(names):
        rec_path = os.path.join(a.out_rec_dir, _stem(name) + '.json')
        img = cv2.imread(os.path.join(a.image_dir, name))
        box_path = os.path.join(a.box_dir, _stem(name) + '.json')
        boxes = json.load(open(box_path)) if os.path.exists(box_path) else []
        rects = F.merge_regions(boxes, img.shape[1], img.shape[0], a.sr_scale, a.box_scale) if boxes else []
        if not rects:
            json.dump([], open(rec_path, 'w'))  # no focus image; fused result = global branch
            continue
        canvas, rec = F.build_focus_image(img, rects, a.sr_scale, sr_model, device)
        # Same file name as the source image, i.e. saved as JPEG (OpenCV default quality 95).
        cv2.imwrite(os.path.join(a.out_img_dir, name), canvas)
        with open(rec_path, 'w') as f:
            json.dump(rec, f, indent=2)
        print('[pack] {}/{} {}: {} boxes -> {} regions, canvas {}x{}'.format(
            i + 1, len(names), name, len(boxes), len(rects), canvas.shape[1], canvas.shape[0]))


def cmd_global(a):
    os.makedirs(a.out_dir, exist_ok=True)
    for name in F.list_images(a.image_dir):
        w, h = Image.open(os.path.join(a.image_dir, name)).size
        dets = F.load_yolo_txt(os.path.join(a.det_dir, _stem(name) + '.txt'))
        rows = [(d[0], *F.yolo_to_xyxy(d, w, h), d[5]) for d in ([] if dets is None else dets)]
        F.write_visdrone_rows(os.path.join(a.out_dir, _stem(name) + '.txt'), rows)


def _load_focus(a, name, max_area=None):
    """Mapped focus-branch detections of one image, or None if unavailable."""
    rec_path = os.path.join(a.rec_dir, _stem(name) + '.json')
    dets = F.load_yolo_txt(os.path.join(a.focus_det_dir, _stem(name) + '.txt'))
    if dets is None or not os.path.exists(rec_path):
        return None
    rec = json.load(open(rec_path))
    if len(rec) == 0 or len(dets) == 0:
        return [[] for _ in range(F.NUM_CLASSES)]
    cw, ch = Image.open(os.path.join(a.focus_img_dir, name)).size
    return F.map_to_original(dets, rec, cw, ch, a.sr_scale, max_area)


def cmd_focus(a):
    os.makedirs(a.out_dir, exist_ok=True)
    for name in F.list_images(a.image_dir):
        mapped = _load_focus(a, name)
        if mapped is not None:
            F.write_visdrone(os.path.join(a.out_dir, _stem(name) + '.txt'), mapped)


def cmd_fuse(a):
    os.makedirs(a.out_dir, exist_ok=True)
    for name in F.list_images(a.image_dir):
        g = F.load_yolo_txt(os.path.join(a.global_det_dir, _stem(name) + '.txt'))
        g = np.empty((0, 6)) if g is None else g
        w, h = Image.open(os.path.join(a.image_dir, name)).size
        per_class = [[] for _ in range(F.NUM_CLASSES)]
        for d in g:
            per_class[int(round(d[0]))].append([*F.yolo_to_xyxy(d, w, h), d[5]])
        focus = _load_focus(a, name, max_area=a.max_area) or [[] for _ in range(F.NUM_CLASSES)]
        F.write_visdrone(os.path.join(a.out_dir, _stem(name) + '.txt'), F.fuse(per_class, focus, a.nms_iou))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('boxes')
    s.add_argument('--image_dir', required=True)
    s.add_argument('--tehnet', required=True)
    s.add_argument('--out_dir', required=True)
    s.add_argument('--box_size', type=int, default=96)
    s.add_argument('--nms_kernel', type=int, default=3)
    s.add_argument('--score_thresh', type=float, default=0.1)
    s.add_argument('--top_k', type=int, default=500)
    s.add_argument('--d_min', type=float, default=6.0)
    s.add_argument('--device', default='cuda')

    s = sub.add_parser('pack')
    s.add_argument('--image_dir', required=True)
    s.add_argument('--box_dir', required=True)
    s.add_argument('--out_img_dir', required=True)
    s.add_argument('--out_rec_dir', required=True)
    s.add_argument('--sr_scale', type=int, default=2, choices=[1, 2])
    s.add_argument('--edsr', default=None, help='EDSR x2 weights (needed when --sr_scale 2)')
    s.add_argument('--box_scale', type=float, default=1.5)
    s.add_argument('--device', default='cuda')

    s = sub.add_parser('global')
    s.add_argument('--image_dir', required=True)
    s.add_argument('--det_dir', required=True)
    s.add_argument('--out_dir', required=True)

    for name in ('focus', 'fuse'):
        s = sub.add_parser(name)
        s.add_argument('--image_dir', required=True)
        s.add_argument('--rec_dir', required=True)
        s.add_argument('--focus_img_dir', required=True)
        s.add_argument('--focus_det_dir', required=True)
        s.add_argument('--sr_scale', type=float, default=2)
        s.add_argument('--out_dir', required=True)
        if name == 'fuse':
            s.add_argument('--global_det_dir', required=True)
            s.add_argument('--max_area', type=float, default=1024, help='keep focus boxes with area <= this (px^2)')
            s.add_argument('--nms_iou', type=float, default=0.6)

    a = p.parse_args()
    {'boxes': cmd_boxes, 'pack': cmd_pack, 'global': cmd_global, 'focus': cmd_focus, 'fuse': cmd_fuse}[a.cmd](a)


if __name__ == '__main__':
    main()
