"""VisDrone-DET evaluation.

AP@0.50:0.95 and AP@0.50 (all objects) follow the VisDrone-DET toolkit
(ignored regions, maxDets = 500). AP/AR for small / medium / large objects
extend it with MS COCO semantics (see size_eval.py).

Usage:
    python eval/visdrone_eval.py <det_dir> <gt_annotation_dir> <image_dir>
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visdrone_eval_utils import find_image_list, save_anno_res, calc_accuracy_full  # noqa: E402
from size_eval import SizeEvaluator, from_visdrone_arrays  # noqa: E402


def main():
    p = argparse.ArgumentParser(description='VisDrone-DET evaluation')
    p.add_argument('det_dir', help='detections, one VisDrone-format txt per image')
    p.add_argument('gt_dir', help='VisDrone annotation directory')
    p.add_argument('image_dir', help='image directory (for image sizes)')
    a = p.parse_args()

    det_dir, gt_dir, image_dir = (os.path.abspath(x) for x in (a.det_dir, a.gt_dir, a.image_dir))
    names = find_image_list(gt_dir)
    allgt, alldet = save_anno_res(gt_dir, det_dir, image_dir, len(names), names)

    max_dets = [10, 100, 300, 500, 1000, 10000]
    k = max_dets.index(500)
    AP, _, classes = calc_accuracy_full(len(names), allgt, alldet, max_dets, sml_mode=False, verbose=True)
    c = np.array(classes) - 1
    ap = np.mean(AP[c, :, k]) / 100
    ap50 = np.mean(AP[c, 0, k]) / 100

    ev = SizeEvaluator(profile='coco', max_dets=500, class_average='macro',
                       ap_method='coco101', boundary='left_closed')
    summ = ev.evaluate(*from_visdrone_arrays(allgt, alldet))['summary']
    sizes = ('small', 'medium', 'large')

    print()
    print(f'AP@0.50:0.95(all)={ap:.4f}')
    print(f'AP@0.50(all)={ap50:.4f}')
    for s in sizes:
        print(f'AP@0.50:0.95({s})={summ[s]["AP"]:.4f}')
    for s in sizes:
        print(f'AR@0.50:0.95({s})={summ[s]["AR"]:.4f}')


if __name__ == '__main__':
    main()
