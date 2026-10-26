# TEHNet-FocusDet

Code for the paper

**Rethinking Focus-and-Detect for Aerial Small Object Detection Under a Frozen-Detector Protocol: A Translation-Equivariance-Guided Approach**
Chuwei Li, Zhilong Zhang, Ping Zhong. *Remote Sensing* 2026, 18(21), 3660. [doi:10.3390/rs18213660](https://doi.org/10.3390/rs18213660)

This repository contains the complete inference pipeline on VisDrone with a frozen
TPH-YOLOv5: the translation-equivariant heatmap network (TEHNet), heatmap decoding,
focus-image construction (region merging, EDSR x2 super-resolution, image packing),
inverse coordinate mapping, fusion and evaluation. The detector is used off the shelf,
without modification or retraining.

```
image ─┬─ global branch: TPH-YOLOv5 on the full image ─────────────────────────────┐
       └─ focus branch:  TEHNet heatmap → peaks → 96×96 boxes                        │
                         → 1.5× expansion + greedy merging → EDSR ×2 per region      │
                         → packing into one focus image → TPH-YOLOv5                 │
                         → inverse mapping, keep boxes with area ≤ 32² ──────────────┤
                                                              per-class NMS (0.6) ◄──┘
```

## Installation

Python 3.11, PyTorch 2.2.2, CUDA 11.8:

```bash
conda env create -f environment.yml
conda activate tehnet-focusdet
```

or, in an existing Python 3.11 environment:

```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

> Note: the pip step of `environment.yml` downloads a ~2.5 GB PyTorch wheel **without
> a progress bar** — depending on your network it may sit silently for 10–30 minutes
> after `Executing transaction: done`. It is still downloading, not stuck.

## Weights

`tehnet_best.pth` (ours) is already included in this repository under `weights/`.
The other two files are put into `weights/` after downloading:

- All three weights in one archive: [Baidu Drive](https://pan.baidu.com/s/1cL3pQ0sC8oM5e-9jbku7yQ?pwd=fjpa) (pw: `fjpa`)
- Or from the original sources:

| File | Size | MD5 | Source |
|---|---|---|---|
| `tehnet_best.pth` | 14.8 MB | `dfff6cb48d15258f5732413a0516255a` | included in this repository |
| `yolov5l-xs-2.pt` | 140 MB | `be02bb8aa695fc27a645814d8e0efa24` | official TPH-YOLOv5 release: [Google Drive](https://drive.google.com/file/d/1VmORvxNtvMVMvmY7cCwvp0BoL6L3RGiq/view) / [Baidu (pw: vffz)](https://pan.baidu.com/s/19S84EevP86yJIvnv9KYXDA) |
| `visdrone_x2.pt` | 5.5 MB | `31620c728f68a2332b777b772dc03400` | EDSR ×2 released by GLSAN: [`models/visdrone_x2.pt`](https://github.com/dengsutao/glsan/raw/main/models/visdrone_x2.pt) |

## Data

Download VisDrone2019-DET `val` and `test-dev` from the
[VisDrone repository](https://github.com/VisDrone/VisDrone-Dataset) and arrange them as

```
data/VisDrone/
├── VisDrone2019-DET-val/{images,annotations}/
└── VisDrone2019-DET-test-dev/{images,annotations}/
```

(or set `VISDRONE_ROOT=/path/to/VisDrone`). No other preprocessing is needed.

## Evaluation

```bash
bash run_focus_det.sh val       --sr true    # full method
bash run_focus_det.sh val       --sr false   # without super-resolution
bash run_focus_det.sh test-dev  --sr true
bash run_focus_det.sh test-dev  --sr false
```

Results are written to `results/<subset>/<sr|nosr>/summary.md` (global branch, focus
branch, fused result). Detections of the global branch and the TEHNet boxes are cached
in `results/<subset>/` and shared by both settings; remove the directory to recompute.

mAP@0.5:0.95 and mAP@0.5 follow the VisDrone-DET toolkit (Python port, `eval/`);
AP/AR for small, medium and large objects extend it with MS COCO semantics.

## Results

VisDrone-val, TPH-YOLOv5 (`yolov5l-xs-2.pt`, frozen, 1996 px with TTA), produced by this code:

| Method | mAP@0.5:0.95 | mAP@0.5 | AP_S | AP_M | AP_L | AR_S |
|---|---|---|---|---|---|---|
| baseline | 40.95 | 65.92 | 32.49 | 52.27 | 63.67 | 49.18 |
| ours w/o SR | 41.37 | 66.84 | 33.05 | 52.33 | 63.59 | 50.92 |
| ours | **41.92** | **67.66** | **33.78** | **52.56** | 63.60 | **51.55** |

## Single-image demo

```bash
python focusdet/demo.py path/to/image.jpg                # add --sr_scale 1 to disable SR
```

writes the heatmap, the 96×96 boxes, the merged regions, the focus image, the
detections on it, a baseline-vs-fused comparison and per-stage timing to `demo_out/`.

## Repository structure

```
run_focus_det.sh          evaluation pipeline
focusdet/tehnet.py        TEHNet and heatmap decoding
focusdet/focus.py         focus-image construction, inverse mapping, fusion
focusdet/pipeline.py      command-line steps used by run_focus_det.sh
focusdet/demo.py          single-image demo
eval/                     VisDrone evaluation
third_party/tph_yolov5/   TPH-YOLOv5 inference code (see MODIFICATIONS.md)
third_party/ufp/          box expansion, merging and packing from UFPMP-Det
third_party/edsr/         EDSR definition from GLSAN
```

## Acknowledgements

[TPH-YOLOv5](https://github.com/cv516Buaa/tph-yolov5) (detector),
[UFPMP-Det](https://github.com/PuAnysh/UFPMP-Det) (box expansion, merging, packing, inverse mapping),
[GLSAN](https://github.com/dengsutao/glsan) (EDSR super-resolution weights),
[VisDrone](https://github.com/VisDrone/VisDrone-Dataset) (dataset and evaluation toolkit).

## License

GPL-3.0, as required by the included TPH-YOLOv5 code. `third_party/ufp` is
Apache-2.0 (UFPMP-Det).

## Citation

```bibtex
@article{li2026tehnet,
  title   = {Rethinking Focus-and-Detect for Aerial Small Object Detection Under a Frozen-Detector Protocol: A Translation-Equivariance-Guided Approach},
  author  = {Li, Chuwei and Zhang, Zhilong and Zhong, Ping},
  journal = {Remote Sensing},
  volume  = {18},
  number  = {21},
  pages   = {3660},
  year    = {2026},
  doi     = {10.3390/rs18213660}
}
```

Contact: Chuwei Li (iammusili@qq.com)
