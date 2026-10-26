# Modifications to TPH-YOLOv5

Vendored from https://github.com/cv516Buaa/tph-yolov5 (commit 052dfeb, GPL-3.0).
Only the files needed by `val.py` and `detect.py` are kept. Changes:

- `models/__init__.py`: GELU compatibility shim for loading the released checkpoints with torch >= 1.12.
- `models/experimental.py`: `torch.load(..., weights_only=False)` (the default changed in torch >= 2.6).
- `utils/general.py`, `utils/datasets.py`: `np.int` -> `np.int_` (removed in numpy >= 1.24).
- `utils/plots.py`: `font.getsize` fallback to `font.getbbox` (removed in Pillow >= 10); fall back to PIL's
  default font when `Arial.ttf` cannot be downloaded (offline machines).
- `requirements.txt`: reduced to the runtime dependencies of inference.
