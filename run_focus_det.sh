#!/usr/bin/env bash
# TEHNet-FocusDet on VisDrone with a frozen TPH-YOLOv5.
#
# Usage:   bash run_focus_det.sh [val|test-dev] [--sr true|false]
# Env:     VISDRONE_ROOT (default: ./data/VisDrone), PYTHON (default: python)
# Output:  results/<subset>/<sr|nosr>/summary.md
#
# Detections and TEHNet boxes are cached under results/<subset>/; delete that
# directory after changing weights or code.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBSET=val
SR=true
while [ $# -gt 0 ]; do
    case "$1" in
        --sr)   SR="$2"; shift 2 ;;
        --sr=*) SR="${1#*=}"; shift ;;
        val|test-dev) SUBSET="$1"; shift ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done
case "$SR" in
    true)  SR_SCALE=2; TAG=sr ;;
    false) SR_SCALE=1; TAG=nosr ;;
    *) echo "--sr must be true or false" >&2; exit 1 ;;
esac

PY="${PYTHON:-python}"
DATA="${VISDRONE_ROOT:-$REPO/data/VisDrone}/VisDrone2019-DET-$SUBSET"
IMG="$DATA/images"
ANN="$DATA/annotations"
TPH="$REPO/third_party/tph_yolov5"
DET_W="$REPO/weights/yolov5l-xs-2.pt"
TEHNET_W="$REPO/weights/tehnet_best.pth"
EDSR_W="$REPO/weights/visdrone_x2.pt"

for f in "$IMG" "$ANN" "$DET_W" "$TEHNET_W"; do
    [ -e "$f" ] || { echo "not found: $f" >&2; exit 1; }
done
[ "$SR_SCALE" = 1 ] || [ -e "$EDSR_W" ] || { echo "not found: $EDSR_W" >&2; exit 1; }

CACHE="$REPO/results/$SUBSET"
OUT="$CACHE/$TAG"
mkdir -p "$OUT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

nonempty() { [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; }

# Frozen TPH-YOLOv5, official test setting (1996 px, TTA). The batch size is
# kept fixed because rectangular batching makes the letterbox shape depend on it.
detect() {  # <image_dir> <batch_size> <out_label_dir>
    cat > "$TMP/data.yaml" <<EOF
path: $1
train: .
val: .
nc: 10
names: ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
EOF
    (cd "$TPH" && "$PY" val.py --data "$TMP/data.yaml" --weights "$DET_W" --img 1996 --augment \
        --batch-size "$2" --task val --save-txt --save-conf --project "$TMP" --name det --exist-ok)
    mkdir -p "$3"
    find "$TMP/det/labels" -name '*.txt' -exec cp -t "$3" {} +
    rm -rf "$TMP/det" "$1.cache"
}

evaluate() {  # <det_dir> <log>
    "$PY" "$REPO/eval/visdrone_eval.py" "$1" "$ANN" "$IMG" > "$2"
}

echo "== [1/5] global branch"
nonempty "$CACHE/global_raw" || detect "$IMG" 4 "$CACHE/global_raw"
"$PY" "$REPO/focusdet/pipeline.py" global --image_dir "$IMG" --det_dir "$CACHE/global_raw" --out_dir "$OUT/global"
evaluate "$OUT/global" "$OUT/eval_global.txt"

echo "== [2/5] TEHNet focus boxes"
nonempty "$CACHE/tehnet_boxes" || "$PY" "$REPO/focusdet/pipeline.py" boxes \
    --image_dir "$IMG" --tehnet "$TEHNET_W" --out_dir "$CACHE/tehnet_boxes"

echo "== [3/5] focus images (sr_scale=$SR_SCALE)"
nonempty "$OUT/focus_recs" || "$PY" "$REPO/focusdet/pipeline.py" pack \
    --image_dir "$IMG" --box_dir "$CACHE/tehnet_boxes" --sr_scale "$SR_SCALE" --edsr "$EDSR_W" \
    --out_img_dir "$OUT/focus_images" --out_rec_dir "$OUT/focus_recs"

echo "== [4/5] focus branch"
nonempty "$OUT/focus_raw" || detect "$OUT/focus_images" 2 "$OUT/focus_raw"
FOCUS_ARGS=(--image_dir "$IMG" --rec_dir "$OUT/focus_recs" --focus_img_dir "$OUT/focus_images"
            --focus_det_dir "$OUT/focus_raw" --sr_scale "$SR_SCALE")
"$PY" "$REPO/focusdet/pipeline.py" focus "${FOCUS_ARGS[@]}" --out_dir "$OUT/focus"
evaluate "$OUT/focus" "$OUT/eval_focus.txt"

echo "== [5/5] fusion"
"$PY" "$REPO/focusdet/pipeline.py" fuse "${FOCUS_ARGS[@]}" --global_det_dir "$CACHE/global_raw" --out_dir "$OUT/fused"
evaluate "$OUT/fused" "$OUT/eval_fused.txt"

KEYS=("AP@0.50:0.95(all)" "AP@0.50(all)" "AP@0.50:0.95(small)" "AP@0.50:0.95(medium)"
      "AP@0.50:0.95(large)" "AR@0.50:0.95(small)" "AR@0.50:0.95(medium)" "AR@0.50:0.95(large)")
metric() { grep -F "$1=" "$2" | cut -d= -f2; }
row() {  # <label> <log>
    printf '| %s |' "$1"
    for k in "${KEYS[@]}"; do awk -v v="$(metric "$k" "$2")" 'BEGIN{printf " %.2f |", v*100}'; done
    echo
}
{
    echo "# VisDrone-$SUBSET, TPH-YOLOv5 (frozen), sr=$SR"
    echo
    echo "| | mAP@0.5:0.95 | mAP@0.5 | AP_S | AP_M | AP_L | AR_S | AR_M | AR_L |"
    echo "|---|---|---|---|---|---|---|---|---|"
    row "global branch (baseline)" "$OUT/eval_global.txt"
    row "focus branch only" "$OUT/eval_focus.txt"
    row "**fused (ours)**" "$OUT/eval_fused.txt"
    printf '| Δ (fused − baseline) |'
    for k in "${KEYS[@]}"; do
        awk -v a="$(metric "$k" "$OUT/eval_fused.txt")" -v b="$(metric "$k" "$OUT/eval_global.txt")" 'BEGIN{printf " %+.2f |", (a-b)*100}'
    done
    echo
} > "$OUT/summary.md"
cat "$OUT/summary.md"
