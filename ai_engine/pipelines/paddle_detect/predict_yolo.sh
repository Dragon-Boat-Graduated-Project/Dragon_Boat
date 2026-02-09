#!/bin/bash
set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT in .env}"

cd "$PROJECT_ROOT"

MODEL="runs/detect/rf_pseudo_paddle_v2/weights/best.pt"
SRC_DIR="${1:?Usage: predict_yolo.sh <video_path>}"
OUTROOT="${PREDICT_OUTPUT:-$PROJECT_ROOT/Predict_Output}/yolo_pred"
NAME="paddle"

yolo detect predict \
  model="$MODEL" \
  source="$SRC_DIR" \
  imgsz=640 \
  conf=0.01 \
  device=0 \
  workers=8 \
  save=True \
  save_conf=True \
  save_txt=True \
  project="$OUTROOT" \
  name="$NAME" \
  exist_ok=True

echo "Done. Results: $OUTROOT/$NAME"
