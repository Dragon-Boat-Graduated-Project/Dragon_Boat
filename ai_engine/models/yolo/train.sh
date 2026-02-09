#!/bin/bash
set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT in .env}"
: "${YOLO_DATASET:?Set YOLO_DATASET in .env}"

MODEL="yolo11s.pt"
DATA="$YOLO_DATASET/data.yaml"

PROJECT="runs/detect"
NAME="rf_pseudo_paddle_v2"

cd "$PROJECT_ROOT"

yolo detect train \
  model="$MODEL" \
  data="$DATA" \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  patience=15 \
  single_cls=True \
  project="$PROJECT" \
  name="$NAME" \
  exist_ok=True
