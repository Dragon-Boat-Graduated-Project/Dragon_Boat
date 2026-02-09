#!/bin/bash

: "${SOURCE_VIDEOS:?Set SOURCE_VIDEOS in .env}"
: "${PREDICT_OUTPUT:?Set PREDICT_OUTPUT in .env}"
: "${MODEL_CHECKPOINT:?Set MODEL_CHECKPOINT in .env}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for f in "$SOURCE_VIDEOS"/*.mp4; do
  [ -e "$f" ] || continue
  base="$(basename "$f" .mp4)"
  out="$PREDICT_OUTPUT/$base"
  if [[ -d "$out" ]]; then
    echo "Skip $f (output dir exists: $out)"
    continue
  fi
  echo "Processing $f ..."
  python3 "$SCRIPT_DIR/predict_video.py" \
    --model "$MODEL_CHECKPOINT" \
    --video "$f" \
    --save both \
    --conf 0.25 \
    --target_fps 5
done
