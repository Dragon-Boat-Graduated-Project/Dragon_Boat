#!/bin/bash
set -euo pipefail

: "${YOLO_DATASET:?Set YOLO_DATASET in .env}"

ROOT="$YOLO_DATASET"

# Video IDs to use as validation set (~1/6 of total)
VAL_IDS=(A04 A09 B04 C03)

mkdir -p "$ROOT/images/val" "$ROOT/labels/val"

for id in "${VAL_IDS[@]}"; do
    echo "Moving $id to val ..."
    for img in "$ROOT/images/train/${id}_"*.jpg; do
        [ -e "$img" ] || continue

        base="$(basename "$img")"
        stem="${base%.jpg}"

        mv "$ROOT/images/train/$base"    "$ROOT/images/val/$base"
        mv "$ROOT/labels/train/$stem.txt" "$ROOT/labels/val/$stem.txt"
    done
done

echo "Done splitting train / val."
