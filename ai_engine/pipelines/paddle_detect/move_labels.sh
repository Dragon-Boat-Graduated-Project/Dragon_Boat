#!/bin/bash
set -euo pipefail

: "${PREDICT_OUTPUT:?Set PREDICT_OUTPUT in .env}"
: "${YOLO_DATASET:?Set YOLO_DATASET in .env}"

SRC_ROOT="$PREDICT_OUTPUT"
DST_ROOT="$YOLO_DATASET"

mkdir -p "$DST_ROOT/images/train" "$DST_ROOT/labels/train"

for dir in "$SRC_ROOT"/*/; do
    stem="$(basename "$dir")"
    img_dir="$dir/images"
    lbl_dir="$dir/labels"

    echo "Processing $stem ... "
    [ -d "$img_dir" ] || continue

    for img in "$img_dir"/*.jpg; do
        [ -e "$img" ] || continue
        base="$(basename "$img")"
        name_no_ext="${base%.jpg}"
        label="$lbl_dir/$name_no_ext.txt"

        [ -f "$label" ] || continue

        new_base="${stem}_${base}"
        new_label="${stem}_${name_no_ext}.txt"

        cp "$img"   "$DST_ROOT/images/train/$new_base"
        cp "$label" "$DST_ROOT/labels/train/$new_label"
    done
done
