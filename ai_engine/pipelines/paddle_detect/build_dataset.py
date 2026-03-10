#!/usr/bin/env python
import os
import shutil
import random
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "."))

# Source: RF-DETR predicted frames / labels
IMG_SRC = Path(os.environ.get("PREDICT_OUTPUT", PROJECT_ROOT / "Predict_Output")) / "images"
LBL_SRC = Path(os.environ.get("PREDICT_OUTPUT", PROJECT_ROOT / "Predict_Output")) / "labels"

# Destination: YOLO dataset
DST_ROOT = Path(os.environ.get("YOLO_DATASET", PROJECT_ROOT / "dragonboat_yolo"))
IMG_TRAIN = DST_ROOT / "images" / "train"
IMG_VAL = DST_ROOT / "images" / "val"
LBL_TRAIN = DST_ROOT / "labels" / "train"
LBL_VAL = DST_ROOT / "labels" / "val"


def main():
    for d in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
        d.mkdir(parents=True, exist_ok=True)

    print("IMG_SRC =", IMG_SRC)
    print("LBL_SRC =", LBL_SRC)

    stems = []

    for txt in LBL_SRC.glob("*.txt"):
        stem = txt.stem
        jpg = IMG_SRC / f"{stem}.jpg"
        if jpg.exists():
            stems.append(stem)
        else:
            print(f"[warn] No matching image: {jpg}")

    if not stems:
        raise SystemExit(
            "No paired jpg+txt samples found. Check Predict_Output/images and /labels."
        )

    print(f"Total labeled samples: {len(stems)}")

    random.seed(0)
    random.shuffle(stems)

    # 5:1 train:val split
    split_idx = int(len(stems) * 5 / 6)
    train_stems = stems[:split_idx]
    val_stems = stems[split_idx:]

    print(f"train: {len(train_stems)}, val: {len(val_stems)}")

    def copy_split(split_stems, img_dst, lbl_dst):
        for stem in split_stems:
            shutil.copy2(IMG_SRC / f"{stem}.jpg", img_dst / f"{stem}.jpg")
            shutil.copy2(LBL_SRC / f"{stem}.txt", lbl_dst / f"{stem}.txt")

    copy_split(train_stems, IMG_TRAIN, LBL_TRAIN)
    copy_split(val_stems, IMG_VAL, LBL_VAL)

    print("Copy complete")
    print(f"images/train -> {IMG_TRAIN}")
    print(f"images/val   -> {IMG_VAL}")
    print(f"labels/train -> {LBL_TRAIN}")
    print(f"labels/val   -> {LBL_VAL}")


if __name__ == "__main__":
    main()
