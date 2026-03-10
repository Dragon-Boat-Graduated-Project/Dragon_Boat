#!/usr/bin/env python3
import os
import json
import cv2
import numpy as np
import argparse

# ======================= CLI ======================
parser = argparse.ArgumentParser(description="Visualize COCO keypoints with skeleton")
parser.add_argument("--img-dir", type=str, required=True, help="Image folder")
parser.add_argument("--coco-json", type=str, required=True, help="COCO JSON file")
parser.add_argument(
    "--out-dir", type=str, default="outputs_debug_vis", help="Output folder"
)
args = parser.parse_args()

IMG_DIR = args.img_dir
COCO_JSON = args.coco_json
OUT_DIR = args.out_dir

os.makedirs(OUT_DIR, exist_ok=True)

# ==================== Load COCO ===================
with open(COCO_JSON) as f:
    coco = json.load(f)

# key: image_id -> image info
img_map = {img["id"]: img for img in coco["images"]}

# skeleton: 1-based -> 0-based
skeleton_1_based = coco["categories"][0]["skeleton"]
SKELETON = [(i - 1, j - 1) for i, j in skeleton_1_based]

# ==================== Main Loop ===================
for ann in coco["annotations"]:
    img_info = img_map[ann["image_id"]]
    img_path = os.path.join(IMG_DIR, img_info["file_name"])
    img = cv2.imread(img_path)

    if img is None:
        print(f"[WARN] Image not found: {img_path}")
        continue

    kpts = np.array(ann["keypoints"]).reshape(-1, 3)

    # ---- Draw keypoints ----
    for i, (x, y, v) in enumerate(kpts):
        if v > 0:
            cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
            cv2.putText(
                img,
                str(i),
                (int(x) + 3, int(y) - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )

    # ---- Draw skeleton ----
    for i, j in SKELETON:
        if kpts[i][2] > 0 and kpts[j][2] > 0:
            cv2.line(
                img,
                (int(kpts[i][0]), int(kpts[i][1])),
                (int(kpts[j][0]), int(kpts[j][1])),
                (255, 0, 0),
                2,
            )

    # ---- Draw bbox ----
    x, y, w, h = ann["bbox"]
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)

    # ---- Save ----
    out_path = os.path.join(OUT_DIR, img_info["file_name"])
    cv2.imwrite(out_path, img)

print(f"[OK] Visualization saved to {OUT_DIR}")
