#!/usr/bin/env python3
import os
import json
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("img_dir")
parser.add_argument("coco_json")
parser.add_argument("out_dir")
args = parser.parse_args()

IMG_DIR = args.img_dir
JSON_FILE = args.coco_json
OUT_DIR = args.out_dir

os.makedirs(OUT_DIR, exist_ok=True)

with open(JSON_FILE) as f:
    data = json.load(f)

kp_map = {}
for ann in data["annotations"]:
    kp_map.setdefault(ann["image_id"], []).append(ann["keypoints"])

# COCO skeleton (17 keypoints)
skeleton = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

img_map = {img["id"]: img["file_name"] for img in data["images"]}

for img_id, file_name in img_map.items():
    p = os.path.join(IMG_DIR, file_name)
    im = cv2.imread(p)
    if im is None:
        continue

    for kp in kp_map.get(img_id, []):
        pts, vis = [], []
        for i in range(len(kp) // 3):
            x = int(kp[i * 3])
            y = int(kp[i * 3 + 1])
            v = kp[i * 3 + 2]
            pts.append((x, y))
            vis.append(v)

        # draw skeleton
        for a, b in skeleton:
            if vis[a] > 0 and vis[b] > 0:
                cv2.line(im, pts[a], pts[b], (0, 255, 0), 2)

        # draw keypoints
        for (x, y), v in zip(pts, vis):
            if v > 0:
                cv2.circle(im, (x, y), 3, (0, 0, 255), -1)

    idx = int(file_name.split("_")[-1].split(".")[0])
    out_path = os.path.join(OUT_DIR, f"frame_{idx:06d}.jpg")
    cv2.imwrite(out_path, im)
