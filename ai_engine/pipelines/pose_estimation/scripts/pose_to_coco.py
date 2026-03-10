#!/usr/bin/env python3
import os
import json
import mmcv
import numpy as np
from mmpose.apis import init_model, inference_topdown
import argparse

# ======================= CLI ======================
parser = argparse.ArgumentParser()
parser.add_argument("--img-dir", required=True)
parser.add_argument("--bbox-json", required=True)
parser.add_argument("--pose-config", required=True)
parser.add_argument("--pose-checkpoint", required=True)
parser.add_argument("--out-train", required=True)
parser.add_argument("--out-val", required=True)
parser.add_argument("--out-test", required=True)
parser.add_argument("--train-ratio", type=float, default=0.7)
parser.add_argument("--val-ratio", type=float, default=0.15)
parser.add_argument("--debug-dir", default=None)
args = parser.parse_args()

IMG_DIR = args.img_dir
CENTER_JSON = args.bbox_json
POSE_CFG = args.pose_config
POSE_CKPT = args.pose_checkpoint
OUT_TRAIN = args.out_train
OUT_VAL = args.out_val
OUT_TEST = args.out_test
TRAIN_RATIO = args.train_ratio
VAL_RATIO = args.val_ratio

# ================= COCO keypoints =================
COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
COCO_SKELETON = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
]


def new_coco():
    return {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "keypoints": COCO_KEYPOINTS,
                "skeleton": COCO_SKELETON,
            }
        ],
    }


# ==================== Load bbox ===================
with open(CENTER_JSON) as f:
    center_bboxes = json.load(f)

# =================== Init model ===================
pose_model = init_model(POSE_CFG, POSE_CKPT, device="cuda:0")
pose_model.eval()

img_list = sorted(center_bboxes.keys())
n = len(img_list)

num_train = int(n * TRAIN_RATIO)
num_val = int(n * VAL_RATIO)

coco_train = new_coco()
coco_val = new_coco()
coco_test = new_coco()

train_img_id = val_img_id = test_img_id = 1
train_ann_id = val_ann_id = test_ann_id = 1

# ==================== Main loop ===================
for idx, img_name in enumerate(img_list):
    if idx < num_train:
        split = "train"
        target = coco_train
    elif idx < num_train + num_val:
        split = "val"
        target = coco_val
    else:
        split = "test"
        target = coco_test

    img_path = os.path.join(IMG_DIR, img_name)
    img = mmcv.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]
    x1, y1, x2, y2 = center_bboxes[img_name]
    det_bbox = np.array([[x1, y1, x2, y2]], dtype=np.float32)

    pose_results = inference_topdown(
        pose_model, img_path, bboxes=det_bbox, bbox_format="xyxy"
    )

    if len(pose_results) == 0:
        continue

    kpts = pose_results[0].pred_instances.keypoints
    if hasattr(kpts, "cpu"):
        kpts = kpts.cpu().numpy()
    if kpts.ndim == 3:
        kpts = kpts[0]

    if kpts.shape[1] == 2:
        kpts = np.concatenate([kpts, np.full((17, 1), 2)], axis=1)

    kpts_flat = []
    num_visible = 0
    for x, y, v in kpts:
        x = float(np.clip(x, 0, w - 1))
        y = float(np.clip(y, 0, h - 1))
        v = int(v)
        kpts_flat.extend([x, y, v])
        if v > 0:
            num_visible += 1

    if split == "train":
        img_id, ann_id = train_img_id, train_ann_id
        train_img_id += 1
        train_ann_id += 1
    elif split == "val":
        img_id, ann_id = val_img_id, val_ann_id
        val_img_id += 1
        val_ann_id += 1
    else:
        img_id, ann_id = test_img_id, test_ann_id
        test_img_id += 1
        test_ann_id += 1

    target["images"].append(
        {"id": img_id, "file_name": img_name, "width": w, "height": h}
    )

    target["annotations"].append(
        {
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "area": float((x2 - x1) * (y2 - y1)),
            "iscrowd": 0,
            "keypoints": kpts_flat,
            "num_keypoints": num_visible,
        }
    )

# ====================== Save ======================
os.makedirs(os.path.dirname(OUT_TRAIN), exist_ok=True)
with open(OUT_TRAIN, "w") as f:
    json.dump(coco_train, f, indent=2)
with open(OUT_VAL, "w") as f:
    json.dump(coco_val, f, indent=2)
with open(OUT_TEST, "w") as f:
    json.dump(coco_test, f, indent=2)
print("[OK] Generated train / val / test COCO JSON")
