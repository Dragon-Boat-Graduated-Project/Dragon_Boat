#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
from mmpose.apis import init_model, inference_topdown

# ================== CLI arguments =================
parser = argparse.ArgumentParser()
parser.add_argument("--img-dir", required=True, help="Path to image folder")
parser.add_argument("--coco-json", required=True, help="Path to COCO bbox JSON")
parser.add_argument("--out-dir", required=True, help="Output folder to save results")
parser.add_argument(
    "--out-json", default="results_coco.json", help="Output JSON filename"
)
parser.add_argument(
    "--pose-config", default="configs/vitpose_custom.py", help="Pose model config"
)
parser.add_argument(
    "--pose-checkpoint", default="work_dirs/vitpose_custom/best_AP.pth",
    help="Pose model checkpoint"
)
args = parser.parse_args()

IMG_DIR = args.img_dir
JSON_FILE = args.coco_json
OUT_DIR = args.out_dir
OUT_JSON = os.path.join(OUT_DIR, args.out_json)
os.makedirs(OUT_DIR, exist_ok=True)

# =============== Load COCO bbox JSON ==============
with open(JSON_FILE, "r") as f:
    coco_bbox = json.load(f)

# img_id -> bbox list (xyxy)
img_bboxes = {}
for ann in coco_bbox["annotations"]:
    img_id = ann["image_id"]
    x, y, w, h = ann["bbox"]
    bbox_xyxy = [x, y, x + w, y + h]
    img_bboxes.setdefault(img_id, []).append(bbox_xyxy)

# img_id -> file_name
img_id2file = {
    img["id"]: os.path.join(IMG_DIR, img["file_name"]) for img in coco_bbox["images"]
}

# ================= Init pose model ================
pose_model = init_model(args.pose_config, args.pose_checkpoint, device="cuda:0")

# ============= Run top-down inference =============
ann_id = 1
coco_out = {
    "images": [],
    "annotations": [],
    "categories": [
        {
            "id": 1,
            "name": "person",
            "keypoints": [
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
            ],
        }
    ],
}

for img_id, bboxes in img_bboxes.items():
    img_file = img_id2file[img_id]

    results = inference_topdown(pose_model, img_file, bboxes=bboxes)

    for pred in results:
        instances = pred.pred_instances
        kpts = np.squeeze(instances.keypoints)  # shape: (num_kpts, 2/3)
        if kpts.size == 0:
            continue

        # Distinguish whether keypoints have scores
        if kpts.shape[1] == 3:
            coords = kpts[:, :2]
            scores = kpts[:, 2]
        else:
            coords = kpts
            scores = getattr(instances, "keypoint_scores", None)
            if scores is None:
                scores = np.ones(coords.shape[0], dtype=float)

        coords = coords.astype(float)

        # flatten scores
        scores = np.array(scores).flatten()
        if scores.shape[0] != coords.shape[0]:
            scores = np.ones(coords.shape[0], dtype=float)

        # COCO format keypoints
        kpts_flat = []
        xs, ys = [], []
        for i in range(coords.shape[0]):
            x, y = coords[i, 0], coords[i, 1]
            s = scores[i]
            v = 2 if s > 0.3 else 0
            kpts_flat += [x, y, v]
            xs.append(x)
            ys.append(y)

        # calculate bbox
        x_min, y_min = min(xs), min(ys)
        w_box, h_box = max(xs) - x_min, max(ys) - y_min

        # save COCO format
        coco_out["images"].append(
            {"id": img_id, "file_name": os.path.basename(img_file)}
        )
        coco_out["annotations"].append(
            {
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x_min, y_min, w_box, h_box],
                "area": w_box * h_box,
                "iscrowd": 0,
                "keypoints": kpts_flat,
                "num_keypoints": sum(v > 0 for v in kpts_flat[2::3]),
            }
        )
        ann_id += 1

# ================ Save COCO results ===============
with open(OUT_JSON, "w") as f:
    json.dump(coco_out, f, indent=2)

print(f"[OK] Inference finished! COCO results saved to {OUT_JSON}")
