#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import mmcv
from mmdet.apis import DetInferencer
import cv2


def main(args):
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)

    # ============= Initialize Faster-RCNN =============
    inferencer = DetInferencer(
        model=args.det_cfg, weights=args.det_weights, device=args.device
    )

    # ================== Picture list ==================
    img_list = sorted(
        [
            f
            for f in os.listdir(args.img_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
    )

    center_bboxes = {}

    for fname in tqdm(img_list, desc="Detect center person"):
        img_path = os.path.join(args.img_dir, fname)
        img = mmcv.imread(img_path)
        if img is None:
            print(f"[WARNING] Cannot read {img_path}, skipped")
            continue
        h, w = img.shape[:2]
        img_center = np.array([w / 2, h / 2])

        # ==================== Corollary ===================
        result = inferencer(img_path)
        pred = result["predictions"][0]

        bboxes = np.array(pred["bboxes"])
        scores = np.array(pred["scores"])
        labels = np.array(pred["labels"])

       # Take only person
        mask = labels == 0
        person_boxes = bboxes[mask]
        person_scores = scores[mask]

        if len(person_boxes) == 0:
            continue

        # Filter high score boxes
        conf_mask = person_scores > args.score_thr
        if conf_mask.sum() == 0:
            conf_mask = person_scores >= person_scores.max()
        person_boxes = person_boxes[conf_mask]

        # Find the middle person
        def center_distance(box):
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            return np.linalg.norm(np.array([cx, cy]) - img_center)

        idx = np.argmin([center_distance(b) for b in person_boxes])
        center_box = person_boxes[idx].tolist()
        center_bboxes[fname] = center_box

        # ================== Visualization =================
        vis_img = img.copy()
        for box in person_boxes.astype(int):
            color = (
                (0, 255, 0)
                if np.array_equal(box, person_boxes[idx].astype(int))
                else (0, 0, 255)
            )
            cv2.rectangle(vis_img, (box[0], box[1]),
                          (box[2], box[3]), color, 2)
        # center point
        cv2.circle(vis_img, tuple(img_center.astype(int)), 5, (255, 0, 0), -1)
        cv2.imwrite(os.path.join(args.vis_dir, fname), vis_img)

    # Archive JSON
    with open(args.out_json, "w") as f:
        json.dump(center_bboxes, f, indent=2)

    print(f"[INFO] Center bboxes saved at {args.out_json}")
    print(f"[INFO] Visualization images saved at {args.vis_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str, default="data/dataset/images")
    parser.add_argument(
        "--det_cfg", type=str, default="configs/faster-rcnn_r50_fpn_1x_coco.py"
    )
    parser.add_argument(
        "--det_weights", type=str, default="weights/faster_rcnn_r50_fpn_coco.pth"
    )
    parser.add_argument(
        "--out-json", type=str, default="data/dataset/annotations/center_bboxes.json"
    )
    parser.add_argument("--score-thr", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vis-dir", type=str, default="outputs_center_vis")
    args = parser.parse_args()

    main(args)
