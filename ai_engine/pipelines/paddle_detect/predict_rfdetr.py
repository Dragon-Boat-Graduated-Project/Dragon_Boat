#!/usr/bin/env python3
# Output structure:
# $PREDICT_OUTPUT/
# +-- video/           annotated prediction videos
# +-- images/          clean frames (no boxes, for YOLO training)
# +-- labels/          YOLO-format labels matching images

import os
import cv2
import torch
import supervision as sv
from rfdetr import RFDETRBase
import argparse

PRED_ROOT = os.environ.get("PREDICT_OUTPUT", "Predict_Output")


def load_core_model(model_path, device, num_classes=1, model_size="medium"):
    """
    Load RF-DETR checkpoint and map weights into the core detection model.
    """
    model = RFDETRBase(model_size=model_size, num_classes=num_classes, pretrained=False)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    weights = ckpt.get("model") or ckpt.get("model_ema") or ckpt
    core = model.model.model
    state_dict = core.state_dict()

    filtered = {}
    for k, v in weights.items():
        if k in state_dict and state_dict[k].shape == v.shape:
            filtered[k] = v

    core.load_state_dict(filtered, strict=False)
    core.to(device).eval()

    print(f"Loaded {len(filtered)}/{len(weights)} layers")

    try:
        model.optimize_for_inference()
    except Exception as e:
        print(f"[warn] Skip optimize_for_inference(): {e}")

    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to RF-DETR checkpoint .pth")
    ap.add_argument("--video", required=True, help="input video path")
    ap.add_argument(
        "--save",
        choices=["video", "frames", "both"],
        default="both",
        help="what to save: annotated video, clean frames+labels, or both",
    )
    ap.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="confidence threshold for detections",
    )
    ap.add_argument(
        "--target_fps",
        type=float,
        default=5.0,
        help="how many frames per second to SAVE (for frames/labels). "
        "Video keeps original fps.",
    )
    ap.add_argument(
        "--num_classes",
        type=int,
        default=1,
        help="number of classes when building RF-DETR",
    )
    ap.add_argument(
        "--model_size",
        default="medium",
        choices=["nano", "small", "base", "medium", "large"],
        help="RF-DETR backbone size",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = load_core_model(
        args.model, device, num_classes=args.num_classes, model_size=args.model_size
    )

    video_name = os.path.splitext(os.path.basename(args.video))[0]

    # Global output dirs
    video_root = os.path.join(PRED_ROOT, "video")
    images_root = os.path.join(PRED_ROOT, "images")
    labels_root = os.path.join(PRED_ROOT, "labels")
    os.makedirs(video_root, exist_ok=True)
    os.makedirs(images_root, exist_ok=True)
    os.makedirs(labels_root, exist_ok=True)

    out_video = os.path.join(video_root, f"{video_name}_pred.mp4")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video info: {w}x{h}, {fps:.1f} FPS, {total} frames")

    if fps > 0 and args.target_fps > 0:
        interval = max(int(round(fps / args.target_fps)), 1)
    else:
        interval = 1
    print(f"Sampling interval = {interval} frames (~{args.target_fps} fps)")

    vw = None
    if args.save in ("video", "both"):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(out_video, fourcc, fps if fps > 0 else 25.0, (w, h))

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    frame_id = 0
    saved_frames = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        clean_bgr = frame_bgr.copy()
        annotated_bgr = frame_bgr

        run_model = frame_id % interval == 0

        if run_model:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            det = model.predict(images=rgb, threshold=args.conf)

            labels = [f"paddle {conf:.2f}" for conf in det.confidence]
            anno = box_annotator.annotate(scene=rgb.copy(), detections=det)
            anno = label_annotator.annotate(scene=anno, detections=det, labels=labels)
            annotated_bgr = cv2.cvtColor(anno, cv2.COLOR_RGB2BGR)

            if args.save in ("frames", "both"):
                stem = f"{video_name}_frame_{frame_id:06d}"
                img_path = os.path.join(images_root, stem + ".jpg")
                label_path = os.path.join(labels_root, stem + ".txt")

                cv2.imwrite(img_path, clean_bgr)

                if len(det.xyxy):
                    img_w, img_h = clean_bgr.shape[1], clean_bgr.shape[0]
                    with open(label_path, "w") as f:
                        for x1, y1, x2, y2 in det.xyxy:
                            x_center = (x1 + x2) / 2 / img_w
                            y_center = (y1 + y2) / 2 / img_h
                            width = (x2 - x1) / img_w
                            height = (y2 - y1) / img_h
                            f.write(
                                f"0 {x_center:.6f} {y_center:.6f} "
                                f"{width:.6f} {height:.6f}\n"
                            )

                saved_frames += 1
                if saved_frames % 100 == 0:
                    print(
                        f"Saved {saved_frames} frames "
                        f"(read {frame_id+1}/{total}) from {video_name}"
                    )

        if vw is not None:
            vw.write(annotated_bgr)

        frame_id += 1

    cap.release()
    if vw is not None:
        vw.release()

    print(f"Done: {video_name}")
    if args.save in ("frames", "both"):
        print(f"  Clean images: {images_root}")
        print(f"  Labels      : {labels_root}")
        print(f"  Saved frames: {saved_frames}")
    if args.save in ("video", "both"):
        print(f"  Video       : {out_video}")


if __name__ == "__main__":
    main()
