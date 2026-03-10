# predict_video.py
import os
import cv2
import torch
import numpy as np
from PIL import Image
import supervision as sv
from rfdetr import RFDETRBase
import argparse


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
    ap.add_argument("--model", required=True, help="path to checkpoint .pth")
    ap.add_argument("--video", required=True, help="input video path")
    ap.add_argument(
        "--save",
        choices=["video", "frames", "both"],
        default="both",
        help="what to save: annotated video, frames, or both",
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
        help="how many frames per second to SAVE.",
    )
    ap.add_argument(
        "--num_classes",
        type=int,
        default=1,
        help="number of classes used when building the model",
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

    predict_output = os.environ.get("PREDICT_OUTPUT", "Predict_Output")
    base_output = os.path.join(predict_output, video_name)

    img_dir = os.path.join(base_output, "images")
    label_dir = os.path.join(base_output, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    out_video = os.path.join(base_output, f"{video_name}_pred.mp4")

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
        vw = cv2.VideoWriter(out_video, fourcc, fps, (w, h))

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    frame_id = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out_frame = frame

        if frame_id % interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            det = model.predict(images=rgb, threshold=args.conf)

            labels = [f"paddle {conf:.2f}" for conf in det.confidence]
            anno = box_annotator.annotate(scene=rgb.copy(), detections=det)
            anno = label_annotator.annotate(scene=anno, detections=det, labels=labels)
            out_frame = cv2.cvtColor(anno, cv2.COLOR_RGB2BGR)

            img_name = f"frame_{frame_id:06d}.jpg"
            img_path = os.path.join(img_dir, img_name)
            cv2.imwrite(img_path, out_frame)

            if len(det.xyxy):
                img_width, img_height = out_frame.shape[1], out_frame.shape[0]
                label_path = os.path.join(
                    label_dir, os.path.splitext(img_name)[0] + ".txt"
                )
                with open(label_path, "w") as f:
                    for x1, y1, x2, y2 in det.xyxy:
                        x_center = (x1 + x2) / 2 / img_width
                        y_center = (y1 + y2) / 2 / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        f.write(
                            f"0 {x_center:.6f} {y_center:.6f} "
                            f"{width:.6f} {height:.6f}\n"
                        )

            processed += 1

        if vw is not None:
            vw.write(out_frame)

        if processed and processed % 50 == 0:
            print(f"Processed {processed} frames (read {frame_id+1}/{total})")

        frame_id += 1

    cap.release()
    if vw is not None:
        vw.release()

    print(f"Done.")
    print(f"  Images: {img_dir}")
    print(f"  Labels: {label_dir}")
    if args.save in ("video", "both"):
        print(f"  Video: {out_video}")


if __name__ == "__main__":
    main()
