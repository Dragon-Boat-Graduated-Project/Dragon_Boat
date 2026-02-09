import os
from pathlib import Path
import cv2

root = Path(os.environ.get("DATASET_PATH", "DataSet"))

# Collect all .mp4 video paths
video_paths = list(root.rglob("*.mp4"))

target_fps = 10  # desired frames per second

for video_path in video_paths:
    # Create directory to save extracted frames
    save_dir = video_path.parent / f"{video_path.stem}_frames"
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    interval = max(1, int(round(fps / target_fps)))

    frame_count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            frame_file = save_dir / f"frame_{saved:05d}.jpg"
            cv2.imwrite(str(frame_file), frame)
            saved += 1

        frame_count += 1

    cap.release()
    print(
        f"Extracted {saved} frames from {video_path.relative_to(root)} -> {save_dir.relative_to(root)}"
    )
