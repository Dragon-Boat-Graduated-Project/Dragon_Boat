#!/bin/bash
set -e

# =============================================================
# Dragon Boat Pose Estimation Pipeline
# Uses ViTPose (MMPose) for 2D human pose estimation
#
# Usage:
#   cd ai_engine/pipelines/pose_estimation
#   bash run_pipeline.sh
#
# Prerequisites:
#   - MMPose environment set up (see required.md)
#   - mmpose source cloned into this directory
#   - Videos placed in videos/
#   - Faster R-CNN weights in weights/
# =============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

#####################################
# Config
#####################################
FPS=30

# Official ViTPose (for pseudo-label generation)
POSE_CFG_PSEUDO="mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192.py"
POSE_PRETRAIN_PSEUDO="mmpose/weights/td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192-0b8234ea_20230407.pth"

# Custom ViTPose (fine-tuned)
POSE_CFG_CUSTOM="configs/vitpose_custom.py"
WORK_DIR="work_dirs/vitpose_custom"

IMG_DIR="data/dataset/images"
ANN_DIR="data/dataset/annotations"

export PYTHONPATH=$PYTHONPATH:$(pwd)/mmpose

#####################################
# 0. Prepare dirs
#####################################
mkdir -p frames "$IMG_DIR" "$ANN_DIR"
mkdir -p skeleton_vis outputs_finetuned "$WORK_DIR"

#####################################
# 1. Video -> Frames
#####################################
echo "=== [1] Extract frames ==="
for v in videos/*.{mp4,MP4}; do
  [ -f "$v" ] || continue
  name=$(basename "$v" | sed 's/\.[mM][pP]4//')
  mkdir -p frames/$name
  ffmpeg -y -i "$v" -vf fps=$FPS frames/$name/frame_%06d.jpg
done

#####################################
# 2. Collect frames
#####################################
echo "=== [2] Collect frames ==="
rm -f "$IMG_DIR"/*.jpg
c=1
for d in frames/*; do
  for f in "$d"/*.jpg; do
    printf -v n "frame_%06d.jpg" "$c"
    cp "$f" "$IMG_DIR/$n"
    ((c++))
  done
done

#####################################
# 3. Detect center person bbox
#####################################
echo "=== [3] Detect center person bbox ==="
python scripts/detect_center_bbox.py \
  --img-dir "$IMG_DIR" \
  --out-json "$ANN_DIR/center_bboxes.json"

#####################################
# 4. Generate pseudo keypoints (COCO)
#####################################
echo "=== [4] Generate pseudo keypoints (COCO, OFFICIAL ViTPose) ==="
python scripts/pose_to_coco.py \
  --img-dir "$IMG_DIR" \
  --bbox-json "$ANN_DIR/center_bboxes.json" \
  --pose-config "$POSE_CFG_PSEUDO" \
  --pose-checkpoint "$POSE_PRETRAIN_PSEUDO" \
  --out-train "$ANN_DIR/train.json" \
  --out-val "$ANN_DIR/val.json" \
  --out-test "$ANN_DIR/test.json" \
  --train-ratio 0.8 \
  --val-ratio 0.1

#####################################
# 4.5 Visualize pseudo keypoints
#####################################
echo "=== [4.5] Visualize pseudo keypoints ==="
for split in train val test; do
    python scripts/visualize_coco_pose.py \
      --img-dir "$IMG_DIR" \
      --coco-json "$ANN_DIR/${split}.json" \
      --out-dir outputs_debug_vis/$split
done

#####################################
# 5. Train ViTPose (fine-tune)
#####################################
echo "=== [5] Train ViTPose ==="
python scripts/train.py \
  "$POSE_CFG_CUSTOM" \
  --work-dir "$WORK_DIR"

#####################################
# 6. Inference (finetuned)
#####################################
echo "=== [6] Inference finetuned ==="
for split in train val test; do
    echo "--- Inference for $split ---"
    mkdir -p "outputs_finetuned/$split"
    python scripts/inferencer_vitpose_finetuned.py \
        --img-dir "$IMG_DIR" \
        --coco-json "$ANN_DIR/${split}.json" \
        --out-dir "outputs_finetuned/$split" \
        --out-json results_coco.json
done

#####################################
# 7. Draw skeleton for all splits
#####################################
echo "=== [7] Draw skeleton ==="
for split in train val test; do
    echo "--- Draw skeleton for $split ---"
    OUT_SKEL="skeleton_vis/$split"
    mkdir -p "$OUT_SKEL"
    python scripts/draw_skeleton.py \
        "$IMG_DIR" \
        "outputs_finetuned/$split/results_coco.json" \
        "$OUT_SKEL"
done

#####################################
# 8. Skeleton -> Video
#####################################
echo "=== [8] Skeleton video ==="
for split in train val test; do
    echo "--- Skeleton video for $split ---"
    ffmpeg -y -r $FPS -i "skeleton_vis/$split/frame_%06d.jpg" \
        -c:v libx264 -pix_fmt yuv420p "skeleton_${split}.mp4"
done

echo "=== PIPELINE DONE ==="
