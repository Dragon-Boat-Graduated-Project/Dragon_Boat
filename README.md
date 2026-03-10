# Dragon Boat Analysis

Computer vision pipeline for dragon boat racing analysis, combining **paddle detection** (RF-DETR / YOLOv11) and **human pose estimation** (ViTPose / MMPose) to extract quantitative motion metrics from video.

## Project Structure

```
dragon-boat-analysis/
├── docs/                          # Documentation & reference materials
│   └── pose_estimation_references/
├── web/                           # Frontend application
├── backend/                       # FastAPI backend
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   └── utils/
│   ├── requirements.txt
│   └── main.py
└── ai_engine/                     # ML training & inference
    ├── models/
    │   ├── rfdetr/                # RF-DETR teacher model training
    │   └── yolo/                  # YOLOv11 student model training
    ├── pipelines/
    │   ├── paddle_detect/         # Paddle detection pipeline
    │   └── pose_estimation/       # Human pose estimation pipeline
    │       ├── configs/           # ViTPose & detector configs
    │       ├── scripts/           # Detection, training, inference, visualization
    │       └── run_pipeline.sh    # End-to-end pose estimation pipeline
    ├── shared_data/               # Shared datasets (gitignored)
    └── requirements.txt
```

## Pipelines

### 1. Paddle Detection

Detects paddles in video using a two-stage model distillation approach (RF-DETR teacher -> YOLOv11 student).

1. **Frame extraction** (`clip.py`) - Extract frames from raw videos at target FPS
2. **RF-DETR prediction** (`predict_rfdetr.py`) - Generate pseudo-labels with the teacher model
3. **Dataset building** (`build_dataset.py` / `move_labels.sh` + `split_dataset.sh`) - Assemble YOLO-format dataset
4. **YOLO training** (`yolo/train.sh`) - Train lightweight student model on pseudo-labels
5. **YOLO inference** (`predict_yolo.sh`) - Run fast inference on new videos

### 2. Pose Estimation

Extracts 2D human body keypoints (17 COCO joints) from athlete videos using ViTPose within the MMPose framework.

1. **Frame extraction** - Videos -> frames via FFmpeg
2. **Human detection** (`detect_center_bbox.py`) - Locate center athlete using Faster R-CNN
3. **Pseudo-label generation** (`pose_to_coco.py`) - Generate keypoint labels with pretrained ViTPose
4. **Fine-tuning** (`train.py`) - Fine-tune ViTPose on custom dragon boat data
5. **Inference** (`inferencer_vitpose_finetuned.py`) - Run fine-tuned model on all splits
6. **Visualization** (`draw_skeleton.py`) - Draw skeleton overlays and reassemble into video

```bash
cd ai_engine/pipelines/pose_estimation
bash run_pipeline.sh
```

## Setup

```bash
# 1. Clone the repo
git clone <repo-url> && cd dragon-boat-analysis

# 2. Configure environment paths
cp .env.example .env
# Edit .env to match your server paths

# 3. Source the env (or use dotenv)
set -a && source .env && set +a

# 4. Install Python dependencies
pip install -r ai_engine/requirements.txt
```

### Pose Estimation Additional Setup

MMPose requires specific versions. See `ai_engine/pipelines/pose_estimation/` for details:

```bash
# Install PyTorch with CUDA 11.8
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118

# Install OpenMMLab suite
pip install -U openmim
mim install mmcv==2.0.0

# Clone mmpose source (needed for configs/weights)
cd ai_engine/pipelines/pose_estimation
git clone https://github.com/open-mmlab/mmpose.git
```

## Environment Variables

See `.env.example` for all required paths:

**Paddle Detection:**
- `PROJECT_ROOT` - Base directory on the training server
- `SOURCE_VIDEOS` - Directory containing raw `.mp4` files
- `MODEL_CHECKPOINT` - Path to RF-DETR trained weights
- `PREDICT_OUTPUT` - Where prediction outputs are saved
- `YOLO_DATASET` - YOLO-format dataset directory

**Pose Estimation:**
- `POSE_VIDEOS` - Directory containing athlete videos
- `DET_WEIGHTS` - Faster R-CNN detection weights
- `VITPOSE_PRETRAIN` - ViTPose pretrained checkpoint
- `VITPOSE_WORK_DIR` - Fine-tuned model output directory
