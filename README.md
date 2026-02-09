# Dragon Boat Analysis

Computer vision pipeline for detecting paddles in dragon boat racing videos using a two-stage model distillation approach (RF-DETR teacher -> YOLOv11 student).

## Project Structure

```
dragon-boat-analysis/
├── docs/                  # Documentation
├── web/                   # Frontend application
├── backend/               # FastAPI backend
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   └── utils/
│   ├── requirements.txt
│   └── main.py
└── ai_engine/             # ML training & inference
    ├── models/
    │   ├── rfdetr/        # RF-DETR teacher model training
    │   └── yolo/          # YOLOv11 student model training
    ├── pipelines/
    │   └── paddle_detect/ # End-to-end paddle detection pipeline
    └── shared_data/       # Shared datasets (gitignored)
```

## Pipeline Overview

1. **Frame extraction** (`clip.py`) - Extract frames from raw videos at target FPS
2. **RF-DETR prediction** (`predict_rfdetr.py`) - Generate pseudo-labels with the teacher model
3. **Dataset building** (`build_dataset.py` / `move_labels.sh` + `split_dataset.sh`) - Assemble YOLO-format dataset
4. **YOLO training** (`yolo/train.sh`) - Train lightweight student model on pseudo-labels
5. **YOLO inference** (`predict_yolo.sh`) - Run fast inference on new videos

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
pip install rfdetr supervision torch torchvision opencv-python
```

## Environment Variables

See `.env.example` for all required paths:
- `PROJECT_ROOT` - Base directory on the training server
- `SOURCE_VIDEOS` - Directory containing raw `.mp4` files
- `MODEL_CHECKPOINT` - Path to RF-DETR trained weights
- `PREDICT_OUTPUT` - Where prediction outputs are saved
- `YOLO_DATASET` - YOLO-format dataset directory
