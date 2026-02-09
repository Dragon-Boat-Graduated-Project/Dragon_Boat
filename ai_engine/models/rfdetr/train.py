import os
from rfdetr import RFDETRBase
import torch

# GPU / CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

DATASET_PATH = os.environ.get("DATASET_PATH", "DataSet")
OUTPUT_PATH = os.environ.get("MODEL_OUTPUT", "Output")

model = RFDETRBase(model_size="medium", num_classes=1, pretrained=True)

model.train(
    dataset_dir=DATASET_PATH,
    epochs=100,
    batch_size=8,
    grad_accum_steps=2,
    lr=2e-4,
    output_dir=OUTPUT_PATH,
    save_interval=1,
    num_workers=4,
    prefetch_factor=2,
    persistent_workers=False,
    multi_scale=False,
    expanded_scales=False,
    device=device,
)
