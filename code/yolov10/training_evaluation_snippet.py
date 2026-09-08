# YOLOv10 Training and Evaluation Snippet
# ----------------------------------------
# Representative implementation used in the ADAS
# object-detection comparative benchmark.
#
# The complete implementation, dataset, model weights,
# and proprietary dataset annotations are not included.

import torch
from pathlib import Path
from yolov10 import YOLOv10


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = Path("./dataset")
DATASET_YAML = BASE_DIR / "dataset.yaml"

EPOCHS = 400
BATCH_SIZE = 16
IMAGE_SIZE = 640
DEVICE = "0"

MODEL_SIZE = "n"
NUM_CLASSES = 27

OUTPUT_DIR = Path("./research_outputs")
MODEL_OUTPUT = OUTPUT_DIR / "trained_models" / "YOLOv10"


# ---------------------------------------------------------
# Device selection
# ---------------------------------------------------------

device = torch.device(
    f"cuda:{DEVICE}" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")


# ---------------------------------------------------------
# Model initialization
# ---------------------------------------------------------

model_name = f"yolov10{MODEL_SIZE}"
model = YOLOv10(model_name)

print(f"Loaded model: {model_name}")


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

results = model.train(
    data=str(DATASET_YAML),
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=int(DEVICE),
    patience=50,
    save=True,
    save_period=10,
    project=str(MODEL_OUTPUT),
    name="training",
    verbose=True,
    plots=True
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

val_results = model.val()

print("Validation completed.")


# ---------------------------------------------------------
# Prediction on validation images
# ---------------------------------------------------------

predictions = model.predict(
    source="path/to/validation/images",
    conf=0.25,
    device=int(DEVICE),
    save=True,
    project=str(MODEL_OUTPUT),
    name="predictions"
)
