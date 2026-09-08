# YOLOv8 Training and Evaluation Snippet
# ---------------------------------------
# Representative implementation used in the ADAS
# object-detection comparative benchmark.
#
# Complete implementation, dataset, annotations, and
# trained model weights are not included.

import os
import torch
import yaml
import pandas as pd

from ultralytics import YOLO


# ---------------------------------------------------------
# Device Configuration
# ---------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ---------------------------------------------------------
# Dataset Configuration
# ---------------------------------------------------------

BASE_DIR = "path/to/Adas_Project"

TRAIN_IMAGES = os.path.join(
    BASE_DIR, "dataset/images/train"
)

VAL_IMAGES = os.path.join(
    BASE_DIR, "dataset/images/val"
)

TRAIN_LABELS = os.path.join(
    BASE_DIR, "dataset/labels/train"
)

VAL_LABELS = os.path.join(
    BASE_DIR, "dataset/labels/val"
)

dataset_yaml = {
    "path": BASE_DIR + "/dataset",
    "train": "images/train",
    "val": "images/val",
    "nc": 27,
    "names": [f"class_{i}" for i in range(27)]
}


# ---------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------

EPOCHS = 400
IMAGE_SIZE = 640
BATCH_SIZE = 8

OUTPUT_DIR = "./research_outputs/YOLOv8"

os.makedirs(OUTPUT_DIR, exist_ok=True)

yaml_path = os.path.join(
    OUTPUT_DIR, "dataset.yaml"
)

with open(yaml_path, "w") as f:
    yaml.dump(dataset_yaml, f)


# ---------------------------------------------------------
# Model Initialization
# ---------------------------------------------------------

model = YOLO("yolov8n.pt")

print("YOLOv8 model loaded")


# ---------------------------------------------------------
# Model Training
# ---------------------------------------------------------

results = model.train(
    data=yaml_path,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=0,
    workers=4,
    optimizer="SGD",
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    patience=50,
    project=OUTPUT_DIR,
    name="adas_yolov8",
    save=True,
    save_period=10,
    plots=True,
    cache=True,
    verbose=True
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

metrics = model.val()

print(metrics)


# ---------------------------------------------------------
# Epoch-wise Training Log
# ---------------------------------------------------------

results_df = pd.read_csv(
    os.path.join(
        OUTPUT_DIR,
        "adas_yolov8",
        "results.csv"
    )
)

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "yolov8_training_log.csv"
    ),
    index=False
)

print("Training log saved")
