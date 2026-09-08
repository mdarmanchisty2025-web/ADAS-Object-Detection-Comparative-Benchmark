# Representative ResNet-101 Faster R-CNN Training and Evaluation Snippet
# ADAS Object Detection Comparative Benchmark

import os
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.amp import GradScaler, autocast

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 400
BATCH_SIZE = 4
IMAGE_SIZE = 640
NUM_CLASSES = 2
LEARNING_RATE = 1e-4
ACCUMULATION_STEPS = 2

# Dataset paths
BASE_DIR = Path("./dataset")
TRAIN_IMAGES = os.path.join(BASE_DIR, "dataset/images/train")
TRAIN_LABELS = os.path.join(BASE_DIR, "dataset/labels/train")
VAL_IMAGES = os.path.join(BASE_DIR, "dataset/images/val")
VAL_LABELS = os.path.join(BASE_DIR, "dataset/labels/val")

# ResNet-101 + FPN backbone
backbone = resnet_fpn_backbone(
    "resnet101",
    weights="DEFAULT"
)

model = FasterRCNN(
    backbone,
    num_classes=NUM_CLASSES
).to(DEVICE)

# Optimizer and mixed precision
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

scaler = GradScaler(device="cuda")

# Training
for epoch in range(EPOCHS):

    model.train()
    optimizer.zero_grad()

    # images, targets = load_training_batch(...)

    # with autocast(device_type="cuda"):
    #     loss_dict = model(
    #         [img.to(DEVICE) for img in images],
    #         [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
    #     )
    #     loss = sum(loss_dict.values())

    # loss = loss / ACCUMULATION_STEPS
    # scaler.scale(loss).backward()

    # if (step + 1) % ACCUMULATION_STEPS == 0:
    #     scaler.step(optimizer)
    #     scaler.update()
    #     optimizer.zero_grad()

# Save trained model
torch.save(
    model.state_dict(),
    "resnet101_faster_rcnn.pth"
)
