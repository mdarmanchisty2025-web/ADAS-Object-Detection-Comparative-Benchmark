"""
Representative Deformable DETR Training Snippet
------------------------------------------------
This file documents the principal training configuration and procedure
used in the comparative ADAS object-detection study.

Note:
- Representative research code only.
- Dataset files, model weights, and proprietary annotations are not included.
- Replace local paths with the appropriate dataset location.
"""

import os
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import DeformableDetrForObjectDetection


# ============================================================
# 1. DEVICE CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. EXPERIMENT CONFIGURATION
# ============================================================

BATCH_SIZE = 2
EPOCHS = 400
LEARNING_RATE = 1e-4
IMAGE_SIZE = 640

# The original implementation used 28 labels.
# Verify the class-index convention before reproducing results.
NUM_CLASSES = 28


# ============================================================
# 3. DATASET PATHS
# ============================================================

BASE_DIR = "path/to/Adas_Project"

TRAIN_IMAGES = os.path.join(
    BASE_DIR, "dataset", "images", "train"
)

VAL_IMAGES = os.path.join(
    BASE_DIR, "dataset", "images", "val"
)

TRAIN_LABELS = os.path.join(
    BASE_DIR, "dataset", "labels", "train"
)

VAL_LABELS = os.path.join(
    BASE_DIR, "dataset", "labels", "val"
)


# ============================================================
# 4. IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# ============================================================
# 5. DATASET
# ============================================================

class ADASDataset(Dataset):

    def __init__(self, image_dir, label_dir):

        self.image_dir = image_dir
        self.label_dir = label_dir

        self.images = sorted(
            os.listdir(image_dir)
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_name = self.images[idx]

        img_path = os.path.join(
            self.image_dir,
            img_name
        )

        label_path = os.path.join(
            self.label_dir,
            img_name.replace(".jpg", ".txt")
        )

        image = Image.open(
            img_path
        ).convert("RGB")

        pixel_values = transform(image)

        boxes = []
        class_labels = []

        if os.path.exists(label_path):

            with open(label_path) as f:

                for line in f.readlines():

                    c, x, y, w, h = map(
                        float,
                        line.split()
                    )

                    boxes.append([
                        x, y, w, h
                    ])

                    class_labels.append(
                        int(c)
                    )

        if len(boxes) == 0:

            boxes = torch.zeros(
                (0, 4)
            )

            class_labels = torch.zeros(
                (0,),
                dtype=torch.long
            )

        else:

            boxes = torch.tensor(
                boxes
            )

            class_labels = torch.tensor(
                class_labels
            )

        target = {
            "class_labels": class_labels,
            "boxes": boxes
        }

        return pixel_values, target


# ============================================================
# 6. COLLATE FUNCTION
# ============================================================

def collate_fn(batch):

    pixel_values = []
    labels = []

    for img, target in batch:

        pixel_values.append(img)
        labels.append(target)

    pixel_values = torch.stack(
        pixel_values
    )

    return pixel_values, labels


# ============================================================
# 7. DATA LOADERS
# ============================================================

train_dataset = ADASDataset(
    TRAIN_IMAGES,
    TRAIN_LABELS
)

val_dataset = ADASDataset(
    VAL_IMAGES,
    VAL_LABELS
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=True
)

print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset)
)


# ============================================================
# 8. MODEL INITIALIZATION
# ============================================================

model = DeformableDetrForObjectDetection.from_pretrained(
    "SenseTime/deformable-detr",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)

model.to(DEVICE)


# ============================================================
# 9. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 10. MIXED-PRECISION TRAINING
# ============================================================

scaler = torch.cuda.amp.GradScaler()


# ============================================================
# 11. TRAINING LOOP
# ============================================================

training_log = []

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0.0

    for pixel_values, labels in tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    ):

        pixel_values = pixel_values.to(
            DEVICE
        )

        labels = [
            {
                k: v.to(DEVICE)
                for k, v in target.items()
            }
            for target in labels
        ]

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():

            outputs = model(
                pixel_values=pixel_values,
                labels=labels
            )

            loss = outputs.loss

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        epoch_loss += loss.item()

    avg_loss = (
        epoch_loss /
        len(train_loader)
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {avg_loss:.4f}"
    )

    training_log.append([
        epoch + 1,
        avg_loss
    ])

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# 12. MODEL CHECKPOINT
# ============================================================

torch.save(
    model.state_dict(),
    "deformable_detr_final.pth"
)

print(
    "Deformable DETR training completed."
)
