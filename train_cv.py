"""
AI Car Price Advisor - Computer Vision Training
================================================
Trains an EfficientNet-B0 model to classify vehicle body type from images.
Classes: Coupe, Hatchback, Sedan, SUV
Dataset: CarBodyStyles/ (organized by class subfolder)
Saved model: vehicle_classifier.pt
"""

import json
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from tqdm import tqdm

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "CarBodyStyles"
MODEL_SAVE = BASE_DIR / "vehicle_classifier.pt"
CLASS_NAMES_FILE = BASE_DIR / "cv_class_names.json"
SPLIT_DIR = BASE_DIR / "cv_data"

EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4
VAL_SPLIT = 0.2
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["Coupe", "Hatchback", "Sedan", "SUV"]

print("=" * 60)
print("AI Car Price Advisor — CV Training (EfficientNet-B0)")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Data:   {DATA_DIR}")

# ─── PREPARE TRAIN/VAL SPLIT ───────────────────────────────────────────────────
random.seed(SEED)

def is_valid_image(path: Path) -> bool:
    """Return True if the file can be opened as a PIL image."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def prepare_split(src_dir: Path, out_dir: Path, val_split: float = 0.2):
    """Copy valid images into out_dir/train/<class>/ and out_dir/val/<class>/."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for cls in CLASS_NAMES:
        (out_dir / "train" / cls).mkdir(parents=True, exist_ok=True)
        (out_dir / "val" / cls).mkdir(parents=True, exist_ok=True)
        candidates = list((src_dir / cls).glob("*.jpg"))
        candidates += list((src_dir / cls).glob("*.jpeg"))
        candidates += list((src_dir / cls).glob("*.png"))
        images = [p for p in candidates if is_valid_image(p)]
        skipped = len(candidates) - len(images)
        if skipped:
            print(f"  {cls}: skipped {skipped} corrupted file(s)")
        random.shuffle(images)
        split = int(len(images) * (1 - val_split))
        for img in images[:split]:
            shutil.copy(img, out_dir / "train" / cls / img.name)
        for img in images[split:]:
            shutil.copy(img, out_dir / "val" / cls / img.name)
        print(f"  {cls}: {split} train / {len(images) - split} val")

print("\nPreparing train/val split...")
prepare_split(DATA_DIR, SPLIT_DIR, VAL_SPLIT)

# ─── DATA TRANSFORMS ───────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(str(SPLIT_DIR / "train"), transform=train_tf)
val_ds = datasets.ImageFolder(str(SPLIT_DIR / "val"), transform=val_tf)

# Preserve class order: Coupe, Hatchback, Sedan, SUV
class_names = train_ds.classes
print(f"\nClasses ({len(class_names)}): {class_names}")
# ImageFolder sorts alphabetically; 'SUV' < 'Sedan' because uppercase 'U' < lowercase 'e'

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Train images: {len(train_ds)} | Val images: {len(val_ds)}")

# ─── MODEL ─────────────────────────────────────────────────────────────────────
model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

# Freeze backbone, train classifier head only
for param in model.parameters():
    param.requires_grad = False

in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features, len(class_names)),
)
model = model.to(DEVICE)

# ─── TRAINING ──────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

history = {"train_loss": [], "train_acc": [], "val_acc": []}
best_val_acc = 0.0

print("\nTraining...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    train_loss = running_loss / total

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]  "):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            val_correct += preds.eq(labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total

    scheduler.step()
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"  Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "num_classes": len(class_names),
                "val_acc": val_acc,
                "epoch": epoch,
            },
            MODEL_SAVE,
        )
        print(f"  [Saved best model — val_acc={val_acc:.4f}]")

print(f"\nTraining complete. Best Val Accuracy: {best_val_acc:.4f}")

# ─── FULL EVALUATION ON VALIDATION SET ────────────────────────────────────────
checkpoint = torch.load(MODEL_SAVE, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# ─── CONFUSION MATRIX ──────────────────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(6, 5))
sns_hm = ax.imshow(cm, cmap="Blues")
plt.colorbar(sns_hm, ax=ax)
ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=30)
ax.set_yticklabels(class_names)
for i in range(len(class_names)):
    for j in range(len(class_names)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix — EfficientNet-B0")
plt.tight_layout()
plt.savefig(BASE_DIR / "cv_confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved cv_confusion_matrix.png")

# ─── TRAINING CURVE ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history["train_loss"], marker="o", label="Train Loss")
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history["train_acc"], marker="o", label="Train Acc")
axes[1].plot(history["val_acc"], marker="s", label="Val Acc")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig(BASE_DIR / "cv_training_curve.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved cv_training_curve.png")

# ─── SAVE CLASS NAMES ──────────────────────────────────────────────────────────
with CLASS_NAMES_FILE.open("w") as f:
    json.dump(class_names, f, indent=2)
print(f"Class names saved to {CLASS_NAMES_FILE}")
print("\nDone!")
