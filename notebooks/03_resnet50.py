import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report

# -------------------------
# Config
# -------------------------
class Config:
    dataset_root = Path("../data/raw/labeled_images")
    metadata_csv = Path("../data/raw/metadata.csv")
    output_dir = Path("../results")

    image_size = 224
    batch_size = 32
    epochs = 20
    lr = 0.01
    weight_decay = 1e-4

    pretrained = True
    dropout = 0.5
    freeze_backbone = True
    unfreeze_last_blocks = 1

    patience = 5

    seed = 42

# -------------------------
# Utils
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------
# Dataset & Dataloaders
# -------------------------
class CapsuleDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])
        if not img_path.is_absolute():
            img_path = self.root / img_path
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row["encoded_label"])
        return image, label


def get_transforms(image_size=224):
    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_tfms, val_tfms


def create_dataloader(df, root, batch_size=32, image_size=224, is_train=True):
    train_tfms, val_tfms = get_transforms(image_size)
    transform = train_tfms if is_train else val_tfms
    dataset = CapsuleDataset(df, root, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=2)
    return loader


def stratified_group_split(df, n_splits=5, fold_index=0, seed=42):
    y = df["encoded_label"].values
    groups = df["video_id"].values
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(sgkf.split(np.zeros(len(df)), y, groups))
    train_idx, val_idx = splits[fold_index]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)

# -------------------------
# Model
# -------------------------
def create_resnet50(num_classes, pretrained=True, dropout=0.5, freeze_backbone=True, unfreeze_last_blocks=1):
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

        stages = [model.layer4, model.layer3, model.layer2, model.layer1]
        for i in range(min(unfreeze_last_blocks, len(stages))):
            for p in stages[i].parameters():
                p.requires_grad = True

    return model

# -------------------------
# Callbacks (EarlyStopping + Checkpoint)
# -------------------------
class EarlyStopping:
    def __init__(self, patience=5, mode="min"):
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            return
        improve = (score < self.best_score) if self.mode == "min" else (score > self.best_score)
        if improve:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

class ModelCheckpoint:
    def __init__(self, filepath, monitor="val_bacc", mode="max", save_best_only=True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None

    def step(self, score, model, extra=None):
        if self.best_score is None:
            self.best_score = score

        improve = (score < self.best_score) if self.mode == "min" else (score > self.best_score)
        if improve or not self.save_best_only:
            self.best_score = score
            ckpt = {"model_state": model.state_dict()}
            if extra:
                ckpt.update(extra)
            torch.save(ckpt, self.filepath)
            print(f"[INFO] Model saved to {self.filepath} (score={score:.4f})")

# -------------------------
# Training & Validation
# -------------------------
def run_one_epoch(model, loader, criterion, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()
    losses, all_preds, all_tgts = [], [], []
    for imgs, tgts in loader:
        imgs, tgts = imgs.to(device), tgts.to(device)
        if train:
            optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, tgts)
        if train:
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        preds = logits.argmax(1).detach().cpu().numpy().tolist()
        all_preds.extend(preds)
        all_tgts.extend(tgts.cpu().numpy().tolist())
    acc = accuracy_score(all_tgts, all_preds)
    bacc = balanced_accuracy_score(all_tgts, all_preds)
    return {"loss": np.mean(losses), "acc": acc, "bacc": bacc}

# -------------------------
# Visualization
# -------------------------
def plot_history(history):
    train_acc = [h["train"]["acc"] for h in history]
    val_acc = [h["val"]["acc"] for h in history]
    train_loss = [h["train"]["loss"] for h in history]
    val_loss = [h["val"]["loss"] for h in history]
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_acc, label="train")
    plt.plot(val_acc, label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.title("Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.show()

def plot_confusion_matrix(cm, classes, title="Confusion Matrix", cmap="Blues"):
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1)
    plt.figure(figsize=(6,5))
    plt.imshow(cm_norm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

# -------------------------
# Main
# -------------------------
def main(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    df = pd.read_csv(cfg.metadata_csv)
    print(df.head(5))
    
    lb = LabelEncoder()
    df["encoded_label"] = lb.fit_transform(df["finding_class"])

    # Split train/val
    train_df, val_df = stratified_group_split(df, n_splits=5, fold_index=0, seed=cfg.seed)

    # DataLoaders
    train_loader = create_dataloader(train_df, cfg.dataset_root, cfg.batch_size, cfg.image_size, is_train=True)
    val_loader = create_dataloader(val_df, cfg.dataset_root, cfg.batch_size, cfg.image_size, is_train=False)

    # Model, loss, optimizer
    model = create_resnet50(num_classes=len(lb.classes_),
                            pretrained=cfg.pretrained,
                            dropout=cfg.dropout,
                            freeze_backbone=cfg.freeze_backbone,
                            unfreeze_last_blocks=cfg.unfreeze_last_blocks).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=cfg.lr, momentum=0.9, nesterov=True, weight_decay=cfg.weight_decay)

    # Callbacks
    early_stopper = EarlyStopping(patience=cfg.patience, mode="min")
    checkpointer = ModelCheckpoint(str(cfg.output_dir / "best_model.pt"), monitor="val_bacc", mode="max")

    # Training loop
    history = []
    for epoch in range(1, cfg.epochs+1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        train_metrics = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = run_one_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f" train: {train_metrics}")
        print(f"   val: {val_metrics}")
        history.append({"train": train_metrics, "val": val_metrics})

        early_stopper.step(val_metrics["loss"])
        checkpointer.step(val_metrics["bacc"], model, extra={"classes": lb.classes_})
        if early_stopper.should_stop:
            print("[INFO] Early stopping triggered.")
            break

    # Load best model
    ckpt = torch.load(cfg.output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    classes = ckpt["classes"]

    # Evaluate on val set
    all_preds, all_tgts = [], []
    model.eval()
    with torch.no_grad():
        for imgs, tgts in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_tgts.extend(tgts.numpy().tolist())

    class_pred_list = lb.inverse_transform(all_preds)
    y_true = lb.inverse_transform(all_tgts)

    acc = accuracy_score(y_true, class_pred_list)
    cm = confusion_matrix(y_true, class_pred_list, labels=lb.classes_)
    report = classification_report(y_true, class_pred_list, target_names=lb.classes_, digits=4)

    print("\nValidation Accuracy:", acc)
    print("\nClassification Report:\n", report)
    plot_confusion_matrix(cm, classes=lb.classes_)
    plot_history(history)

    # Qualitative: show 9 samples
    val_df["predicted_labels"] = class_pred_list
    sample = val_df.sample(n=9, random_state=6)
    plt.figure(figsize=(12,12))
    for i, row in sample.reset_index(drop=True).iterrows():
        img = Image.open(row["image_path"]).convert("RGB")
        plt.subplot(3,3,i+1)
        plt.imshow(img)
        color = "green" if row["finding_class"] == row["predicted_labels"] else "red"
        plt.xlabel(f"pred: {row['predicted_labels']}\ntrue: {row['finding_class']}", color=color)
        plt.xticks([]); plt.yticks([])
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    cfg = Config()
    main(cfg)