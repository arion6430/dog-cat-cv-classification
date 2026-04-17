"""Training loop for the cat/dog classifier.

Usage:
    python -m src.train

Saves the best checkpoint (by val accuracy) to `models/best_model.pt`
and training curves to `reports/figures/training_curves.png`.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    DEVICE,
    FIGURES_DIR,
    LR_FINETUNE,
    LR_HEAD,
    MODELS_DIR,
    MODEL_PATH,
    NUM_EPOCHS,
    NUM_WORKERS,
    SEED,
    WEIGHT_DECAY,
)
from src.dataset import get_datasets
from src.model import build_model, unfreeze_backbone


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    model.train(mode=train)
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, desc="train" if train else "val", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def plot_curves(history: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(epochs: int = NUM_EPOCHS, finetune_after: int = 1) -> None:
    set_seed()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    datasets = get_datasets()
    loaders = {
        split: DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=NUM_WORKERS,
            pin_memory=(DEVICE.type == "cuda"),
        )
        for split, ds in datasets.items()
    }

    model = build_model(pretrained=True, freeze_backbone=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=LR_HEAD,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        if epoch == finetune_after + 1:
            unfreeze_backbone(model, last_block_only=True)
            optimizer = optim.Adam(
                (p for p in model.parameters() if p.requires_grad),
                lr=LR_FINETUNE,
                weight_decay=WEIGHT_DECAY,
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
            print(f"[epoch {epoch}] unfroze layer4 for fine-tuning")

        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, optimizer, DEVICE, train=True)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, optimizer, DEVICE, train=False)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"epoch {epoch}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  saved best checkpoint (val_acc={val_acc:.4f}) -> {MODEL_PATH}")

    plot_curves(history, FIGURES_DIR / "training_curves.png")
    (FIGURES_DIR.parent / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"done. best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--finetune-after", type=int, default=1,
                        help="Epoch after which to unfreeze layer4 for fine-tuning")
    args = parser.parse_args()
    main(epochs=args.epochs, finetune_after=args.finetune_after)
