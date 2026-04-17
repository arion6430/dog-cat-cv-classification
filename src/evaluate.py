"""Evaluate the trained model on the held-out test split.

Usage:
    python -m src.evaluate

Writes `reports/metrics.json` and `reports/figures/confusion_matrix.png`.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DEVICE,
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_PATH,
    NUM_WORKERS,
)
from src.dataset import get_datasets
from src.model import load_model


def collect_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def plot_confusion(cm: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    ax.set_title("Confusion Matrix (test)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No checkpoint at {MODEL_PATH}. Run `python -m src.train` first."
        )
    model = load_model(MODEL_PATH, DEVICE)
    test_ds = get_datasets()["test"]
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    preds, labels = collect_predictions(model, loader, DEVICE)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average="binary", pos_label=1)),
        "recall": float(recall_score(labels, preds, average="binary", pos_label=1)),
        "f1": float(f1_score(labels, preds, average="binary", pos_label=1)),
        "n_test": int(len(labels)),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    cm = confusion_matrix(labels, preds)
    plot_confusion(cm, FIGURES_DIR / "confusion_matrix.png")

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
