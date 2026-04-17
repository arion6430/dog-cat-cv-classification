"""Inference helper used by the Streamlit app and evaluation scripts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from src.config import CLASS_NAMES, DEVICE, MODEL_PATH
from src.dataset import eval_transform
from src.model import load_model


@dataclass
class Prediction:
    label: str
    confidence: float
    probs: dict[str, float]


_model_cache: dict[str, torch.nn.Module] = {}


def get_model(path: Path = MODEL_PATH, device: torch.device = DEVICE) -> torch.nn.Module:
    key = str(path)
    if key not in _model_cache:
        _model_cache[key] = load_model(path, device)
    return _model_cache[key]


def predict(image: Image.Image, model: torch.nn.Module | None = None,
            device: torch.device = DEVICE) -> Prediction:
    if model is None:
        model = get_model(device=device)
    transform = eval_transform()
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return Prediction(
        label=CLASS_NAMES[idx],
        confidence=float(probs[idx]),
        probs={name: float(p) for name, p in zip(CLASS_NAMES, probs)},
    )
