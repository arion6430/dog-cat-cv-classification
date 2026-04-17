# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (CPU-only PyTorch, no CUDA required)
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt

# Prepare data splits (run once after placing data in data/raw/train/)
python -c "from src.dataset import build_splits; build_splits()"

# Train the model
python -m src.train
python -m src.train --epochs 10 --finetune-after 2

# Evaluate on test set
python -m src.evaluate

# Run the web app locally
streamlit run app/streamlit_app.py

# Docker build and run
docker build -t cat-dog-classifier .
docker run --rm -p 8501:8501 cat-dog-classifier
```

## Architecture

Binary image classifier (cat vs dog). Full ML pipeline: data → training → evaluation → web deployment.

**Data flow:**
1. `src/dataset.py` — `build_splits()` reads `data/raw/train/*.jpg`, shuffles with fixed seed, writes index to `data/processed/splits.json`. `CatDogDataset` is lazy (opens images on `__getitem__`). Class label is inferred from filename prefix (`cat.*` → 0, `dog.*` → 1).
2. `src/config.py` — single source of truth for all paths, hyperparameters, and `DEVICE`. Import from here, don't hardcode elsewhere.

**Model (`src/model.py`):**
- `build_model(pretrained, freeze_backbone)` — ResNet18 with `fc` replaced by `Linear(512, 2)`.
- `unfreeze_backbone(model, last_block_only=True)` — call during training after the head converges.
- `load_model(path, device)` — loads `state_dict`, sets `eval()`. Used by `predict.py` and `evaluate.py`.

**Training (`src/train.py`):**
Two-phase training: freeze backbone → train head (lr=1e-3) for `finetune_after` epochs, then unfreeze `layer4` → fine-tune (lr=1e-4). Best checkpoint saved by val accuracy. Outputs `models/best_model.pt` and `reports/figures/training_curves.png`.

**Inference (`src/predict.py`):**
`predict(image) → Prediction(label, confidence, probs)` is the shared inference entry point used by both the Streamlit app and `evaluate.py`. Model is cached in `_model_cache` (dict keyed by path string); in Streamlit use `@st.cache_resource` wrapper `_load()` instead.

**Streamlit app (`app/streamlit_app.py`):**
Adds `PROJECT_ROOT` to `sys.path` so it works when launched with `streamlit run app/streamlit_app.py` from the repo root without installing the package. `@st.cache_resource` wraps model loading to avoid re-reading weights on each request.

## Data layout expected

```
data/raw/train/
    cat.0.jpg
    cat.1.jpg
    ...
    dog.0.jpg
    ...
```

Source: [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats) — download and extract `train.zip`.

## Key files

| File | Purpose |
|------|---------|
| `src/config.py` | All paths and hyperparameters |
| `src/dataset.py` | Dataset class, splits, transforms |
| `src/model.py` | Model definition and loading |
| `src/train.py` | Training loop entry point |
| `src/evaluate.py` | Test metrics + confusion matrix |
| `src/predict.py` | Inference function shared by app and eval |
| `app/streamlit_app.py` | Web UI |
| `Dockerfile` | CPU-only container, serves on port 8501 |
