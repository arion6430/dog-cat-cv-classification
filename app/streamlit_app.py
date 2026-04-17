"""Streamlit UI for the cat/dog classifier."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# Allow running via `streamlit run app/streamlit_app.py` without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_NAMES, METRICS_PATH, MODEL_PATH
from src.predict import predict


st.set_page_config(page_title="Cat vs Dog Classifier", page_icon=":paw_prints:", layout="centered")


@st.cache_resource
def _load():
    from src.predict import get_model
    return get_model()


def _load_metrics() -> dict | None:
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text())
        except json.JSONDecodeError:
            return None
    return None


st.title("Cat vs Dog Classifier")
st.caption("Upload an image to classify it as a cat or a dog. Model: fine-tuned ResNet18.")

if not MODEL_PATH.exists():
    st.error(
        f"Model checkpoint not found at `{MODEL_PATH}`. "
        "Train the model first: `python -m src.train`."
    )
    st.stop()

model = _load()

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_column_width=True)

    with st.spinner("Running inference..."):
        result = predict(image, model=model)

    st.subheader(f"Prediction: **{result.label}**")
    st.metric("Confidence", f"{result.confidence * 100:.1f}%")
    st.bar_chart({name: [result.probs[name]] for name in CLASS_NAMES})

with st.expander("About the model"):
    metrics = _load_metrics()
    if metrics:
        st.markdown(
            f"- **Test accuracy:** {metrics['accuracy']:.4f}\n"
            f"- **Precision:** {metrics['precision']:.4f}\n"
            f"- **Recall:** {metrics['recall']:.4f}\n"
            f"- **F1:** {metrics['f1']:.4f}\n"
            f"- **Test set size:** {metrics['n_test']}"
        )
    else:
        st.info("Run `python -m src.evaluate` to populate test metrics here.")
    st.markdown(
        "Architecture: **ResNet18** pretrained on ImageNet, head replaced with a 2-class "
        "linear layer, last residual block fine-tuned. Input size 224×224."
    )
