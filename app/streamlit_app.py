from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
import torch
import numpy as np
from PIL import Image

from models.tiny_cnn import TinyCNN


st.set_page_config(page_title="Deepfake Defense Demo", layout="wide")


@st.cache_resource
def load_model(checkpoint_path: str) -> TinyCNN:
    model = TinyCNN()
    path = Path(checkpoint_path)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def predict(image: Image.Image, model: TinyCNN) -> tuple[str, float]:
    resized = image.convert("RGB").resize((128, 128))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        probability_fake = model(tensor).softmax(dim=1)[0, 1].item()
    label = "fake" if probability_fake >= 0.5 else "real"
    return label, probability_fake


st.title("Deepfake Detection Defense Demo")
checkpoint = st.sidebar.text_input("Checkpoint", "models/tiny_cnn.pt")
model = load_model(checkpoint)

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is None:
    st.info("Upload an image to run the baseline detector.")
else:
    image = Image.open(uploaded)
    label, score = predict(image, model)
    left, right = st.columns([1, 1])
    with left:
        st.image(image, caption="Input", use_container_width=True)
    with right:
        st.metric("Prediction", label)
        st.metric("Fake probability", f"{score:.3f}")
        st.caption("Baseline demo only. Replace with a validated checkpoint before research reporting.")
