from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.dataset import CLASS_TO_IDX, discover_images, unique_records
from src.fusion import weighted_fusion
from src.preprocessing import image_to_tensor, load_image, preprocess_image
from src.training import build_model, load_training_config


st.set_page_config(page_title="Medical Image Fusion Diagnosis", layout="wide")
st.title("Medical Image Fusion Diagnosis")

with st.sidebar:
    st.header("Project controls")
    dataset_root = st.text_input("Dataset root", value="data/raw")
    config_path = st.text_input("Config file", value="config.yaml")
    checkpoint_path = st.text_input("Checkpoint path", value="models/checkpoints/litebrainnet.pt")
    preview_image = st.file_uploader("Upload image for quick preview", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])


@st.cache_data
def inspect_dataset(root: str):
    try:
        rows, issues = discover_images(root)
        clean = unique_records(rows)
        return rows, clean, issues
    except Exception as exc:
        return [], [], {"error": str(exc)}


if preview_image is not None:
    image = Image.open(preview_image)
    resized, arr = preprocess_image(image, size=(224, 224), grayscale=True)
    st.image(resized, caption="Preprocessed preview", use_container_width=True)


with st.container():
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Dataset summary")
        rows, clean, issues = inspect_dataset(dataset_root)
        if issues.get("error"):
            st.warning(f"Dataset inspection failed: {issues['error']}")
        else:
            st.write(f"Discovered images: {len(clean)}")
            counts = pd.Series([row["label"] for row in clean]).value_counts().sort_index()
            st.dataframe(counts.rename_axis("class").reset_index(name="count"))
            if issues["unlabeled"]:
                st.warning(f"{len(issues['unlabeled'])} files missing valid labels.")
            if issues["corrupt"]:
                st.warning(f"{len(issues['corrupt'])} files could not be opened.")

    with col_b:
        st.subheader("Illustrative fusion demo")
        left = st.file_uploader("Reference image A", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], key="fusion_a")
        right = st.file_uploader("Reference image B", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], key="fusion_b")
        alpha = st.slider("Fusion alpha", 0.0, 1.0, 0.5, 0.05)
        if left and right:
            img_a = Image.open(left).convert("L")
            img_b = Image.open(right).convert("L")
            fused, _ = weighted_fusion(img_a, img_b, alpha=alpha, register=False)
            st.image(fused, caption="Weighted fusion result", use_container_width=True)
        else:
            st.info("Upload two images to preview the fusion stage.")


st.subheader("Model configuration")
config = load_training_config(config_path)
st.json(config)

st.subheader("Quick inference")
if Path(checkpoint_path).exists():
    model = build_model(config)
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.eval()
    demo_file = st.file_uploader("Upload image for inference", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], key="inference_demo")
    if demo_file is not None:
        image = Image.open(demo_file).convert("L")
        tensor = image_to_tensor(image, size=(224, 224), grayscale=True).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        labels = list(CLASS_TO_IDX.keys())
        ranked = sorted(zip(labels, probs.tolist()), key=lambda item: item[1], reverse=True)
        st.dataframe(pd.DataFrame(ranked, columns=["class", "probability"]))
else:
    st.info("Add a trained checkpoint to enable inference. The model can still be inspected before training.")
