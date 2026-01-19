import streamlit as st
from pathlib import Path
import pandas as pd
from utils import loader, plot


"""
To run on the terminal: streamlit run stream-app.py
"""
st.set_page_config(page_title="DimABSA Dashboard", layout="wide")
st.title("DimABSA 2026 - Training & Evaluation Dashboard")

# sidebar 
dataset = st.sidebar.selectbox("Select Dataset", ["laptop", "restaurant"])
subtask = st.sidebar.selectbox("Select Subtask", ["subtask1", "subtask2", "subtask3"])

# List available models dynamically
models = loader.list_models(f"data/{dataset}", subtask)
if not models:
    st.warning("No models/data found for this selection.")
    st.stop()

model_name = st.selectbox("Choose model", models)

# Construct file path with correct extension
file_path_candidates = [f"data/{dataset}/{subtask}/{model_name}{ext}" for ext in [".csv", ".json"]]
file_path = None
for fp in file_path_candidates:
    if Path(fp).exists():
        file_path = fp
        break
if file_path is None:
    st.error(f"File not found for {model_name}")
    st.stop()

# merics
metrics = loader.load_metrics(file_path)

# main visual
st.subheader(f"{dataset.capitalize()} - {subtask} - {model_name}")

if subtask == "subtask1" and isinstance(metrics, pd.DataFrame):
    st.write("### Metrics per Epoch")
    st.dataframe(metrics)
    fig = plot.plot_metrics(metrics)
    st.pyplot(fig)

elif subtask in ["subtask2", "subtask3"] and isinstance(metrics, dict):
    train_loss = metrics.get("train_loss_epoch", [])
    val_loss = metrics.get("test_loss_epoch", [])
    step_losses = metrics.get("all_step_losses", [])
    f1_scores = metrics.get("test_f1", [])
    best_epoch = metrics.get("best_epoch", 0)

    # Training & Validation Loss
    st.write("### Training & Validation Loss")
    fig1 = plot.plot_loss(train_loss, val_loss)
    st.pyplot(fig1)

    # F1 vs Epoch
    if f1_scores:
        st.write("### F1 Score vs Epoch")
        fig_f1 = plot.plot_f1_with_best_epoch(f1_scores, best_epoch)
        st.pyplot(fig_f1)

    # Signal spectrum of step-level losses
    if step_losses:
        st.write("### Signal Spectrum of Training Loss (Step-level)")
        fig2 = plot.plot_signal_spectrum(step_losses)
        st.pyplot(fig2)

    st.write(f"**Best Epoch:** {best_epoch}")

else:
    st.warning("Unsupported file format or subtask")