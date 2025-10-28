# streamlit_app/src/infer_preproc.py
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st
import os
from huggingface_hub import snapshot_download
import sys


MODEL_PATH = Path("streamlit_app/models/model.keras")
"""@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
    # compile=False para máxima compatibilidad TF 2.10+
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model"""

HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "mgonzalz/skinbridge-cnn")
HF_MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "model.keras")

@st.cache_resource(show_spinner=False)
def load_model():
    # Descarga (o usa cache) del repo de modelos de HF
    local_dir = snapshot_download(
        repo_id=HF_MODEL_REPO,
        revision=os.getenv("HF_REVISION", "main"),
        local_dir=os.getenv("HF_LOCAL_DIR", None),  # déjalo None para cache por defecto
        token=os.getenv("HF_TOKEN", None)           # solo si el repo es privado
    )
    model_path = Path(local_dir) / HF_MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró {HF_MODEL_FILENAME} en {local_dir}")
    model = tf.keras.models.load_model(model_path, compile=False)
    return model



def _prepare_rgb_224(pil_img: Image.Image) -> np.ndarray:
    """PIL RGB -> float32[1,224,224,3] en [0,1]"""
    x = pil_img.convert("RGB").resize((224, 224))
    x = np.asarray(x, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    return x

def predict_proba_malignant(pil_rgb_224: Image.Image) -> float:
    """
    Devuelve probabilidad de *Malignant* en [0,1].
    Soporta salidas:
        - Sigmoid: shape (1,1)  -> p = y[0,0]
        - Softmax: shape (1,2)  -> p = y[0,1] (orden Benign, Malignant)
    """
    model = load_model()
    x = _prepare_rgb_224(pil_rgb_224)
    y = model.predict(x, verbose=0)

    if y.ndim == 2 and y.shape[1] == 1:        # binaria sigmoid
        p_mal = float(y[0, 0])
    elif y.ndim == 2 and y.shape[1] == 2:      # softmax benign/malignant
        p_mal = float(y[0, 1])
    else:
        # Caso raro: convertir a prob via softmax y tomar último canal por convención
        sm = np.exp(y - y.max(axis=1, keepdims=True))
        sm = sm / sm.sum(axis=1, keepdims=True)
        p_mal = float(sm[0, -1])
    return max(0.0, min(1.0, p_mal))
