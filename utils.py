"""
utils.py — Utility functions for the Brain Tumor Prediction app.

Handles model loading, image preprocessing, inference,
and class metadata for the CNN-based brain tumor classifier.
"""

import os
import numpy as np
import streamlit as st
from PIL import Image

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brain_tumor_model.keras")

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

CLASS_INFO = {
    "Glioma": {
        "icon": "🔴",
        "color": "#FF4B4B",
        "severity": "High",
        "description": (
            "Gliomas originate from glial cells in the brain or spine. "
            "They are among the most common and aggressive primary brain tumors, "
            "accounting for ~33% of all brain tumors. Treatment typically involves "
            "surgery, radiation, and chemotherapy."
        ),
    },
    "Meningioma": {
        "icon": "🟠",
        "color": "#FF8C00",
        "severity": "Moderate",
        "description": (
            "Meningiomas arise from the meninges — the membranes surrounding the "
            "brain and spinal cord. They are the most common primary brain tumors, "
            "representing ~30% of all cases. Most are benign and slow-growing, "
            "but some may require surgical intervention."
        ),
    },
    "No Tumor": {
        "icon": "🟢",
        "color": "#00C853",
        "severity": "None",
        "description": (
            "No tumor has been detected in the MRI scan. The brain tissue "
            "appears normal. Regular check-ups are still recommended to "
            "maintain neurological health."
        ),
    },
    "Pituitary": {
        "icon": "🟡",
        "color": "#FFD600",
        "severity": "Low-Moderate",
        "description": (
            "Pituitary tumors develop in the pituitary gland at the base of the "
            "brain. They are usually benign (adenomas) and can affect hormone "
            "production. Treatment options include medication, surgery, "
            "and radiation therapy."
        ),
    },
}

# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI model ...")
def load_model():
    """
    Load the trained Keras model (cached across reruns).

    Returns the model on success, or None on failure.
    """
    try:
        import tensorflow as tf

        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found: {MODEL_PATH}")
            return None

        model = tf.keras.models.load_model(MODEL_PATH)
        return model

    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


# ──────────────────────────────────────────────
# Image Preprocessing
# ──────────────────────────────────────────────

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Resize a PIL image for model inference.

    The model has a built-in Rescaling(1/255) layer, so raw pixel
    values in [0, 255] are passed without manual normalisation.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image in any mode/size.

    Returns
    -------
    np.ndarray
        Batch-ready array with shape (1, 150, 150, 3) and values in [0, 255].
    """
    image = image.convert("RGB")
    image = image.resize((150, 150))
    img_array = np.array(image, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────

def predict(model, image: Image.Image) -> dict:
    """
    Run inference on a single image.

    Returns
    -------
    dict
        {
            "label": str,           # Predicted class name
            "confidence": float,    # Confidence for predicted class (0-1)
            "probabilities": dict,  # {class_name: probability} for all classes
        }

    Raises
    ------
    RuntimeError
        If inference fails for any reason.
    """
    try:
        processed = preprocess_image(image)
        predictions = model.predict(processed, verbose=0)[0]

        probabilities = {
            name: float(prob) for name, prob in zip(CLASS_NAMES, predictions)
        }
        label = CLASS_NAMES[int(np.argmax(predictions))]
        confidence = float(np.max(predictions))

        return {
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}") from e
