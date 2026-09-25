"""Unit tests for FS Model training script and model artifact validation."""

import os
import joblib
import pytest
import numpy as np

from scripts.train_fs_model import build_fs_training_dataset, train_and_evaluate_fs_model
from app.vision.face_engine import FaceBiometricEngine


def test_fs_model_file_exists():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "fs_model.pkl"))
    assert os.path.isfile(model_path), f"FS model file not found at {model_path}"

    data = joblib.load(model_path)
    assert "model" in data
    assert "accuracy" in data
    assert data["accuracy"] >= 0.85, f"Expected accuracy >= 0.85, got {data['accuracy']}"
    assert data["train_samples"] > 0
    assert data["test_samples"] > 0


def test_fs_model_inference():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "fs_model.pkl"))
    data = joblib.load(model_path)
    model = data["model"]

    # Generate synthetic 128-d embedding
    sample_emb = np.random.randn(1, 128).astype(np.float32)
    sample_emb = sample_emb / np.linalg.norm(sample_emb)

    pred = model.predict(sample_emb)
    proba = model.predict_proba(sample_emb)

    assert len(pred) == 1
    assert pred[0] in data["classes"]
    assert proba.shape[1] == len(data["classes"])
