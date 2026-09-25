#!/usr/bin/env python3
"""FS (Face Search / Feature Security) Model Training Script.

Trains the FS model using the CCTV surveillance dataset located in the 'Training' folder.
- Split Ratio: 80% Train, 20% Test
- Target Accuracy: >= 85% to 90%+
- Feature Extraction: YuNet 5-point face alignment + SFace 128-d Biometrics
- Model Architecture: Multi-Layer Perceptron & Ensemble Classifier
"""

import os
import sys
import argparse
import joblib
import cv2
import numpy as np

# Ensure root workspace directory is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from app.vision.face_engine import FaceBiometricEngine


def build_fs_training_dataset(training_dir: str, engine: FaceBiometricEngine, max_samples_per_video: int = 80):
    """Extracts quality-gated face embeddings from images and videos in training_dir."""
    ref_photos = {
        '1': '1.jfif',
        '2': '2 p.jpg',
        '3': '3 p.jpg',
        '4': '4 p.jpg',
        '5': '5 p.jpg',
        '6': '6 v.jpg',
        '7': '7 p.jpg'
    }

    videos = {
        '1': '1.v.mp4',
        '2': '2 v.mp4',
        '3': '3 v.mp4',
        '4': '4 v.mp4',
        '5': '5 v.mp4',
        '6': '6 p.mp4',
        '7': '7 v.mp4'
    }

    X, y = [], []

    print(f"[*] Scanning training directory: {training_dir}")

    # 1. Process Reference Photos
    for id_key, fname in ref_photos.items():
        fpath = os.path.join(training_dir, fname)
        if not os.path.isfile(fpath):
            continue
        img = cv2.imread(fpath)
        if img is None:
            continue
        faces = engine.detect_faces(img)
        for f in faces:
            bx, by, bw, bh = [int(v) for v in f['bbox']]
            bx1, by1 = max(0, bx), max(0, by)
            bx2, by2 = min(img.shape[1], bx + bw), min(img.shape[0], by + bh)
            crop = img[by1:by2, bx1:bx2]
            viable, emb = engine.extract_face_embedding(crop, landmarks=f.get('landmarks'), full_frame=img)
            if np.any(emb) and np.linalg.norm(emb) > 0.1:
                # Weight reference photos so ground truth is strongly represented
                for _ in range(15):
                    X.append(emb)
                    y.append(int(id_key))

    # 2. Process Video Frames
    for id_key, fname in videos.items():
        fpath = os.path.join(training_dir, fname)
        if not os.path.isfile(fpath):
            continue
        cap = cv2.VideoCapture(fpath)
        frame_idx = 0
        extracted = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 2 == 0 and extracted < max_samples_per_video:
                faces = engine.detect_faces(frame)
                for f in faces:
                    if f.get('is_viable', True) or f.get('score', 0) > 0.70:
                        bx, by, bw, bh = [int(v) for v in f['bbox']]
                        bx1, by1 = max(0, bx), max(0, by)
                        bx2, by2 = min(frame.shape[1], bx + bw), min(frame.shape[0], by + bh)
                        if (bx2 - bx1) >= 28 and (by2 - by1) >= 28:
                            crop = frame[by1:by2, bx1:bx2]
                            viable, emb = engine.extract_face_embedding(crop, landmarks=f.get('landmarks'), full_frame=frame)
                            if np.any(emb) and np.linalg.norm(emb) > 0.1:
                                X.append(emb)
                                y.append(int(id_key))
                                extracted += 1
                                if extracted >= max_samples_per_video:
                                    break
            frame_idx += 1
        cap.release()
        print(f"    - Identity {id_key} ({fname}): extracted {extracted} video face crops")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y


def train_and_evaluate_fs_model(X: np.ndarray, y: np.ndarray, test_size: float = 0.20, seed: int = 42):
    """Splits dataset 80% train / 20% test, trains FS model ensemble, and returns trained pipeline & accuracy."""
    print("\n" + "=" * 60)
    print(f" DATASET SPLIT: {int((1 - test_size) * 100)}% TRAIN / {int(test_size * 100)}% TEST")
    print("=" * 60)
    print(f"Total Samples:  {X.shape[0]}")
    print(f"Feature Vector: {X.shape[1]}-dimensional embeddings")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    print(f"Training Set:   {X_train.shape[0]} samples")
    print(f"Testing Set:    {X_test.shape[0]} samples")

    print("\n[*] Training FS Ensemble Model (MLP + ExtraTrees + HistGradientBoosting)...")

    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=seed)
    et = ExtraTreesClassifier(n_estimators=200, random_state=seed)
    hgb = HistGradientBoostingClassifier(random_state=seed)

    model = make_pipeline(
        StandardScaler(),
        VotingClassifier(
            estimators=[('mlp', mlp), ('et', et), ('hgb', hgb)],
            voting='soft'
        )
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f" FS MODEL EVALUATION METRICS")
    print("=" * 60)
    print(f" TEST ACCURACY: {accuracy * 100:.2f}%")
    print("-" * 60)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, digits=4))

    return model, accuracy, X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser(description="Train FS Model on Training Dataset (80% Train, 20% Test)")
    parser.add_argument("--training-dir", type=str, default=os.path.join(ROOT_DIR, "Training"),
                        help="Path to Training folder containing images and videos")
    parser.add_argument("--test-split", type=float, default=0.20,
                        help="Fraction of data to use for testing (default: 0.20 = 20 percent)")
    parser.add_argument("--output-model", type=str, default=os.path.join(ROOT_DIR, "models", "fs_model.pkl"),
                        help="Path to save trained FS model pickle")
    parser.add_argument("--min-accuracy", type=float, default=0.85,
                        help="Minimum acceptable accuracy threshold")
    args = parser.parse_args()

    engine = FaceBiometricEngine()
    X, y = build_fs_training_dataset(args.training_dir, engine)

    if len(X) == 0:
        print("[!] Error: No training data could be extracted from Training folder.")
        sys.exit(1)

    model, acc, X_train, X_test, y_train, y_test = train_and_evaluate_fs_model(X, y, test_size=args.test_split)

    if acc >= args.min_accuracy:
        print(f"\n[✓] SUCCESS: Model target accuracy met! ({acc * 100:.2f}% >= {args.min_accuracy * 100:.0f}%)")
    else:
        print(f"\n[!] WARNING: Model accuracy ({acc * 100:.2f}%) below requested target ({args.min_accuracy * 100:.0f}%)")

    # Save trained model
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    joblib.dump({
        "model": model,
        "accuracy": acc,
        "classes": np.unique(y).tolist(),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "embedding_dim": X.shape[1]
    }, args.output_model)

    print(f"[*] Trained FS Model saved successfully to: {args.output_model}")


if __name__ == "__main__":
    main()
