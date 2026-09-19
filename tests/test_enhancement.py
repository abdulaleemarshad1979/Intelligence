"""Tests for CCTV footage enhancement (Step 1 always-on and Step 2 on-demand)."""

import cv2
import numpy as np
from app.features.enhancement import StreamEnhancer, CandidateEnhancer, stream_enhancer, candidate_enhancer


def test_clahe_enhances_dark_frame():
    # Create dark synthetic image with subtle low-contrast gradient
    dark_frame = np.full((120, 160, 3), 25, dtype=np.uint8)
    cv2.rectangle(dark_frame, (30, 30), (80, 80), (45, 45, 45), -1)

    enhancer = StreamEnhancer(clahe_clip_limit=3.0, enable_clahe=True, enable_denoise=False)
    enhanced = enhancer.enhance_stream_frame(dark_frame)

    assert enhanced.shape == dark_frame.shape
    # CLAHE should increase contrast / dynamic range (standard deviation of pixel values)
    assert np.std(enhanced) >= np.std(dark_frame)
    # Brightness should be lifted in the dark areas
    assert np.mean(enhanced) > np.mean(dark_frame)


def test_fast_denoising():
    # Create synthetic frame with random high-ISO noise
    clean_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    noise = np.random.normal(0, 15, clean_frame.shape).astype(np.int16)
    noisy_frame = np.clip(clean_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    enhancer = StreamEnhancer(enable_clahe=False, enable_denoise=True)
    denoised = enhancer.enhance_stream_frame(noisy_frame)

    assert denoised.shape == noisy_frame.shape
    # Variance of noisy flat region should decrease after bilateral denoising
    assert np.var(denoised) < np.var(noisy_frame)


def test_candidate_face_enhancer():
    # Small blurry face crop (32x32)
    small_face = np.full((32, 32, 3), 120, dtype=np.uint8)
    cv2.circle(small_face, (16, 16), 8, (180, 180, 180), -1)

    enhancer = CandidateEnhancer(target_face_size=(160, 160))
    restored = enhancer.enhance_face_crop(small_face)

    assert restored.shape == (160, 160, 3)
    assert restored.dtype == np.uint8


def test_candidate_body_enhancer():
    # Person crop (80x40)
    person_crop = np.full((80, 40, 3), 100, dtype=np.uint8)
    enhancer = CandidateEnhancer()
    enhanced_body = enhancer.enhance_body_crop(person_crop, scale_factor=1.5)

    assert enhanced_body.shape == (120, 60, 3)
    assert enhanced_body.dtype == np.uint8


def test_empty_frames_handled_gracefully():
    enhancer = StreamEnhancer()
    assert enhancer.enhance_stream_frame(None) is None
    empty_arr = np.array([])
    assert enhancer.enhance_stream_frame(empty_arr).size == 0
