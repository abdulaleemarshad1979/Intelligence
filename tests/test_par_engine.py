"""Benchmark & unit tests for Pedestrian Attribute Recognition (PAR) Engine.

Tests:
- UniPAR / SequencePAR visual-textual parsing
- Backpack and carried accessory localization
- Clothing color classification
- Jaccard attribute similarity calculation
- Asymmetric focal loss (LASL) handling for long-tailed surveillance classes
"""

import pytest
import numpy as np
import cv2
from app.vision.par_engine import AttributeParsingEngine


@pytest.fixture
def par_engine():
    return AttributeParsingEngine()


def test_par_labels_specification(par_engine):
    """Verify that all required municipal surveillance attribute classes are defined."""
    expected = [
        "backpack", "single_shoulder_bag", "handbag", "carrying_box",
        "upper_black", "upper_white", "upper_blue", "upper_red",
        "lower_black", "lower_blue", "lower_white",
        "wearing_hat", "wearing_mask", "wearing_glasses"
    ]
    for label in expected:
        assert label in par_engine.ATTRIBUTE_LABELS


def test_backpack_and_clothing_parsing(par_engine):
    """Verify attribute parsing on a synthetic pedestrian crop with dark torso and blue trousers."""
    # Synthetic pedestrian: 256x128 BGR image
    crop = np.full((256, 128, 3), 30, dtype=np.uint8)  # dark background

    # Upper torso: Dark / black jacket (HSV value < 60)
    crop[50:150, 20:108] = (20, 20, 20)

    # Lower wear: Blue denim trousers (BGR blue channel high, red/green low)
    crop[150:240, 25:103] = (160, 40, 20)  # B=160, G=40, R=20 -> Blue

    # Backpack straps on lateral flanks (high-contrast vertical edge lines)
    cv2.line(crop, (28, 60), (28, 130), (220, 220, 220), 2)
    cv2.line(crop, (100, 60), (100, 130), (220, 220, 220), 2)

    result = par_engine.parse_attributes(crop, threshold=0.5)

    assert "attributes" in result
    assert "bitmask" in result
    assert len(result["bitmask"]) == len(par_engine.ATTRIBUTE_LABELS)

    # Torso was black, lower was blue
    assert result["attributes"]["upper_black"]["confidence"] > 0.4
    assert result["attributes"]["lower_blue"]["confidence"] > 0.4

    # High flank edge activity should detect backpack straps
    assert result["attributes"]["backpack"]["confidence"] > 0.5
    assert result["has_backpack"] is True


def test_jaccard_attribute_similarity(par_engine):
    """Test Jaccard metric S_attr = |A ∩ B| / |A ∪ B|."""
    # Identical attributes
    attrs1 = ["backpack", "upper_black", "lower_blue"]
    attrs2 = ["backpack", "upper_black", "lower_blue"]
    assert par_engine.compute_jaccard_similarity(attrs1, attrs2) == 1.0

    # Disjoint attributes
    attrs3 = ["handbag", "upper_red", "lower_white"]
    assert par_engine.compute_jaccard_similarity(attrs1, attrs3) == 0.0

    # Partial overlap: intersection = 2, union = 4 -> 0.5
    attrs4 = ["backpack", "upper_black", "wearing_hat", "wearing_glasses"]
    # union with attrs1: backpack, upper_black, lower_blue, wearing_hat, wearing_glasses (5 items)
    # intersection: backpack, upper_black (2 items) -> 2/5 = 0.4
    sim = par_engine.compute_jaccard_similarity(attrs1, attrs4)
    assert pytest.approx(sim, 0.01) == 0.4

    # Both empty -> consistent 1.0
    assert par_engine.compute_jaccard_similarity([], []) == 1.0


def test_asymmetric_focal_loss_computation(par_engine):
    """Test Asymmetric Focal Loss (LASL) suppressing easy negatives with margin m."""
    y_true = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    # Accurate predictions: positives have high p, negatives have low p (<= margin)
    y_pred_good = np.array([0.95, 0.02, 0.03, 0.92], dtype=np.float32)

    # Inaccurate predictions: positive missed, negative false alarm
    y_pred_bad = np.array([0.20, 0.85, 0.05, 0.30], dtype=np.float32)

    loss_good = par_engine.compute_asymmetric_loss(y_true, y_pred_good, gamma_pos=0.0, gamma_neg=4.0, margin=0.05)
    loss_bad = par_engine.compute_asymmetric_loss(y_true, y_pred_bad, gamma_pos=0.0, gamma_neg=4.0, margin=0.05)

    assert loss_good >= 0.0
    assert loss_bad > loss_good
    # Margin clipping should heavily suppress easy negative sample loss
    assert loss_good < 0.5


def test_empty_crop_handling(par_engine):
    """Test graceful handling of empty/None image crops."""
    res_none = par_engine.parse_attributes(None)
    assert res_none["has_backpack"] is False
    assert sum(res_none["bitmask"]) == 0

    res_empty = par_engine.parse_attributes(np.array([]))
    assert res_empty["has_backpack"] is False
