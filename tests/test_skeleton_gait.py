"""Unit tests for app/features/skeleton_gait.py - Skeleton Biometrics & Gait Signature Engine."""

import numpy as np
import pytest

from app.features.skeleton_gait import (
    GaitConfig,
    GaitSignature,
    GaitProfile,
    MatchResult,
    SkeletonGaitMatcher,
    GaitTrackBuffer,
    extract_signature,
    build_profile,
    enroll_person,
    signatures_from_sequence,
    LSH, RSH, LEL, REL, LWR, RWR,
    LHIP, RHIP, LKN, RKN, LAN, RAN,
    NOSE
)


def _generate_synthetic_pose_sequence(
    num_frames: int = 100,
    fps: float = 25.0,
    view: str = "oblique",
    is_walking: bool = True,
    leg_len: float = 80.0,
    torso_len: float = 60.0
) -> np.ndarray:
    """Generate synthetic COCO-17 keypoint sequence (T, 17, 3)."""
    kpts = np.zeros((num_frames, 17, 3), dtype=np.float32)

    # Base coordinates
    mid_x = 100.0
    torso_top = 40.0
    hip_y = torso_top + torso_len
    thigh_len = leg_len * 0.5
    shin_len = leg_len * 0.5
    arm_len = leg_len * 0.75
    upper_arm = arm_len * 0.5
    fore_arm = arm_len * 0.5

    sh_width = 30.0 if view == "frontal" else (18.0 if view == "oblique" else 5.0)
    hip_width = 25.0 if view == "frontal" else (15.0 if view == "oblique" else 4.0)

    for t in range(num_frames):
        # Stride motion
        phase = 2.0 * np.pi * (t / fps) * 1.5  # ~1.5 Hz walking cadence (90 steps/min)
        stride_offset = (leg_len * 0.3 * np.sin(phase)) if is_walking else 0.0
        arm_offset = (-leg_len * 0.25 * np.sin(phase)) if is_walking else 0.0

        # Head / Neck
        kpts[t, NOSE] = [mid_x, torso_top - 15.0, 0.9]

        # Shoulders
        kpts[t, LSH] = [mid_x - sh_width / 2.0, torso_top, 0.9]
        kpts[t, RSH] = [mid_x + sh_width / 2.0, torso_top, 0.9]

        # Elbows & Wrists
        kpts[t, LEL] = [mid_x - sh_width / 2.0, torso_top + upper_arm, 0.9]
        kpts[t, REL] = [mid_x + sh_width / 2.0, torso_top + upper_arm, 0.9]
        kpts[t, LWR] = [mid_x - sh_width / 2.0 + arm_offset, torso_top + upper_arm + fore_arm, 0.9]
        kpts[t, RWR] = [mid_x + sh_width / 2.0 - arm_offset, torso_top + upper_arm + fore_arm, 0.9]

        # Hips
        kpts[t, LHIP] = [mid_x - hip_width / 2.0, hip_y, 0.9]
        kpts[t, RHIP] = [mid_x + hip_width / 2.0, hip_y, 0.9]

        # Knees
        kpts[t, LKN] = [mid_x - hip_width / 2.0 + stride_offset * 0.5, hip_y + thigh_len, 0.9]
        kpts[t, RKN] = [mid_x + hip_width / 2.0 - stride_offset * 0.5, hip_y + thigh_len, 0.9]

        # Ankles
        kpts[t, LAN] = [mid_x - hip_width / 2.0 + stride_offset, hip_y + thigh_len + shin_len, 0.9]
        kpts[t, RAN] = [mid_x + hip_width / 2.0 - stride_offset, hip_y + thigh_len + shin_len, 0.9]

    return kpts


def test_skeleton_extraction_and_ratios():
    cfg = GaitConfig()
    seq = _generate_synthetic_pose_sequence(num_frames=80, fps=25.0, view="oblique", is_walking=True)
    sig = extract_signature(seq, fps=25.0, cfg=cfg)

    assert sig is not None
    assert sig.view in ("frontal", "oblique", "side")
    assert "leg_torso" in sig.features
    assert "thigh_shin" in sig.features
    assert sig.quality > 0.3
    assert sig.duration_s > 2.0


def test_enrollment_and_matching():
    cfg = GaitConfig()
    # Person A sequence
    seq_a = _generate_synthetic_pose_sequence(num_frames=100, fps=25.0, leg_len=80.0, torso_len=60.0)
    profile_a = enroll_person([(seq_a, 25.0)], cfg=cfg)
    assert profile_a is not None
    assert profile_a.n_windows > 0

    # Probe from Person A (similar proportions)
    seq_probe_a = _generate_synthetic_pose_sequence(num_frames=80, fps=25.0, leg_len=81.0, torso_len=59.5)
    sig_probe_a = extract_signature(seq_probe_a, fps=25.0, cfg=cfg)
    assert sig_probe_a is not None

    matcher = SkeletonGaitMatcher(cfg=cfg)
    match_a = matcher.compare(profile_a, sig_probe_a)
    assert match_a is not None
    assert match_a.score > 0.75
    assert not match_a.veto


def test_disparity_veto_on_incompatible_skeleton():
    cfg = GaitConfig(skeleton_veto_sigmas=3.0)
    # Person A (normal ratio leg/torso ~ 80/60 = 1.33)
    seq_a = _generate_synthetic_pose_sequence(num_frames=100, fps=25.0, leg_len=80.0, torso_len=60.0)
    profile_a = enroll_person([(seq_a, 25.0)], cfg=cfg)

    # Incompatible Person B (very long torso, short legs: 40/80 = 0.5)
    seq_b = _generate_synthetic_pose_sequence(num_frames=80, fps=25.0, leg_len=45.0, torso_len=80.0)
    sig_b = extract_signature(seq_b, fps=25.0, cfg=cfg)
    assert sig_b is not None

    matcher = SkeletonGaitMatcher(cfg=cfg)
    res = matcher.compare(profile_a, sig_b)
    assert res is not None
    # Must trigger geometric veto due to severe bone-proportion mismatch
    assert res.veto is True


def test_gait_track_buffer_live_stream():
    cfg = GaitConfig(min_duration_s=2.0)
    buffer = GaitTrackBuffer(cfg=cfg, window_s=3.0, fps=25.0)

    seq = _generate_synthetic_pose_sequence(num_frames=60, fps=25.0)
    for i in range(len(seq)):
        t = i * (1.0 / 25.0)
        buffer.add("CAM-001", track_id=42, t=t, kpts=seq[i])

    sig = buffer.signature("CAM-001", track_id=42)
    assert sig is not None
    assert sig.duration_s >= 2.0
