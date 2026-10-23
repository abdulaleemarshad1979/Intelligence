"""Accuracy harness: false-match and miss rates for any scorer on labelled comparisons.

Input is a list of trials, each with a similarity score and a ground-truth flag saying whether
the two items really are the same person. The harness never decides identity; it measures how
often a given threshold would have been wrong on footage with known answers.
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class Trial:
    score: float
    same_person: bool
    condition: str = "all"   # e.g. "frontal-indoor", "night", "oblique"


def rates_at(trials: List[Trial], threshold: float) -> Dict[str, Any]:
    """False match rate (different people scored >= threshold) and false non-match rate (same person below)."""
    genuine = [t for t in trials if t.same_person]
    impostor = [t for t in trials if not t.same_person]
    fm = sum(t.score >= threshold for t in impostor)
    fnm = sum(t.score < threshold for t in genuine)
    return {
        "threshold": threshold, "genuine_trials": len(genuine), "impostor_trials": len(impostor),
        "false_matches": fm, "false_match_rate": fm / len(impostor) if impostor else None,
        "misses": fnm, "miss_rate": fnm / len(genuine) if genuine else None,
    }


def _rule_of_three(n: int) -> Optional[float]:
    """95% upper bound on a rate when 0 errors were seen in n trials."""
    return 3.0 / n if n else None


def evaluate(trials: Iterable[Trial], thresholds: Optional[List[float]] = None) -> Dict[str, Any]:
    trials = list(trials)
    if not trials:
        raise ValueError("No trials supplied")
    scores = sorted({t.score for t in trials})
    thresholds = thresholds or [round(x / 100, 2) for x in range(0, 101, 5)]
    table = [rates_at(trials, th) for th in thresholds]
    eer = min(
        (r for r in (rates_at(trials, s) for s in scores) if r["false_match_rate"] is not None and r["miss_rate"] is not None),
        key=lambda r: abs(r["false_match_rate"] - r["miss_rate"]), default=None)
    n_imp = sum(not t.same_person for t in trials)
    n_gen = sum(t.same_person for t in trials)
    zero_fm = [r for r in table if r["false_matches"] == 0 and r["false_match_rate"] is not None]
    return {
        "trials": len(trials), "genuine": n_gen, "impostor": n_imp,
        "equal_error_point": eer,
        "lowest_threshold_with_no_false_match": min((r["threshold"] for r in zero_fm), default=None),
        "false_match_rate_95pct_upper_bound_if_zero": _rule_of_three(n_imp),
        "table": table,
        "caveat": "Rates describe only the footage and people in these trials. With few impostor "
                  "trials the true false-match rate can be far higher than the observed one.",
    }


def evaluate_by_condition(trials: Iterable[Trial], threshold: float) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Trial]] = defaultdict(list)
    for t in trials:
        groups[t.condition].append(t)
    return {k: rates_at(v, threshold) for k, v in sorted(groups.items())}


def load_trials(path: str) -> List[Trial]:
    """JSON list of {"score": float, "same_person": bool, "condition": str}."""
    with open(path) as f:
        return [Trial(float(r["score"]), bool(r["same_person"]), r.get("condition", "all")) for r in json.load(f)]


def sface_trials(references: Dict[str, str], clips: List[Dict[str, Any]], models_dir: str,
                 stride: int = 3) -> List[Trial]:
    """Score reference photos against video frames with OpenCV SFace.

    references: {person: image_path}; clips: [{"path": ..., "person": ..., "condition": ...}].
    Every (reference, face-bearing frame) pair becomes a trial; same_person is the ground truth label.
    """
    import os
    import cv2
    det = cv2.FaceDetectorYN.create(os.path.join(models_dir, "face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.6)
    rec = cv2.FaceRecognizerSF.create(os.path.join(models_dir, "face_recognition_sface_2021dec.onnx"), "")

    def embed(img):
        det.setInputSize((img.shape[1], img.shape[0]))
        _, faces = det.detect(img)
        if faces is None:
            return None
        return rec.feature(rec.alignCrop(img, max(faces, key=lambda f: f[2])))

    ref_emb = {}
    for person, p in references.items():
        e = embed(cv2.imread(p))
        if e is None:
            raise ValueError(f"No face found in reference for {person}")
        ref_emb[person] = e
    trials: List[Trial] = []
    for clip in clips:
        cap, n = cv2.VideoCapture(clip["path"]), 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            if n % stride:
                continue
            e = embed(frame)
            if e is None:
                continue
            for person, re_ in ref_emb.items():
                trials.append(Trial(float(rec.match(re_, e, cv2.FaceRecognizerSF_FR_COSINE)),
                                    person == clip["person"], clip.get("condition", "all")))
        cap.release()
    return trials
