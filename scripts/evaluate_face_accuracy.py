"""Measure false-match and miss rates on labelled data.

Usage:
  python scripts/evaluate_face_accuracy.py trials.json
  python scripts/evaluate_face_accuracy.py --manifest manifest.json --models models/
manifest.json: {"references": {"A": "a.jpg"}, "clips": [{"path": "a.mp4", "person": "A", "condition": "frontal-indoor"}]}
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.casework.evaluation import evaluate, evaluate_by_condition, load_trials, sface_trials  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trials", nargs="?")
    ap.add_argument("--manifest")
    ap.add_argument("--models", default="models")
    ap.add_argument("--threshold", type=float, default=0.363)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.manifest:
        m = json.load(open(a.manifest))
        trials = sface_trials(m["references"], m["clips"], a.models)
    elif a.trials:
        trials = load_trials(a.trials)
    else:
        ap.error("give trials.json or --manifest")
    report = evaluate(trials)
    report["by_condition_at_threshold"] = {"threshold": a.threshold, **evaluate_by_condition(trials, a.threshold)}
    text = json.dumps(report, indent=2)
    open(a.out, "w").write(text) if a.out else print(text)


if __name__ == "__main__":
    main()
