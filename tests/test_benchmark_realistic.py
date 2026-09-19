"""Unit tests for the Realistic CCTV Evaluation & Benchmarking Harness."""

import pytest
from app.database.database import init_db
from app.database.repository import Repository
from app.reid.gallery import SuspectGallery
from app.evaluation.benchmark import ReIDEvaluationHarness, BenchmarkReport
from scripts.benchmark_models import generate_independent_evaluation_dataset


def test_independent_benchmark_harness():
    init_db()
    repo = Repository()
    gallery_mgr = SuspectGallery(repo)
    gallery = gallery_mgr.get_all_suspects()
    assert len(gallery) > 0

    queries, ground_truth = generate_independent_evaluation_dataset(gallery)
    assert len(queries) > 0
    assert len(ground_truth) == len(queries)

    # Check distractors exist
    distractors = [p for p in ground_truth if p[1] is None]
    assert len(distractors) >= 4

    # Run harness
    harness = ReIDEvaluationHarness()
    report = harness.evaluate_gallery(queries, ground_truth, gallery)

    assert isinstance(report, BenchmarkReport)
    assert report.total_query_tracks == len(queries)
    assert report.positive_queries > 0
    assert report.distractor_impostor_queries == len(distractors)
    assert 0.0 <= report.cmc_rank1 <= 1.0
    assert 0.0 <= report.mean_average_precision <= 1.0
    assert 0.0 <= report.false_match_rate <= 1.0
    assert "face_only" in report.ablation_results
    assert "our_dynamic_fusion" in report.ablation_results
    assert len(report.viewpoint_results) >= 5
    assert "reid_models" in report.model_comparisons
    assert "tracking_models" in report.model_comparisons
    assert "gait_models" in report.model_comparisons
