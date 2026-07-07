from __future__ import annotations

from analysis.counter_uav_accuracy_audit import (
    analyze_counter_uav_classifier_failure,
    analyze_generator_separability,
)


def test_bird_recall_stays_reasonable_under_strong_multipath():
    summary = analyze_counter_uav_classifier_failure()
    assert summary["bird_recall_nwalls_4_6"] >= 0.45


def test_bird_overall_recall_stays_above_demo_floor():
    summary = analyze_counter_uav_classifier_failure()
    assert summary["bird_recall_overall"] >= 0.50


def test_multipath_mixing_preserves_bird_drone_separability():
    summary = analyze_generator_separability()
    assert summary["multipath_sep_lr_score"] >= 0.70
