"""Test the modality router decision layer."""
import pytest
from src.serve.router import decide_modality, ModalityDecision


def test_confident_top_class_is_kept():
    d = decide_modality({"coronary_angiography": 0.9, "cerebral_dsa": 0.05, "other_xray": 0.05})
    assert isinstance(d, ModalityDecision)
    assert d.modality == "coronary_angiography"
    assert d.deferred is False
    assert d.reason == "confident"


def test_below_keep_threshold_defers_unknown():
    d = decide_modality({"coronary_angiography": 0.5, "cerebral_dsa": 0.3, "other_xray": 0.2})
    assert d.modality == "unknown"
    assert d.deferred is True
    assert d.reason == "router-uncertain"


def test_thin_margin_between_top_two_defers():
    # top prob clears keep_thr but the runner-up is within `margin` -> ambiguous -> defer
    d = decide_modality({"coronary_angiography": 0.62, "cerebral_dsa": 0.55, "other_xray": 0.0},
                        keep_thr=0.60, margin=0.15)
    assert d.modality == "unknown"
    assert d.deferred is True
    assert d.reason == "router-uncertain"


def test_low_quality_flag_defers_even_if_confident_class():
    d = decide_modality({"coronary_angiography": 0.95, "other_xray": 0.05},
                        quality_prob=0.2, quality_thr=0.5)
    assert d.quality_ok is False
    assert d.deferred is True
    assert d.reason == "low-quality"


def test_reject_bucket_class_is_returned_not_unknown():
    # a confident non-medical image is a real, keepable classification (-> orchestrator will defer as unsupported)
    d = decide_modality({"non_medical": 0.97, "coronary_angiography": 0.03})
    assert d.modality == "non_medical"
    assert d.deferred is False
    assert d.reason == "confident"
