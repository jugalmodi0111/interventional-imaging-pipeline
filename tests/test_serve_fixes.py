"""Unit tests for the serve-layer metric fixes (infer.py).

All torch-/ultralytics-/coreml-free: they exercise the PURE helpers extracted for exactly this
reason. Each test pins one of the confirmed metric bugs so a regression re-surfaces loudly.

(The track.py metric tests that used to live here -- fragmentation, weighted fps, flat-numbering
detection -- were deleted with track.py itself: the video/tracking path is not part of Model One
per the 2026-08-03 audit P3 decision.)
"""
import numpy as np
import pytest

from src.serve.infer import seg_confidence


# --- bug 5: two-sided segmentation confidence consistent with coverage_risk ----------

def test_seg_confidence_uniform_half_defers():
    prob = np.full((32, 32), 0.5)
    c = seg_confidence(prob)
    assert c == pytest.approx(0.5)
    assert c < 0.55            # defer_below default -> DEFERS on max uncertainty


def test_seg_confidence_confident_map_keeps():
    prob = np.where(np.arange(100) < 50, 0.98, 0.02).astype(np.float64)  # crisp fg AND bg
    c = seg_confidence(prob)
    assert c == pytest.approx(0.98)
    assert c >= 0.55           # confident -> KEEPS (does not defer)


def test_seg_confidence_is_two_sided_counts_confident_background():
    # A tiny confident foreground amid a confident background. The old one-sided score averaged
    # only the fg pixels; the two-sided score credits the confident background too.
    prob = np.full(1000, 0.01)
    prob[:5] = 0.99            # 5 confident fg pixels, 995 confident bg pixels
    c = seg_confidence(prob)
    assert c == pytest.approx(0.99, abs=1e-9)   # ~1.0: whole map is confident, so KEEP
    assert 0.5 <= c <= 1.0


def test_seg_confidence_in_unit_range_for_random_maps():
    rng = np.random.default_rng(0)
    for _ in range(20):
        c = seg_confidence(rng.uniform(size=(16, 16)))
        assert 0.5 <= c <= 1.0
