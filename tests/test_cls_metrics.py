"""Binary classification metrics for Model One (B3). Pure numpy -- no torch anywhere.

sensitivity/specificity are the B-requirement vocabulary (screening triage), deliberately not
precision/recall aliases. threshold_at_sensitivity implements B3's 'never a false normal' posture:
pick the operating point from a *sensitivity* target, then report the specificity you got.
"""
import numpy as np
import pytest

from src.eval.cls_metrics import (bootstrap_ci, confusion_counts, sensitivity, specificity,
                                  threshold_at_sensitivity)

PROBS = np.array([0.9, 0.8, 0.6, 0.4, 0.2, 0.1])
LABELS = np.array([1, 1, 0, 1, 0, 0])


def test_confusion_counts_at_half():
    c = confusion_counts(PROBS, LABELS, 0.5)
    assert c == {"tp": 2, "fp": 1, "tn": 2, "fn": 1}


def test_sensitivity_and_specificity_at_half():
    assert sensitivity(PROBS, LABELS, 0.5) == pytest.approx(2 / 3)
    assert specificity(PROBS, LABELS, 0.5) == pytest.approx(2 / 3)


def test_degenerate_inputs_return_zero_not_nan():
    assert sensitivity(PROBS, np.zeros(6), 0.5) == 0.0      # no positives to be sensitive to
    assert specificity(PROBS, np.ones(6), 0.5) == 0.0       # no negatives


def test_threshold_at_sensitivity_hits_target():
    thr = threshold_at_sensitivity(PROBS, LABELS, 1.0)      # must catch ALL positives
    assert sensitivity(PROBS, LABELS, thr) == 1.0
    assert thr <= 0.4                                        # the 0.4 positive must clear it


def test_threshold_none_target_falls_back_to_half():
    assert threshold_at_sensitivity(PROBS, LABELS, None) == 0.5


def test_bootstrap_ci_brackets_the_point_estimate_and_is_deterministic():
    lo, hi = bootstrap_ci(sensitivity, PROBS, LABELS, n_boot=200, seed=7)
    lo2, hi2 = bootstrap_ci(sensitivity, PROBS, LABELS, n_boot=200, seed=7)
    assert (lo, hi) == (lo2, hi2)
    assert lo <= sensitivity(PROBS, LABELS, 0.5) <= hi
