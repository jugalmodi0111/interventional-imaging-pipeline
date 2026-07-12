"""TDD for the confirmed correctness bugs in src.eval.{metrics,calibration,cross_vendor}.

All torch-free: metrics run on plain numpy arrays. Covers:
  * dice/cldice -> NaN on empty GT, unchanged on normal (non-empty GT) cases
  * ece -> NaN on empty input, first bin still includes prob==0
  * coverage_risk -> no fake zero-risk point when nothing is kept
  * leave_one_vendor_out -> held-out atomic vendor truly excludes every dataset with it
"""
import math

import numpy as np
import pytest

from src.eval import metrics as M
from src.eval import calibration as C
from src.eval import cross_vendor as X


# ---- helpers ---------------------------------------------------------------
def _line_masks():
    """A non-empty GT (vertical bar) and a partial prediction of it."""
    g = np.zeros((64, 64), int); g[20:40, 30:33] = 1
    p = g.copy(); p[35:40, 30:33] = 0
    return p, g


# ---- dice: empty GT -> NaN -------------------------------------------------
def test_dice_nan_when_gt_and_pred_both_empty():
    z = np.zeros((16, 16), int)
    assert math.isnan(M.dice(z, z))


def test_dice_nan_when_gt_empty_but_pred_nonempty():
    gt = np.zeros((16, 16), int)
    pred = np.zeros((16, 16), int); pred[2:6, 2:6] = 1
    assert math.isnan(M.dice(pred, gt))


# ---- dice: normal (non-empty GT) behaviour unchanged -----------------------
def test_dice_perfect_overlap_is_one():
    _, g = _line_masks()
    assert M.dice(g, g) == pytest.approx(1.0)


def test_dice_empty_pred_vs_nonempty_gt_is_zero():
    _, g = _line_masks()
    pred = np.zeros_like(g)
    assert M.dice(pred, g) == pytest.approx(0.0, abs=1e-3)


def test_dice_partial_overlap_matches_formula():
    p, g = _line_masks()
    eps = 1e-6
    pb, gb = p.astype(bool), g.astype(bool)
    expected = (2 * (pb & gb).sum() + eps) / (pb.sum() + gb.sum() + eps)
    assert M.dice(p, g) == pytest.approx(expected)


def test_dice_exclude_empty_false_restores_legacy_fake_one():
    z = np.zeros((16, 16), int)
    assert M.dice(z, z, exclude_empty=False) == pytest.approx(1.0)


# ---- cldice: empty GT -> NaN, normal unchanged -----------------------------
def test_cldice_nan_when_gt_and_pred_both_empty():
    z = np.zeros((16, 16), int)
    assert math.isnan(M.cldice(z, z))


def test_cldice_nan_when_gt_empty_but_pred_nonempty():
    gt = np.zeros((16, 16), int)
    pred = np.zeros((16, 16), int); pred[2:6, 2:6] = 1
    assert math.isnan(M.cldice(pred, gt))


def test_cldice_legacy_empty_pair_scored_a_fake_one():
    # Documents the original bug: empty/empty scored ~1.0 (now gated behind exclude_empty=False).
    # exclude_empty=False reaches the skeletonize path -> needs skimage.
    pytest.importorskip("skimage")
    z = np.zeros((16, 16), int)
    assert M.cldice(z, z, exclude_empty=False) == pytest.approx(1.0, abs=1e-4)


def test_cldice_nonempty_gt_unchanged_by_new_param():
    pytest.importorskip("skimage")  # non-empty GT reaches skeletonize
    p, g = _line_masks()
    val = M.cldice(p, g)
    assert math.isfinite(val) and 0.0 < val <= 1.0
    # exclude_empty must not perturb the non-empty-GT path at all.
    assert M.cldice(p, g, exclude_empty=False) == pytest.approx(val)


def test_cldice_empty_pred_vs_nonempty_gt_is_zero():
    pytest.importorskip("skimage")  # non-empty GT reaches skeletonize
    _, g = _line_masks()
    pred = np.zeros_like(g)
    assert M.cldice(pred, g) == pytest.approx(0.0, abs=1e-3)


# ---- ece: empty -> NaN, first bin includes prob==0 -------------------------
def test_ece_nan_on_empty_input():
    assert math.isnan(C.ece([], []))


def test_ece_finite_on_normal_input():
    probs = [0.0, 0.2, 0.8, 1.0]
    labels = [0, 0, 1, 1]
    e = C.ece(probs, labels)
    assert math.isfinite(e) and e >= 0.0


def test_ece_first_bin_includes_prob_zero():
    # A single prob==0 that is wrong (label 1) must count -> ECE 1.0.
    # If the first bin used a strict '>' it would be dropped and ECE would be 0.0.
    assert C.ece([0.0], [1]) == pytest.approx(1.0)


# ---- coverage_risk: no fake zero-risk point --------------------------------
def test_coverage_risk_emits_none_not_zero_when_nothing_kept():
    # conf == 0.5 for every sample, so any threshold > 0.5 keeps nothing.
    probs = np.array([0.5, 0.5, 0.5])
    labels = np.array([0, 1, 0])
    out = C.coverage_risk(probs, labels, thresholds=[0.5, 0.9])

    kept, empty = out[0], out[1]
    # threshold 0.5 keeps everything -> a real float risk (2 of 3 wrong).
    assert kept["coverage"] == pytest.approx(1.0)
    assert kept["risk"] == pytest.approx(round(2 / 3, 3))
    # threshold 0.9 keeps nothing -> risk flagged None (NOT a fake 0.0), coverage 0.
    assert empty["coverage"] == pytest.approx(0.0)
    assert empty["risk"] is None


def test_coverage_risk_never_reports_fake_zero_risk_at_empty_coverage():
    probs = np.array([0.5, 0.5])
    labels = np.array([0, 1])
    out = C.coverage_risk(probs, labels, thresholds=[0.9, 0.99])
    for row in out:
        if row["coverage"] == 0.0:
            assert row["risk"] is None


# ---- leave_one_vendor_out: atomic-vendor exclusion -------------------------
def _folds(datasets):
    return {held: (train, ev) for train, held, ev in X.leave_one_vendor_out(datasets)}


def test_vendor_splits_are_atomic_sets():
    assert X.VENDOR_SPLITS["danilov"] == {"siemens", "ge"}
    assert X.VENDOR_SPLITS["arcade"] == {"philips", "siemens"}
    assert all(isinstance(v, set) for v in X.VENDOR_SPLITS.values())


def test_holding_out_siemens_excludes_every_dataset_with_siemens():
    # danilov is siemens+ge: the old composite logic left it in train while
    # testing 'siemens_ge' -> siemens leaked. It must now be excluded from train.
    train, ev = _folds(["arcade", "xcad", "danilov"])["siemens"]
    assert train == ["xcad"]                      # only dataset with no siemens
    assert set(ev) == {"arcade", "danilov"}       # both siemens carriers held out
    assert "danilov" not in train and "arcade" not in train


def test_holding_out_ge_excludes_every_dataset_with_ge():
    train, ev = _folds(["arcade", "xcad", "danilov"])["ge"]
    assert train == ["arcade"]                    # only dataset with no ge
    assert set(ev) == {"xcad", "danilov"}


def test_holding_out_philips():
    train, ev = _folds(["arcade", "xcad", "danilov"])["philips"]
    assert ev == ["arcade"]
    assert set(train) == {"xcad", "danilov"}


def test_held_vendor_never_appears_in_any_train_dataset():
    datasets = ["arcade", "dca1", "xcad", "danilov"]
    for train, held, ev in X.leave_one_vendor_out(datasets):
        # invariant: no training dataset carries the held-out atomic vendor
        assert all(held not in X.VENDOR_SPLITS[d] for d in train)
        # and every eval dataset does carry it
        assert all(held in X.VENDOR_SPLITS[d] for d in ev)
        assert set(train).isdisjoint(ev)


def test_requires_at_least_two_distinct_vendors():
    with pytest.raises(AssertionError):
        X.leave_one_vendor_out(["xcad"])          # {ge} only -> 1 vendor
    with pytest.raises(AssertionError):
        X.leave_one_vendor_out(["dca1"])          # {imss} only -> 1 vendor


def test_report_gap_unchanged():
    assert X.report_gap(0.78, 0.71) == pytest.approx(0.07)
