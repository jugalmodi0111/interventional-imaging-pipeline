"""Detector quality/SSL gates: F1 floor (qualifies_det, best_f1_from_pr) + SSL disjointness (ssl_enabled).

Torch-free: all three helpers are pure, so the F1 floor and the leak guard are unit-tested without
ultralytics/torch or a GPU val run.
"""
from src.train.train_detector import qualifies_det, best_f1_from_pr, ssl_enabled


# ---- qualifies_det: enforce the F1 floor -------------------------------------------------------

def test_qualifies_det_default_floor_is_0_57():
    assert qualifies_det({"f1": 0.60}, {}) is True          # no cfg -> default floor 0.57
    assert qualifies_det({"f1": 0.50}, {}) is False


def test_qualifies_det_floor_inclusive_at_boundary():
    assert qualifies_det({"f1": 0.57}, {}) is True          # exactly at floor passes


def test_qualifies_det_reads_cfg_target_f1():
    cfg = {"target": {"f1": 0.70}}
    assert qualifies_det({"f1": 0.71}, cfg) is True
    assert qualifies_det({"f1": 0.69}, cfg) is False


def test_qualifies_det_missing_or_null_f1_is_false():
    assert qualifies_det({}, {}) is False                   # no score -> treated as 0.0
    assert qualifies_det({"f1": None}, {}) is False


def test_qualifies_det_none_target_falls_back_to_default():
    assert qualifies_det({"f1": 0.58}, {"target": None}) is True


# ---- best_f1_from_pr: max F1 over the PR curve, div-by-zero safe --------------------------------

def test_best_f1_perfect_pr_is_one():
    assert best_f1_from_pr([1.0], [1.0]) == 1.0


def test_best_f1_p_eq_r_eq_zero_is_zero_no_div_error():
    assert best_f1_from_pr([0.0], [0.0]) == 0.0             # 2PR/(P+R) guarded


def test_best_f1_p_xor_r_zero_is_zero():
    assert best_f1_from_pr([1.0, 0.0], [0.0, 1.0]) == 0.0   # denom>0 but PR=0 -> 0


def test_best_f1_empty_curve_is_zero():
    assert best_f1_from_pr([], []) == 0.0


def test_best_f1_takes_max_over_curve():
    p = [0.9, 0.5, 0.1]                                      # F1s: 0.18, 0.50, 0.18
    r = [0.1, 0.5, 0.9]
    assert best_f1_from_pr(p, r) == 0.5


def test_best_f1_known_scalar_value():
    assert round(best_f1_from_pr([0.6], [0.6]), 6) == 0.6   # P=R=0.6 -> F1 0.6


# ---- ssl_enabled: disjointness guard (config-only) ---------------------------------------------

def test_ssl_enabled_no_dir_forces_both_off():
    assert ssl_enabled({"ssl": {"pseudo_label": True, "seed": "gdino"}}) == (False, False)
    assert ssl_enabled({"ssl": {"pseudo_label": True, "seed": "gdino", "unlabeled_dir": None}}) == (False, False)
    assert ssl_enabled({"ssl": {"pseudo_label": True, "unlabeled_dir": ""}}) == (False, False)
    assert ssl_enabled({}) == (False, False)


def test_ssl_enabled_dir_set_respects_cfg_flags():
    base = {"unlabeled_dir": "data/raw/xcad"}
    assert ssl_enabled({"ssl": dict(base, pseudo_label=True, seed="gdino")}) == (True, True)
    assert ssl_enabled({"ssl": dict(base, pseudo_label=True)}) == (True, False)
    assert ssl_enabled({"ssl": dict(base, seed="gdino")}) == (False, True)
    assert ssl_enabled({"ssl": dict(base)}) == (False, False)  # dir set but no flags on
