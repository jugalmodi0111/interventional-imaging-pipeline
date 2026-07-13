"""Triage-with-abstention for the below-floor stenosis detector (src/serve/stenosis_triage.py).

Torch-/ultralytics-/coreml-free: pure-numpy decision layer over src.eval.calibration. Each test
pins one safety property so a regression (silently keeping an uncertain case, or calling a possible
miss 'clean') re-surfaces loudly. The whole layer is biased toward DEFERRING when a stenosis might
be present, because a MISS is the deadly error.
"""
import numpy as np
import pytest

from src.serve.stenosis_triage import (calibrate_confidences, coverage_risk_report,
                                        triage_decision)


# --- calibration wrapper: mirrors calibration.apply_temperature semantics -------------

def test_calibrate_confidences_temp_one_is_noop():
    confs = [0.05, 0.3, 0.55, 0.9]
    assert np.allclose(calibrate_confidences(confs, 1.0), confs, atol=1e-5)


def test_calibrate_confidences_high_temp_cools_toward_half():
    # T>1 must pull an over-confident score toward 0.5 (uncertainty), never past it.
    hot = float(calibrate_confidences([0.95], temperature=3.0)[0])
    assert 0.5 < hot < 0.95


# --- the four required triage properties ----------------------------------------------

def test_confident_positive_is_kept_not_deferred():
    d = triage_decision([(0, 0, 10, 10, 0.92)])
    assert d["deferred"] is False
    assert d["reason"] == "confident"
    assert len(d["prediction"]) == 1                 # surfaced to the operator


def test_boundary_band_case_defers_low_confidence():
    d = triage_decision([(0, 0, 10, 10, 0.55)])      # inside defer_band (0.3, 0.6)
    assert d["deferred"] is True
    assert d["reason"] == "low-confidence"
    # recall-protective: still surfaced even though we flag it for review
    assert len(d["prediction"]) == 1


def test_ood_case_defers_ood_even_when_confident():
    # A confident detection on an unfamiliar vendor/view: calibration itself is untrustworthy.
    d = triage_decision([(0, 0, 10, 10, 0.92)], ood_score=0.8, ood_thr=0.5)
    assert d["deferred"] is True
    assert d["reason"] == "ood"


def test_all_sub_threshold_defers_instead_of_empty_clean():
    # Nothing clears keep_conf, but a faint detection sits just below the band -> possible miss.
    d = triage_decision([(0, 0, 10, 10, 0.25), (0, 0, 5, 5, 0.22)])
    assert d["deferred"] is True
    assert d["reason"] == "no-detection-uncertain"
    assert d["prediction"] == []                     # not kept, but NOT called clean either


# --- guard rails around the keep / clean boundaries -----------------------------------

def test_genuinely_empty_or_noise_is_clean_not_deferred():
    assert triage_decision([])["reason"] == "clean"
    faint = triage_decision([(0, 0, 10, 10, 0.03)])  # well below the band -> confident negative
    assert faint["deferred"] is False and faint["reason"] == "clean"


def test_temperature_can_flip_a_confident_keep_into_a_defer():
    # Cooling an over-confident 0.62 into the band turns a silent keep into a human review.
    hot = triage_decision([(0, 0, 10, 10, 0.62)])                    # keep, above the band
    cooled = triage_decision([(0, 0, 10, 10, 0.62)], temperature=3.0)
    assert hot["deferred"] is False
    assert cooled["deferred"] is True and cooled["reason"] == "low-confidence"


# --- coverage/risk report drives the defer threshold ----------------------------------

def test_coverage_risk_report_returns_curve_and_operating_point():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=800)
    y = (rng.uniform(size=800) < p).astype(int)      # well-calibrated -> low risk achievable
    rep = coverage_risk_report(p, y, target_risk=0.3)
    assert isinstance(rep["curve"], list) and rep["curve"]
    op = rep["operating_point"]
    assert op is not None and op["risk"] <= 0.3       # abstaining buys an acceptable-risk point


def test_coverage_risk_report_operating_point_none_when_risk_target_infeasible():
    # Pure noise vs labels: no coverage level reaches a 0% risk target.
    rng = np.random.default_rng(1)
    p = rng.uniform(size=400)
    y = rng.integers(0, 2, size=400)
    rep = coverage_risk_report(p, y, target_risk=0.0)
    assert rep["operating_point"] is None
