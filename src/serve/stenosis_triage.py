"""Triage-with-abstention for the stenosis detector. It is BELOW the ship floor (F1 ~0.2), so it
must NOT act as an autonomous classifier. Instead it runs as high-recall *screening* that DEFERS
the uncertain / out-of-distribution cases to a human rather than being silently wrong.

The whole layer is biased toward DEFERRING whenever a stenosis might be present: a missed stenosis
is the deadly error, so 'looks clean' is only allowed when the model is genuinely confident there is
nothing there. 'Wrong but confident' is the dangerous mode.

Pure functions, numpy only — the decision layer sits on top of src.eval.calibration.
"""
import numpy as np

from src.eval.calibration import apply_temperature, coverage_risk


def calibrate_confidences(confs, temperature=1.0, eps=1e-6):
    """Temperature-scale raw detector confidences into calibrated probabilities.

    Detector confidences are already probabilities in [0,1], so we invert to logits and reuse
    calibration.apply_temperature (sigmoid(logit/T)) to keep identical semantics with the seg path.
    T==1.0 is an exact no-op (sigmoid(logit(p)) == p); T>1 cools over-confidence, T<1 sharpens.
    """
    p = np.clip(np.asarray(confs, dtype=float).ravel(), eps, 1.0 - eps)
    logits = np.log(p / (1.0 - p))
    return apply_temperature(logits, temperature)


def _conf_of(det):
    """Pull the confidence out of a detection. Supports the serve-layer box tuple
    (x1,y1,x2,y2,conf), a bare float, or a dict with conf/confidence/score."""
    if isinstance(det, dict):
        for k in ("conf", "confidence", "score"):
            if k in det:
                return float(det[k])
        raise KeyError("detection dict has no conf/confidence/score key")
    if isinstance(det, (int, float, np.floating)):
        return float(det)
    return float(det[-1])                          # (x1,y1,x2,y2,conf) box


def triage_decision(detections, *, temperature=1.0, keep_conf=0.5, defer_band=(0.3, 0.6),
                    ood_score=None, ood_thr=0.5, near_band_margin=0.1):
    """Decide keep / defer for one frame's stenosis detections. Recall-protective by design.

    Returns dict(prediction, calibrated_confs, deferred, reason). `prediction` lists the detections
    whose CALIBRATED confidence clears `keep_conf` (surfaced even when we also defer — never hide a
    possible stenosis from the human).

    Defers (deferred=True), in priority order of most-fundamental distrust first:
      * 'ood' — `ood_score` given and >= `ood_thr`: unfamiliar vendor/view/artifact, so even the
        calibrated confidences are untrustworthy. Defer regardless of what the detector said.
      * 'low-confidence' — the top calibrated confidence lands inside `defer_band`, straddling the
        decision boundary: the 'wrong but confident' danger zone. Defer even if it was kept.
      * 'no-detection-uncertain' — nothing was kept, yet a sub-threshold detection sits just below
        the band (within `near_band_margin`): a possible MISSED stenosis. Defer instead of calling
        it clean.
    Otherwise keeps: reason 'confident' if a detection was kept, else 'clean' (confident negative).
    """
    lo, hi = defer_band
    cal = calibrate_confidences([_conf_of(d) for d in detections], temperature) \
        if len(detections) else np.empty(0)
    prediction = [d for d, c in zip(detections, cal) if c >= keep_conf]
    max_cal = float(cal.max()) if cal.size else 0.0

    if ood_score is not None and float(ood_score) >= ood_thr:
        deferred, reason = True, "ood"
    elif cal.size and lo <= max_cal <= hi:
        deferred, reason = True, "low-confidence"
    elif not prediction and cal.size and max_cal >= lo - near_band_margin:
        # No kept detection, but the best one is faint-near-the-band -> treat as a possible miss.
        deferred, reason = True, "no-detection-uncertain"
    else:
        deferred, reason = False, ("confident" if prediction else "clean")

    return {"prediction": prediction, "calibrated_confs": [float(c) for c in cal],
            "deferred": deferred, "reason": reason}


def coverage_risk_report(confs, labels, *, temperature=1.0, thresholds=None, target_risk=None):
    """Risk-vs-coverage sweep on the stenosis operating points, so a defer threshold can be picked
    from a target risk. Thin wrapper over calibration.coverage_risk (confidences calibrated first).

    Returns dict(curve=[...]). If `target_risk` is given, also returns `operating_point`: the
    highest-coverage sweep point whose risk <= target_risk (most cases auto-handled at acceptable
    risk), or None if no threshold meets it.
    """
    cal = calibrate_confidences(confs, temperature)
    curve = coverage_risk(cal, labels, thresholds)
    report = {"curve": curve}
    if target_risk is not None:
        feasible = [r for r in curve if r["risk"] is not None and r["risk"] <= target_risk]
        report["operating_point"] = max(feasible, key=lambda r: r["coverage"]) if feasible else None
        report["target_risk"] = float(target_risk)
    return report


if __name__ == "__main__":
    print("confident +:", triage_decision([(0, 0, 10, 10, 0.92)])["reason"])            # keep
    print("boundary  :", triage_decision([(0, 0, 10, 10, 0.55)])["reason"])             # low-confidence
    print("ood       :", triage_decision([(0, 0, 10, 10, 0.92)], ood_score=0.8)["reason"])  # ood
    print("near-miss :", triage_decision([(0, 0, 10, 10, 0.25)])["reason"])             # no-detection-uncertain
    print("clean     :", triage_decision([(0, 0, 10, 10, 0.03)])["reason"])             # clean
    rng = np.random.default_rng(0)
    p = rng.uniform(size=500); y = (rng.uniform(size=500) < p).astype(int)
    print("op@risk<=0.2:", coverage_risk_report(p, y, target_risk=0.2)["operating_point"])
