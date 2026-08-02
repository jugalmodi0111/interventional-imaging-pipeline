"""Turn raw model output + the existing triage-with-abstention into typed Findings, and compute the
study-level defer. A below-floor model (entry.floor_ok=False) ALWAYS defers its finding, regardless
of how confident the detector was — a model that hasn't cleared its own bar cannot assert a positive.

Safety default is DEFER, not guess. Two corollaries enforced throughout this module:
  * A triage/seg-result dict that is missing the keys the real producers always populate
    ('deferred', 'reason', 'confidence') is malformed input, not a confident negative -- it must
    fail safe to a deferred finding, never default toward 'clean'/'confident'.
  * An empty findings list is not the same thing as a clean study -- it means nothing screened the
    study (e.g. registry.resolve() found no TaskEntry for the modality), so the study must defer.
"""
from src.serve.report import Finding


def det_to_findings(entry, triage):
    """Detector triage dict -> [Finding]. One finding per modality (the anchor disease), carrying the
    kept boxes. Below-floor forces deferred + reason 'below-floor'. A triage dict missing 'deferred'
    or 'reason' is malformed/incomplete input -- rather than defaulting to a confident 'clean' finding,
    it fails safe to deferred + reason 'malformed-triage'."""
    preds = triage.get("prediction") or []
    confs = triage.get("calibrated_confs") or []
    top = max(confs) if confs else 0.0
    if not entry.floor_ok:
        deferred, reason = True, "below-floor"
    elif "deferred" not in triage or "reason" not in triage:
        deferred, reason = True, "malformed-triage"
    else:
        deferred, reason = bool(triage["deferred"]), triage["reason"]
    return [Finding(label=entry.finding_label, display_name=entry.finding_display,
                    confidence=float(top), deferred=deferred, reason=reason,
                    boxes=[tuple(p) for p in preds])]


def seg_to_finding(entry, seg_res):
    """Segmentation result dict -> a single anatomy/finding entry. Seg is context, not a positive Dx,
    so it never asserts disease; it defers if the model deferred (low confidence). A seg_res missing
    'deferred' or 'confidence' is malformed/incomplete input -- it fails safe to deferred + reason
    'malformed-seg' rather than defaulting to a confidence-0.0 'confident' finding."""
    if not entry.floor_ok:
        deferred, reason = True, "below-floor"
    elif "deferred" not in seg_res or "confidence" not in seg_res:
        deferred, reason = True, "malformed-seg"
    else:
        deferred = bool(seg_res["deferred"])
        reason = "low-confidence" if deferred else "confident"
    return Finding(label=entry.finding_label, display_name=entry.finding_display,
                   confidence=float(seg_res.get("confidence", 0.0)), deferred=deferred, reason=reason)


def study_defer(decision, findings):
    """Study defers if the router deferred/modality unknown, findings is empty (nothing ever screened
    the study -- e.g. registry.resolve() returned None for the modality so no det/seg finding was
    produced), or ANY finding deferred. Returns the most-fundamental reason first (router distrust >
    unscreened study > per-finding distrust)."""
    if decision.deferred or decision.modality in ("unknown", ""):
        return True, decision.reason
    if not findings:
        return True, "no-findings"
    for f in findings:
        if f.deferred:
            return True, f.reason
    return False, ""
