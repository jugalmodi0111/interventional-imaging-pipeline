"""Turn raw model output + the existing triage-with-abstention into typed Findings, and compute the
study-level defer. A below-floor model (entry.floor_ok=False) ALWAYS defers its finding, regardless
of how confident the detector was — a model that hasn't cleared its own bar cannot assert a positive."""
from src.serve.report import Finding


def det_to_findings(entry, triage):
    """Detector triage dict -> [Finding]. One finding per modality (the anchor disease), carrying the
    kept boxes. Below-floor forces deferred + reason 'below-floor'."""
    preds = triage.get("prediction") or []
    confs = triage.get("calibrated_confs") or []
    top = max(confs) if confs else 0.0
    deferred = bool(triage.get("deferred")) or (not entry.floor_ok)
    reason = "below-floor" if not entry.floor_ok else triage.get("reason", "clean")
    return [Finding(label=entry.finding_label, display_name=entry.finding_display,
                    confidence=float(top), deferred=deferred, reason=reason,
                    boxes=[tuple(p) for p in preds])]


def seg_to_finding(entry, seg_res):
    """Segmentation result dict -> a single anatomy/finding entry. Seg is context, not a positive Dx,
    so it never asserts disease; it defers if the model deferred (low confidence)."""
    deferred = bool(seg_res.get("deferred")) or (not entry.floor_ok)
    reason = "below-floor" if not entry.floor_ok else ("low-confidence" if deferred else "confident")
    return Finding(label=entry.finding_label, display_name=entry.finding_display,
                   confidence=float(seg_res.get("confidence", 0.0)), deferred=deferred, reason=reason)


def study_defer(decision, findings):
    """Study defers if the router deferred/modality unknown, or ANY finding deferred. Returns the
    most-fundamental reason first (router distrust > per-finding distrust)."""
    if decision.deferred or decision.modality in ("unknown", ""):
        return True, decision.reason
    for f in findings:
        if f.deferred:
            return True, f.reason
    return False, ""
