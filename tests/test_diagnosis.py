from src.serve.registry import TaskEntry
from src.serve.diagnosis import det_to_findings, seg_to_finding, study_defer
from src.serve.router import ModalityDecision

ENTRY = TaskEntry("coronary_angiography", "det", "best.pt", "Coronary angiography",
                  "coronary_stenosis", "Possible coronary artery stenosis", floor_ok=True)

def test_confident_detection_becomes_kept_finding():
    triage = {"prediction": [(0, 0, 10, 10, 0.9)], "calibrated_confs": [0.9],
              "deferred": False, "reason": "confident"}
    fs = det_to_findings(ENTRY, triage)
    assert len(fs) == 1
    assert fs[0].label == "coronary_stenosis" and fs[0].deferred is False
    assert fs[0].boxes == [(0, 0, 10, 10, 0.9)] and fs[0].confidence == 0.9

def test_below_floor_entry_forces_defer_even_if_triage_confident():
    below = TaskEntry("coronary_angiography", "det", "best.pt", "Coronary angiography",
                      "coronary_stenosis", "Possible coronary artery stenosis", floor_ok=False)
    triage = {"prediction": [(0, 0, 10, 10, 0.95)], "calibrated_confs": [0.95],
              "deferred": False, "reason": "confident"}
    fs = det_to_findings(below, triage)
    assert fs[0].deferred is True and fs[0].reason == "below-floor"

def test_clean_negative_is_a_nonpositive_kept_finding():
    triage = {"prediction": [], "calibrated_confs": [], "deferred": False, "reason": "clean"}
    fs = det_to_findings(ENTRY, triage)
    assert fs[0].confidence == 0.0 and fs[0].deferred is False and fs[0].reason == "clean"

def test_study_defer_true_when_any_finding_deferred():
    dec = ModalityDecision("coronary_angiography", None, True, 0.9, False, "confident")
    triage = {"prediction": [(0, 0, 5, 5, 0.5)], "calibrated_confs": [0.5],
              "deferred": True, "reason": "low-confidence"}
    fs = det_to_findings(ENTRY, triage)
    deferred, reason = study_defer(dec, fs)
    assert deferred is True and reason == "low-confidence"

def test_study_defer_true_when_modality_unknown():
    dec = ModalityDecision("unknown", None, True, 0.4, True, "router-uncertain")
    deferred, reason = study_defer(dec, [])
    assert deferred is True and reason == "router-uncertain"


# --- Finding 1: study_defer must not fail open on an unscreened study --------------------------
# A confidently-resolved KNOWN modality with an empty findings list is reachable whenever
# registry.resolve(modality) returns None (no TaskEntry configured yet) -- the orchestrator never
# calls det_to_findings/seg_to_finding, so findings=[] arrives here alongside decision.deferred=False.
# That must defer, not silently clear the study.

def test_study_defer_true_when_findings_empty_but_modality_confidently_known():
    dec = ModalityDecision("coronary_angiography", None, True, 0.9, False, "confident")
    deferred, reason = study_defer(dec, [])
    assert deferred is True
    assert reason == "no-findings"


# --- Finding 2: seg_to_finding coverage, including the floor_ok-forces-defer behavior -----------

SEG_ENTRY = TaskEntry("carotid_us", "seg", "seg.pt", "Carotid ultrasound",
                      "carotid_plaque", "Possible carotid plaque", floor_ok=True)

def test_seg_confident_result_becomes_nondeferred_finding():
    seg_res = {"confidence": 0.87, "deferred": False}
    f = seg_to_finding(SEG_ENTRY, seg_res)
    assert f.label == "carotid_plaque" and f.confidence == 0.87
    assert f.deferred is False and f.reason == "confident"

def test_seg_low_confidence_result_defers():
    seg_res = {"confidence": 0.4, "deferred": True}
    f = seg_to_finding(SEG_ENTRY, seg_res)
    assert f.deferred is True and f.reason == "low-confidence"

def test_seg_below_floor_entry_forces_defer_even_if_seg_res_confident():
    below = TaskEntry("carotid_us", "seg", "seg.pt", "Carotid ultrasound",
                      "carotid_plaque", "Possible carotid plaque", floor_ok=False)
    seg_res = {"confidence": 0.95, "deferred": False}
    f = seg_to_finding(below, seg_res)
    assert f.deferred is True and f.reason == "below-floor"


# --- Finding 3: malformed/incomplete input must fail safe (defer), not fail open -----------------

def test_det_to_findings_malformed_empty_triage_defers():
    fs = det_to_findings(ENTRY, {})
    assert fs[0].deferred is True
    assert fs[0].reason != "clean"
    assert fs[0].confidence == 0.0 and fs[0].boxes == []

def test_det_to_findings_missing_deferred_key_defers():
    triage = {"prediction": [], "calibrated_confs": [], "reason": "clean"}
    fs = det_to_findings(ENTRY, triage)
    assert fs[0].deferred is True

def test_det_to_findings_missing_reason_key_defers():
    triage = {"prediction": [], "calibrated_confs": [], "deferred": False}
    fs = det_to_findings(ENTRY, triage)
    assert fs[0].deferred is True

def test_seg_to_finding_malformed_empty_result_defers():
    f = seg_to_finding(SEG_ENTRY, {})
    assert f.deferred is True
    assert f.reason != "confident"
    assert f.confidence == 0.0

def test_seg_to_finding_missing_confidence_key_defers():
    seg_res = {"deferred": False}
    f = seg_to_finding(SEG_ENTRY, seg_res)
    assert f.deferred is True

def test_seg_to_finding_missing_deferred_key_defers():
    seg_res = {"confidence": 0.9}
    f = seg_to_finding(SEG_ENTRY, seg_res)
    assert f.deferred is True
