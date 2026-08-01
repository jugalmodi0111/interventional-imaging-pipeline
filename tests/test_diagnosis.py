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
