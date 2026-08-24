"""cls output -> Finding. Same fail-safe grammar as det/seg: malformed input and below-floor both
defer -- a classifier that didn't demonstrably run + clear its floor never emits a confident call."""
from src.serve.diagnosis import cls_to_finding
from src.serve.registry import TaskEntry


def _entry(floor_ok=True):
    return TaskEntry("avf_fistulography", "cls", "head.pt", "AVF fistulography",
                     "avf_ja_stenosis", "Possible juxta-anastomotic stenosis", floor_ok=floor_ok)


def test_confident_positive_becomes_kept_finding():
    f = cls_to_finding(_entry(), {"prob": 0.9, "confidence": 0.9, "deferred": False,
                                  "reason": "confident", "threshold": 0.5})
    assert not f.deferred and f.confidence == 0.9 and f.label == "avf_ja_stenosis"


def test_defer_band_result_stays_deferred():
    f = cls_to_finding(_entry(), {"prob": 0.45, "confidence": 0.55, "deferred": True,
                                  "reason": "defer-band", "threshold": 0.5})
    assert f.deferred and f.reason == "defer-band"


def test_below_floor_forces_defer_even_when_confident():
    f = cls_to_finding(_entry(floor_ok=False), {"prob": 0.95, "confidence": 0.95,
                                                "deferred": False, "reason": "confident",
                                                "threshold": 0.5})
    assert f.deferred and f.reason == "below-floor"


def test_malformed_output_defers():
    f = cls_to_finding(_entry(), {"confidence": 0.9})       # no prob, no deferred
    assert f.deferred and f.reason == "malformed-cls"


def test_confident_negative_is_still_a_finding_not_an_empty_list():
    """A confident NEGATIVE must produce a Finding. An empty findings list means 'nothing screened
    this study' and study_defer turns it into a defer -- returning [] for a clean read would be
    indistinguishable from the model never running (diagnosis.py module docstring)."""
    f = cls_to_finding(_entry(), {"prob": 0.02, "confidence": 0.98, "deferred": False,
                                  "reason": "confident", "threshold": 0.5})
    assert f is not None and not f.deferred and f.boxes == []
