import json

import numpy as np
import pytest

from src.serve.report import Finding, StudyReport


def test_finding_defaults():
    f = Finding(label="coronary_stenosis", display_name="Possible coronary artery stenosis",
                confidence=0.8, deferred=False, reason="confident")
    assert f.severity is None and f.boxes == []


def test_studyreport_to_dict_is_json_safe():
    f = Finding("coronary_stenosis", "Possible coronary artery stenosis", 0.8, False, "confident",
                boxes=[(1, 2, 3, 4, 0.8)])
    r = StudyReport(modality="coronary_angiography", view=None, quality_ok=True, findings=[f],
                    deferred=False, defer_reason="", frames_analyzed=1,
                    model_versions={"coronary_stenosis": "best.pt"})
    d = r.to_dict()
    s = json.dumps(d)                    # must not raise (tuples -> lists)
    assert json.loads(s)["findings"][0]["boxes"][0] == [1, 2, 3, 4, 0.8]
    assert d["deferred"] is False


def test_studyreport_to_dict_casts_numpy_scalars_to_builtin_json_types():
    # infer.py's CoreML box parser emits numpy.float32 coordinates/confidences; json.dumps raises
    # TypeError on any numpy scalar, which turned EVERY positive finding into a 500 at the /analyze
    # endpoint (2026-08-03 audit, P3 critical 3). to_dict must yield built-in Python types for every
    # box coordinate and confidence -- and for any numpy scalar anywhere else in the payload.
    f = Finding("coronary_stenosis", "Possible coronary artery stenosis",
                confidence=np.float32(0.83), deferred=np.bool_(False), reason="confident",
                boxes=[(np.float32(12.5), np.float32(40.0), np.float32(99.9),
                        np.float32(150.1), np.float32(0.83))])
    r = StudyReport(modality="coronary_angiography", view=None, quality_ok=True, findings=[f],
                    deferred=False, defer_reason="", frames_analyzed=np.int64(1),
                    model_versions={"coronary_stenosis": "best.pt"})
    d = r.to_dict()
    s = json.dumps(d)                                   # the regression: must not raise TypeError
    got = json.loads(s)["findings"][0]
    assert got["boxes"][0] == [pytest.approx(12.5), pytest.approx(40.0), pytest.approx(99.9),
                               pytest.approx(150.1), pytest.approx(0.83)]
    assert all(type(v) is float for v in d["findings"][0]["boxes"][0])
    assert type(d["findings"][0]["confidence"]) is float
    assert type(d["findings"][0]["deferred"]) is bool
    assert type(d["frames_analyzed"]) is int
