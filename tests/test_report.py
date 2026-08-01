import json
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
