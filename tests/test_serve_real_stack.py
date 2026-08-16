"""End-to-end proof that the REAL serve stack answers — no fakes anywhere in this file.

Every other serve test injects a fake gate and a fake model, which is right for pinning the
decision logic but means the suite stayed green for weeks while `/analyze` could not actually
serve anything (no router weights ever existed). This file closes that gap: real
`configs/orchestrator.yaml`, real `ValidityGate`, real CoreML `SegModel`, real val frames.

Skipped rather than failed when coremltools or the exported artifact is absent — the artifact is
gitignored (local/release only), so a checkout without it must not report a red suite.
"""
import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("coremltools")

CFG = "configs/orchestrator.yaml"
MLPACKAGE = "outputs/coronary_student_clgeodice/student.mlpackage"
VAL_IMG = "data/processed/coronary/val/img"

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(MLPACKAGE) and os.path.isdir(VAL_IMG)),
    reason=f"needs the exported artifact ({MLPACKAGE}) and val frames ({VAL_IMG})")


@pytest.fixture(scope="module")
def orch():
    from src.serve.orchestrator import build_orchestrator
    return build_orchestrator(CFG)


@pytest.fixture(scope="module")
def frame():
    paths = sorted(p for p in os.listdir(VAL_IMG) if p.endswith(".png"))
    assert paths, f"no val frames under {VAL_IMG}"
    return cv2.imread(os.path.join(VAL_IMG, paths[0]), cv2.IMREAD_GRAYSCALE)


def test_real_angiogram_yields_a_confident_undeferred_verdict(orch, frame):
    # The claim this file exists to prove: a real frame through the real stack produces an actual
    # answer, not a defer. Before the B3 validity gate landed, every real call returned
    # deferred=True/"router-unavailable" because the modality router had no weights.
    rep = orch.analyze_frame(frame)
    assert rep.modality == "coronary_angiography"
    assert rep.deferred is False, f"still deferring: {rep.defer_reason!r}"
    assert len(rep.findings) == 1
    f = rep.findings[0]
    assert f.deferred is False and f.reason == "confident"
    assert 0.0 <= f.confidence <= 1.0


def test_report_serializes_to_json_safe_types(orch, frame):
    # Regression pin for audit P3.3: a numpy float32 in the payload used to 500 the endpoint.
    import json
    json.dumps(orch.analyze_frame(frame).to_dict())


@pytest.mark.parametrize("bad,reason", [
    (np.full((512, 512), 128, np.uint8), "degenerate-contrast"),
    (np.zeros((512, 512), np.uint8), "degenerate-contrast"),
    (np.full((512, 512, 3), 90, np.uint8), "not-grayscale"),
    (np.full((64, 64), 90, np.uint8), "too-small"),
])
def test_gate_refuses_bad_input_before_any_model_runs(orch, bad, reason):
    rep = orch.analyze_frame(bad)
    assert rep.modality == "unknown", "a rejected frame must never resolve to a real modality"
    assert rep.deferred is True and rep.defer_reason == reason
    assert rep.findings == []


def test_http_analyze_returns_the_real_verdict(frame):
    # Same claim, through the actual endpoint, so the FastAPI layer is proven too.
    TestClient = pytest.importorskip("fastapi.testclient").TestClient
    from src.serve import app as app_mod

    png = cv2.imencode(".png", frame)[1].tobytes()
    r = TestClient(app_mod.app).post("/analyze?kind=image",
                                     files={"file": ("f.png", png, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["deferred"] is False and body["modality"] == "coronary_angiography"
    assert body["model_versions"]["gate"].startswith("validity-gate/")
