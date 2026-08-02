"""Test the POST /analyze endpoint: HTTP behavior, image-vs-video routing, and failure modes.

No real model/weights load here: the orchestrator singleton (`app_mod._orch`) is monkeypatched to a
fake, matching the torch-free-test convention used throughout `tests/test_orchestrator.py` and
`tests/test_router.py`. This file tests the endpoint's own plumbing (routing, error handling, JSON
shape) -- never model accuracy.
"""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import src.serve.app as app_mod
from src.serve.report import StudyReport, Finding


class FakeOrch:
    """Records what it was called with so tests can assert routing without touching real models."""

    def __init__(self):
        self.frame_calls = []
        self.video_calls = []

    def analyze_frame(self, frame):
        self.frame_calls.append(frame)
        return StudyReport("coronary_angiography", None, True,
                           [Finding("coronary_stenosis", "Possible finding — clinician review required",
                                    0.0, True, "below-floor")],
                           True, "below-floor", 1, {"router": "r"})

    def analyze_video(self, path, **kw):
        self.video_calls.append(path)
        return StudyReport("coronary_angiography", None, True, [], True, "router-uncertain", 0,
                           {"router": "r"})


class RaisingOrch:
    """Used to prove a branch (e.g. undecodable image) short-circuits BEFORE the model ever runs."""

    def analyze_frame(self, frame):
        raise AssertionError("model must not run on an undecodable frame")

    def analyze_video(self, path, **kw):
        raise AssertionError("model must not run on an unusable video path")


class BuggyOrch:
    """Simulates an unexpected bug inside the orchestrator itself (not a defer, a real crash) --
    the endpoint must still never bubble this up as a raw 500."""

    def analyze_frame(self, frame):
        raise RuntimeError("boom")

    def analyze_video(self, path, **kw):
        raise RuntimeError("boom")


def test_analyze_image_returns_report(monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod, "_orch", FakeOrch())
    monkeypatch.setattr(app_mod, "_decode_image", lambda raw: object())
    c = TestClient(app_mod.app)
    resp = c.post("/analyze?kind=image", files={"file": ("f.png", b"x", "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["deferred"] is True and body["defer_reason"] == "below-floor"
    assert body["findings"][0]["label"] == "coronary_stenosis"


def test_analyze_defaults_to_image_kind(monkeypatch):
    # kind is optional; default must be "image" (not video).
    fake = FakeOrch()
    monkeypatch.setattr(app_mod, "_orch", fake)
    monkeypatch.setattr(app_mod, "_decode_image", lambda raw: object())
    c = TestClient(app_mod.app)
    resp = c.post("/analyze", files={"file": ("f.png", b"x", "image/png")})
    assert resp.status_code == 200
    assert len(fake.frame_calls) == 1 and fake.video_calls == []


def test_analyze_video_routes_to_analyze_video_and_cleans_up_tempfile(monkeypatch):
    fake = FakeOrch()
    monkeypatch.setattr(app_mod, "_orch", fake)
    c = TestClient(app_mod.app)
    resp = c.post("/analyze?kind=video", files={"file": ("clip.mp4", b"not-a-real-video", "video/mp4")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["deferred"] is True and body["defer_reason"] == "router-uncertain"
    assert len(fake.video_calls) == 1
    # the temp file handed to analyze_video must not survive the request
    import os
    assert not os.path.exists(fake.video_calls[0])


def test_analyze_unsupported_kind_is_a_clean_400_not_a_500(monkeypatch):
    monkeypatch.setattr(app_mod, "_orch", FakeOrch())
    c = TestClient(app_mod.app)
    resp = c.post("/analyze?kind=pdf", files={"file": ("f.pdf", b"x", "application/pdf")})
    assert resp.status_code == 400


def test_analyze_corrupt_image_defers_without_running_model(monkeypatch):
    # cv2.imdecode returns None (not an exception) on undecodable bytes -- the endpoint must catch
    # that itself and defer, never pass None into the orchestrator as if it were a real frame.
    monkeypatch.setattr(app_mod, "_orch", RaisingOrch())
    monkeypatch.setattr(app_mod, "_decode_image", lambda raw: None)
    c = TestClient(app_mod.app)
    resp = c.post("/analyze?kind=image", files={"file": ("f.png", b"garbage", "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["deferred"] is True
    assert "diagnosis:" not in str(body).lower()


def test_analyze_real_corrupt_bytes_defer_end_to_end(monkeypatch):
    # No monkeypatch of _decode_image: real cv2.imdecode on garbage bytes returns None. Proves the
    # real decode path (not just a faked one) also defers cleanly.
    monkeypatch.setattr(app_mod, "_orch", RaisingOrch())
    c = TestClient(app_mod.app)
    resp = c.post("/analyze?kind=image", files={"file": ("f.png", b"totally not an image", "image/png")})
    assert resp.status_code == 200
    assert resp.json()["deferred"] is True


def test_analyze_model_unavailable_at_orchestrator_build_defers_not_500(monkeypatch):
    # Simulate a broken/missing registry config: _orch is unset and building one fails. Must never
    # surface as an unhandled 500 -- a clean deferred report instead.
    monkeypatch.setattr(app_mod, "_orch", None)

    def _boom(*a, **kw):
        raise FileNotFoundError("configs/orchestrator.yaml not found")

    monkeypatch.setattr(app_mod, "_get_orch", _boom)
    c = TestClient(app_mod.app)
    resp = c.post("/analyze?kind=image", files={"file": ("f.png", b"x", "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["deferred"] is True and body["defer_reason"] == "model-unavailable"


def test_analyze_unexpected_orchestrator_exception_defers_not_500(monkeypatch):
    monkeypatch.setattr(app_mod, "_orch", BuggyOrch())
    monkeypatch.setattr(app_mod, "_decode_image", lambda raw: object())
    c = TestClient(app_mod.app)
    resp = c.post("/analyze?kind=image", files={"file": ("f.png", b"x", "image/png")})
    assert resp.status_code == 200
    assert resp.json()["deferred"] is True


def test_analyze_output_never_claims_autonomous_diagnosis(monkeypatch):
    # Output copy must read as a screening flag needing clinician review, never a bare "diagnosis: X".
    monkeypatch.setattr(app_mod, "_orch", FakeOrch())
    monkeypatch.setattr(app_mod, "_decode_image", lambda raw: object())
    c = TestClient(app_mod.app)
    resp = c.post("/analyze?kind=image", files={"file": ("f.png", b"x", "image/png")})
    body = resp.json()
    display = body["findings"][0]["display_name"]
    assert "diagnosis:" not in display.lower()
    assert "clinician review required" in display.lower()


def test_infer_route_still_works(monkeypatch):
    # Backward compatibility: the pre-existing /infer route must keep working unmodified.
    def _fake_model(frame):
        return {"deferred": False, "confidence": 0.9, "mask": __import__("numpy").zeros((2, 2))}

    monkeypatch.setattr(app_mod, "_get_model", lambda: _fake_model)
    c = TestClient(app_mod.app)
    resp = c.post("/infer", files={"file": ("f.png", b"x", "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["deferred"] is False and body["vessel_pixels"] == 0
