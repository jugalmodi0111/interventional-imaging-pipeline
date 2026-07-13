"""Sequence-level stenosis inference wiring (src/serve/stenosis_infer.py).

Torch-/ultralytics-/GPU-free: :func:`predict_sequence` takes an injectable ``detect_fn`` so the
detector -> temporal-voting -> triage composition is pinned on a SYNTHETIC cine window. Each test
locks one shipped property -- gap recovery (recall), flicker rejection (precision), safe abstention
-- so a regression that loses a lesion frame, resurrects a flicker, or silently keeps a boundary
case re-surfaces loudly. The predict_image single-frame merge is exercised with a fake detector.
"""
import os

import pytest

from src.serve.stenosis_infer import predict_sequence


# --- synthetic 5-frame cine window (no model) ----------------------------------------------------
# Persistent lesion A (high conf) observed on frames 0, 1, 3 -> 3/5 frames, with the detector MISSING
# frame 2 (a one-frame flicker fires there instead). Boundary-confidence lesion B on frames 3 & 4.

def _A(cx, conf=0.85):                                  # high-confidence persistent lesion
    return {"box": (cx, 0.30, 0.20, 0.20), "conf": conf}


def _B(cx, conf=0.55):                                  # boundary-confidence persistent lesion
    return {"box": (cx, 0.70, 0.20, 0.20), "conf": conf}


def _flicker():                                         # one-frame false positive, far from A and B
    return {"box": (0.10, 0.90, 0.10, 0.10), "conf": 0.40}


def _synthetic_sequence(detector, frames, **_):
    """Injectable fake ``detect_fn``: ignores the (absent) model and returns fixed per-frame dets."""
    return [
        [_A(0.50)],                     # f0  A observed
        [_A(0.52)],                     # f1  A observed
        [_flicker()],                   # f2  A MISSED (detector dropout); a flicker fires instead
        [_A(0.54), _B(0.80)],           # f3  A observed + boundary lesion B appears
        [_B(0.82)],                     # f4  boundary lesion B only
    ]


def _run(**kw):
    return predict_sequence("<no-model>", ["f0", "f1", "f2", "f3", "f4"],
                            detect_fn=_synthetic_sequence, **kw)


def _boxes(result):                                     # flatten surfaced detections across frames
    return [d for frame in result for d in frame["detections"]]


# --- recall: temporal voting recovers the frame the detector missed ------------------------------

def test_persistent_lesion_survives_on_every_spanned_frame():
    res = _run()
    # A's track spans frames 0..3; each carries the surfaced lesion (frame 2 recovered by voting).
    assert all(len(res[fi]["detections"]) >= 1 for fi in (0, 1, 2, 3))


def test_missed_frame_is_recovered_by_interpolation():
    res = _run()
    # Frame 2 (raw = flicker only) comes back holding the INTERPOLATED lesion A, not the flicker.
    recovered = res[2]["detections"]
    assert len(recovered) == 1
    assert recovered[0]["interpolated"] is True
    assert recovered[0]["box"][0] == pytest.approx(0.53)          # halfway between f1 0.52 and f3 0.54


def test_gap_recovery_needs_temporal_voting():
    # Without voting the detector's frame-2 dropout stays a miss: nothing is interpolated and no
    # lesion is recovered at cx=0.53 (triage off so we read the raw stabilized stream directly).
    raw = _run(temporal=False, triage=False)
    assert all(not d.get("interpolated", False) for d in _boxes(raw))
    assert all(abs(d["box"][0] - 0.53) > 1e-3 for d in raw[2]["detections"])


# --- precision: the one-frame flicker is dropped -------------------------------------------------

def test_single_frame_flicker_is_dropped():
    res = _run()
    # The flicker lived near cx=0.10; min_hits=2 must remove it from every frame's output.
    assert all(abs(d["box"][0] - 0.10) > 1e-6 for d in _boxes(res))


def test_flicker_survives_without_temporal_voting():
    # Sanity that the flicker is real in the raw stream and it is VOTING (not triage) that kills it:
    # with both off it is present; enabling voting removes it (see test above).
    raw = _run(temporal=False, triage=False)
    assert any(abs(d["box"][0] - 0.10) < 1e-6 for d in _boxes(raw))


# --- safe abstention: the boundary-confidence frame defers to a human ----------------------------

def test_boundary_confidence_frame_is_deferred():
    res = _run()
    # Frame 4 holds only lesion B (calibrated conf 0.55, inside defer_band) -> route to a human.
    assert res[4]["deferred"] is True
    assert res[4]["reason"] == "low-confidence"
    assert len(res[4]["detections"]) == 1                          # still surfaced, never hidden


def test_confident_frames_are_not_deferred():
    res = _run()
    for fi in (0, 1, 2, 3):                                        # A (0.85) dominates -> confident
        assert res[fi]["deferred"] is False
        assert res[fi]["reason"] == "confident"


def test_temperature_can_push_confident_lesion_into_defer():
    # Cooling the 0.85 lesion enough drags its calibrated conf into the band -> the kept frames defer.
    res = _run(temperature=6.0)
    assert res[0]["deferred"] is True and res[0]["reason"] == "low-confidence"


# --- composition contract ------------------------------------------------------------------------

def test_result_shape_and_keys_match_frames():
    res = _run()
    assert len(res) == 5
    assert all(set(r) == {"detections", "deferred", "reason"} for r in res)


def test_triage_disabled_surfaces_stabilized_dets_without_defer():
    res = _run(triage=False)
    assert all(r["deferred"] is False and r["reason"] == "no-triage" for r in res)
    # Voting still runs: flicker gone, gap recovered even though triage is off.
    assert all(abs(d["box"][0] - 0.10) > 1e-6 for d in _boxes(res))
    assert res[4]["detections"] and res[4]["detections"][0]["box"][0] == pytest.approx(0.82)


def test_detect_fn_receives_forwarded_runtime_kwargs():
    seen = {}

    def spy(detector, frames, **kw):
        seen.update(detector=detector, frames=frames, **kw)
        return [[] for _ in frames]

    predict_sequence("weights.pt", ["a", "b"], detect_fn=spy,
                     conf=0.002, imgsz=512, device="cpu")
    assert seen["detector"] == "weights.pt" and seen["frames"] == ["a", "b"]
    assert seen["conf"] == 0.002 and seen["imgsz"] == 512 and seen["device"] == "cpu"


# --- predict_image single-frame triage merge (fake detector, no ultralytics) ---------------------

def _install_fake_yolo(monkeypatch, boxes):
    """Inject a fake ``ultralytics.YOLO`` whose predict() returns ``boxes`` as (x1,y1,x2,y2,conf)."""
    import sys, types
    import numpy as np

    class _FakeBox:
        def __init__(self, x1, y1, x2, y2, conf):
            self.xyxy = np.array([[x1, y1, x2, y2]], dtype=float)
            self.conf = np.array([conf], dtype=float)

    class _FakeResult:
        def __init__(self, boxes):
            self.boxes = [_FakeBox(*b) for b in boxes]

    class _FakeYOLO:
        def __init__(self, weights):
            pass

        def predict(self, img, conf=0.25, imgsz=640, verbose=False):
            return [_FakeResult(boxes)]

    fake = types.ModuleType("ultralytics")
    fake.YOLO = _FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake)


def _write_frame(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    path = tmp_path / "frame.png"
    img = np.random.default_rng(0).integers(0, 255, (48, 48), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return str(path)


def test_predict_image_defers_on_boundary_confidence(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    _install_fake_yolo(monkeypatch, [(10, 10, 20, 20, 0.55)])       # inside defer_band
    from src.serve import predict_image
    img = _write_frame(tmp_path)
    out = str(tmp_path / "anno.png")
    r = predict_image.predict(img, "fake.pt", out=out, defer=True)
    assert r["n_blockages"] == 1
    assert r["deferred"] is True and r["reason"] == "low-confidence"
    assert os.path.exists(r["out"])


def test_predict_image_confident_box_is_not_deferred_and_keeps_old_keys(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    _install_fake_yolo(monkeypatch, [(10, 10, 20, 20, 0.92)])       # clearly above the band
    from src.serve import predict_image
    img = _write_frame(tmp_path)
    r = predict_image.predict(img, "fake.pt", out=str(tmp_path / "a.png"))
    assert r["deferred"] is False and r["reason"] == "confident"
    assert {"out", "n_blockages", "vis"} <= set(r)                  # backward-compatible return
    assert r["n_blockages"] == 1


def test_predict_image_no_detection_is_clean(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    _install_fake_yolo(monkeypatch, [])                             # detector fires nothing
    from src.serve import predict_image
    img = _write_frame(tmp_path)
    r = predict_image.predict(img, "fake.pt", out=str(tmp_path / "a.png"))
    assert r["n_blockages"] == 0
    assert r["deferred"] is False and r["reason"] == "clean"
