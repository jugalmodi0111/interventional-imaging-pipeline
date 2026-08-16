"""Validity gate (Dialygo B3) — the input-plausibility check that replaces the modality router.

B3: "reject any input that is not a valid vascular-access angiogram (wrong modality, corrupt or
unrelated image) rather than attempting to read it." Safety default is REJECT: anything the gate
cannot positively vouch for defers, and the gate itself must never raise into the request path.
"""
import subprocess
import sys

import numpy as np
import pytest

from src.serve.validity import ModalityDecision, ValidityGate, assess_frame


def _angiogram(side=512, seed=0):
    """A plausible grayscale angiographic frame: wide dynamic range, few clipped pixels."""
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 210, size=(side, side), dtype=np.uint8)
    base[side // 3: side // 3 + 6, :] = 20          # a dark vessel-like structure
    return base


def _gate(**kw):
    return ValidityGate("coronary_angiography", **kw)


# --- assess_frame: the pure predicate ------------------------------------------------------------

def test_plausible_grayscale_frame_is_accepted():
    ok, score, reason = assess_frame(_angiogram())
    assert ok is True and reason == "valid"
    assert 0.0 <= score <= 1.0


def test_colour_frame_is_rejected():
    # An angiographic frame is single-channel. A 3-channel image is a different kind of thing.
    ok, _, reason = assess_frame(np.dstack([_angiogram()] * 3))
    assert ok is False and reason == "not-grayscale"


def test_undersized_frame_is_rejected():
    ok, _, reason = assess_frame(_angiogram(side=64))
    assert ok is False and reason == "too-small"


def test_blank_frame_is_rejected():
    ok, _, reason = assess_frame(np.full((512, 512), 128, dtype=np.uint8))
    assert ok is False and reason == "degenerate-contrast"


def test_mostly_clipped_frame_is_rejected():
    # Blown-out or crushed acquisition: almost every pixel is pinned at an endpoint.
    f = _angiogram()
    f[:, : int(f.shape[1] * 0.95)] = 255
    ok, _, reason = assess_frame(f)
    assert ok is False and reason == "saturated"


def test_none_is_rejected_rather_than_crashing():
    ok, _, reason = assess_frame(None)
    assert ok is False and reason == "no-frame"


def test_garbage_input_is_rejected_rather_than_raising():
    # Fail closed: the gate sits in the request path, so a surprising type must defer, not raise.
    for junk in ("not an image", 42, {"frame": 1}, np.zeros((4, 4, 4, 4))):
        ok, _, reason = assess_frame(junk)
        assert ok is False, f"{junk!r} was accepted"
        assert reason, "a rejection must carry a reason"


# --- ValidityGate.classify: the router-shaped adapter --------------------------------------------

def test_gate_accepts_a_valid_frame_as_the_configured_modality():
    dec = _gate().classify(_angiogram())
    assert isinstance(dec, ModalityDecision)
    assert dec.modality == "coronary_angiography"
    assert dec.deferred is False and dec.quality_ok is True and dec.reason == "valid"


def test_gate_defers_an_invalid_frame_to_unknown_modality():
    # Critical: a rejected frame must NOT resolve to a real modality, or the orchestrator would
    # happily run a disease model on an unrelated image.
    dec = _gate().classify(np.full((512, 512), 7, dtype=np.uint8))
    assert dec.modality == "unknown"
    assert dec.deferred is True and dec.quality_ok is False
    assert dec.reason == "degenerate-contrast"


def test_gate_defers_in_the_uncertain_band():
    # Contrast sits between the reject floor and the accept threshold: the honest answer is
    # "uncertain", which must defer rather than resolve either way.
    f = np.full((512, 512), 120, dtype=np.uint8)
    f[::2] = 145                                     # narrow but nonzero dynamic range
    dec = _gate().classify(f)
    assert dec.deferred is True
    assert dec.modality == "unknown" and dec.reason == "validity-uncertain"


def test_gate_never_raises_on_garbage():
    dec = _gate().classify("not an image")
    assert dec.deferred is True and dec.modality == "unknown"


def test_gate_exposes_a_version_string_for_the_audit_trail():
    assert isinstance(_gate().weights, str) and _gate().weights


def test_validity_module_imports_without_torch():
    # Repo invariant, mirrors test_router.py: the decision layer must import on a machine that has
    # no torch. A fresh interpreter, so an earlier test file's torch import can't mask a regression.
    code = ("import sys; sys.modules['torch'] = None; "
            "import src.serve.validity as v; "
            "assert v.assess_frame(None)[0] is False; print('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout


@pytest.mark.parametrize("thr,expect_deferred", [(0.10, False), (0.50, True)])
def test_accept_threshold_is_a_configurable_knob(thr, expect_deferred):
    # The accept threshold is a deployment knob, not a constant baked into the request path.
    # This marginal frame scores ~0.125: raising the bar past it must flip accept -> defer.
    f = np.full((512, 512), 120, dtype=np.uint8)
    f[::2] = 145
    assert 0.10 < assess_frame(f)[1] < 0.50, "fixture must sit between the two thresholds"
    dec = _gate(accept_score=thr).classify(f)
    assert dec.deferred is expect_deferred
