"""Test the modality router decision layer."""
import pytest
from src.serve.router import decide_modality, ModalityDecision, ModalityRouter


def test_confident_top_class_is_kept():
    d = decide_modality({"coronary_angiography": 0.9, "cerebral_dsa": 0.05, "other_xray": 0.05})
    assert isinstance(d, ModalityDecision)
    assert d.modality == "coronary_angiography"
    assert d.deferred is False
    assert d.reason == "confident"


def test_below_keep_threshold_defers_unknown():
    d = decide_modality({"coronary_angiography": 0.5, "cerebral_dsa": 0.3, "other_xray": 0.2})
    assert d.modality == "unknown"
    assert d.deferred is True
    assert d.reason == "router-uncertain"


def test_thin_margin_between_top_two_defers():
    # top prob clears keep_thr but the runner-up is within `margin` -> ambiguous -> defer
    d = decide_modality({"coronary_angiography": 0.62, "cerebral_dsa": 0.55, "other_xray": 0.0},
                        keep_thr=0.60, margin=0.15)
    assert d.modality == "unknown"
    assert d.deferred is True
    assert d.reason == "router-uncertain"


def test_low_quality_flag_defers_even_if_confident_class():
    d = decide_modality({"coronary_angiography": 0.95, "other_xray": 0.05},
                        quality_prob=0.2, quality_thr=0.5)
    assert d.quality_ok is False
    assert d.deferred is True
    assert d.reason == "low-quality"


def test_reject_bucket_class_is_returned_not_unknown():
    # a confident non-medical image is a real, keepable classification (-> orchestrator will defer as unsupported)
    d = decide_modality({"non_medical": 0.97, "coronary_angiography": 0.03})
    assert d.modality == "non_medical"
    assert d.deferred is False
    assert d.reason == "confident"


def test_boundary_top_prob_exactly_at_keep_threshold_is_accepted():
    # At-threshold is accepted, not deferred: top_p == keep_thr (not <)
    d = decide_modality({"coronary_angiography": 0.60, "cerebral_dsa": 0.40},
                        keep_thr=0.60, margin=0.15)
    assert d.modality == "coronary_angiography"
    assert d.deferred is False
    assert d.reason == "confident"


def test_boundary_margin_exactly_at_threshold_is_accepted():
    # At-threshold is accepted, not deferred: (top_p - runner_p) == margin (not <)
    d = decide_modality({"coronary_angiography": 0.75, "cerebral_dsa": 0.60},
                        keep_thr=0.60, margin=0.15)
    assert d.modality == "coronary_angiography"
    assert d.deferred is False
    assert d.reason == "confident"


def test_boundary_quality_exactly_at_threshold_is_accepted():
    # At-threshold is accepted, not deferred: quality_prob == quality_thr (uses >=)
    d = decide_modality({"coronary_angiography": 0.9, "cerebral_dsa": 0.1},
                        quality_prob=0.5, quality_thr=0.5)
    assert d.quality_ok is True
    assert d.deferred is False
    assert d.reason == "confident"


# ---- ModalityRouter (edge classifier wrapper, torch-free tests) -------------

def test_router_classify_uses_decide_modality(monkeypatch):
    # No model load: bypass __init__ and inject a fake _probs so this stays torch-free.
    r = ModalityRouter.__new__(ModalityRouter)          # bypass __init__ (no model load)
    r.labels = ["coronary_angiography", "other_xray"]
    r.thresholds = {"keep_thr": 0.60, "margin": 0.15, "quality_thr": 0.5}
    r.size = 224
    monkeypatch.setattr(r, "_probs", lambda frame: {"coronary_angiography": 0.92, "other_xray": 0.08})
    d = r.classify(frame=object())
    assert d.modality == "coronary_angiography" and d.deferred is False


def test_router_classify_defers_on_thin_margin(monkeypatch):
    # Wrapper must route through decide_modality's judgment, not just echo the top class.
    r = ModalityRouter.__new__(ModalityRouter)
    r.labels = ["coronary_angiography", "cerebral_dsa"]
    r.thresholds = {"keep_thr": 0.60, "margin": 0.15, "quality_thr": 0.5}
    r.size = 224
    monkeypatch.setattr(r, "_probs", lambda frame: {"coronary_angiography": 0.62, "cerebral_dsa": 0.55})
    d = r.classify(frame=object())
    assert d.modality == "unknown" and d.deferred is True and d.reason == "router-uncertain"


def test_router_init_sets_default_thresholds_without_loading_model():
    # __init__ must not eagerly load the model (no weights file needed here) and must default
    # thresholds to match decide_modality's own defaults.
    r = ModalityRouter(weights="unused.pt", labels=["coronary_angiography", "other_xray"])
    assert r._model is None
    assert r.thresholds == {"keep_thr": 0.60, "margin": 0.15, "quality_thr": 0.5}
    assert r.size == 224


def test_router_module_imports_without_torch():
    # Guardrail: importing src.serve.router must not drag in torch/timm/cv2 — those belong only
    # inside ModalityRouter._load/_probs. Run in a FRESH interpreter (subprocess) so torch already
    # loaded into sys.modules by an EARLIER test file can't defeat the check — the property under
    # test is router's OWN import, not global state.
    import os, subprocess, sys, textwrap
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = textwrap.dedent("""
        import sys, importlib
        importlib.import_module("src.serve.router")
        for mod in ("torch", "timm", "cv2", "coremltools", "ultralytics", "transformers"):
            assert mod not in sys.modules, f"router import pulled in {mod}"
    """)
    r = subprocess.run([sys.executable, "-c", code], cwd=repo_root,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
