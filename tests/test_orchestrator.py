"""Test the diagnostic orchestrator: route -> resolve -> infer -> typed findings -> StudyReport.

Gate and model are both fakes/injected -- no torch import at module scope anywhere in this file,
matching the import-safe convention the orchestrator itself follows. (One test runs the REAL
`ValidityGate`, which is numpy-only and needs no weights.)
"""
import pytest

from src.serve.registry import TaskEntry
from src.serve.validity import ModalityDecision, ValidityGate
from src.serve import orchestrator as orch_mod
from src.serve.orchestrator import DiagnosticOrchestrator


class FakeRouter:
    def __init__(self, decision):
        self.d = decision

    def classify(self, frame):
        return self.d


def _reg(floor_ok):
    return {"coronary_angiography": TaskEntry(
        "coronary_angiography", "det", "best.pt", "Coronary angiography",
        "coronary_stenosis", "Possible coronary artery stenosis", floor_ok=floor_ok)}


def _det_factory(boxes):
    def factory(entry):
        return lambda frame: {"boxes": boxes, "top_conf": max([b[4] for b in boxes], default=0.0),
                              "deferred": False}
    return factory


def test_confident_coronary_frame_reports_kept_finding():
    router = FakeRouter(ModalityDecision("coronary_angiography", None, True, 0.95, False, "confident"))
    orch = DiagnosticOrchestrator(router, _reg(floor_ok=True), _det_factory([(0, 0, 9, 9, 0.9)]))
    r = orch.analyze_frame(frame_gray=object())
    assert r.modality == "coronary_angiography" and r.deferred is False
    assert r.findings[0].label == "coronary_stenosis" and r.findings[0].deferred is False


def test_unknown_modality_defers_with_no_disease_finding():
    router = FakeRouter(ModalityDecision("unknown", None, True, 0.4, True, "router-uncertain"))
    orch = DiagnosticOrchestrator(router, _reg(floor_ok=True), _det_factory([(0, 0, 9, 9, 0.9)]))
    r = orch.analyze_frame(frame_gray=object())
    assert r.deferred is True and r.defer_reason == "router-uncertain" and r.findings == []


# NOTE: the brief's draft expected defer_reason == "unsupported-modality" here. diagnosis.py was
# hardened after the brief was written: study_defer() now treats an EMPTY findings list as an
# unscreened study and returns ("no-findings") on its own -- this is EXACTLY the resolve()-returns-
# None path (see diagnosis.py's own docstring/comment for this case). The task brief explicitly says
# not to add a second, conflicting guard in the orchestrator for this -- so the orchestrator does not
# invent an "unsupported-modality" reason; it lets findings stay empty and defers to study_defer's
# existing "no-findings" guard. This test's expectation is corrected to match that authoritative,
# on-disk behavior rather than the brief's draft.
def test_supported_modality_with_no_registry_entry_defers_unsupported():
    router = FakeRouter(ModalityDecision("cerebral_dsa", None, True, 0.9, False, "confident"))
    orch = DiagnosticOrchestrator(router, _reg(floor_ok=True), _det_factory([]))
    r = orch.analyze_frame(frame_gray=object())
    assert r.deferred is True and r.defer_reason == "no-findings"
    assert r.findings == []


def test_below_floor_model_defers_finding():
    router = FakeRouter(ModalityDecision("coronary_angiography", None, True, 0.95, False, "confident"))
    orch = DiagnosticOrchestrator(router, _reg(floor_ok=False), _det_factory([(0, 0, 9, 9, 0.95)]))
    r = orch.analyze_frame(frame_gray=object())
    assert r.deferred is True and r.findings[0].reason == "below-floor"


# --- C2: a det output with no "boxes" key at all must not be read as a confident negative --------

def test_det_output_missing_boxes_key_defers_malformed_det():
    # A detector exported without NMS (or any producer whose output schema the caller could not
    # parse) may omit "boxes" entirely. `.get("boxes", [])` would silently read that as "zero
    # detections" -> triage_decision([]) -> "clean". "boxes" must be REQUIRED, not defaulted.
    router = FakeRouter(ModalityDecision("coronary_angiography", None, True, 0.95, False, "confident"))

    def factory(entry):
        return lambda frame: {"top_conf": 0.0, "deferred": True}   # no "boxes" key at all

    orch = DiagnosticOrchestrator(router, _reg(floor_ok=True), factory)
    r = orch.analyze_frame(frame_gray=object())
    assert r.deferred is True
    assert r.findings[0].deferred is True and r.findings[0].reason == "malformed-det"
    assert r.findings[0].boxes == []


# --- C4: the detector's OWN abstention flag must be honored, not just triage_decision's view ------

def test_det_model_own_deferred_flag_is_honored_even_when_triage_says_clean():
    # DetModel signals deferred=True (top_conf 0.15 < defer_below 0.4) on a box whose calibrated
    # confidence, read through triage_decision ALONE, falls through every defer branch to "clean"
    # (a confident negative). The model's own abstention must not be silently discarded.
    router = FakeRouter(ModalityDecision("coronary_angiography", None, True, 0.95, False, "confident"))

    def factory(entry):
        return lambda frame: {"boxes": [(0, 0, 9, 9, 0.15)], "top_conf": 0.15, "deferred": True}

    orch = DiagnosticOrchestrator(router, _reg(floor_ok=True), factory)
    r = orch.analyze_frame(frame_gray=object())
    assert r.deferred is True
    assert r.findings[0].deferred is True and r.findings[0].reason == "low-confidence"


def test_det_output_missing_deferred_key_fails_safe_to_deferred():
    # A det output missing its own "deferred" key (not merely False) must fail safe to treated-as-
    # deferred -- never silently treated as a confident False.
    router = FakeRouter(ModalityDecision("coronary_angiography", None, True, 0.95, False, "confident"))

    def factory(entry):
        return lambda frame: {"boxes": [], "top_conf": 0.0}        # "deferred" key absent

    orch = DiagnosticOrchestrator(router, _reg(floor_ok=True), factory)
    r = orch.analyze_frame(frame_gray=object())
    assert r.deferred is True
    assert r.findings[0].deferred is True and r.findings[0].reason == "low-confidence"


# --- audit-trail wiring: every prediction that actually reaches a model is logged ----------------

def test_confident_prediction_is_audit_logged(monkeypatch):
    calls = []
    monkeypatch.setattr(orch_mod, "record", lambda *a, **kw: calls.append((a, kw)))
    router = FakeRouter(ModalityDecision("coronary_angiography", None, True, 0.95, False, "confident"))
    orch = DiagnosticOrchestrator(router, _reg(floor_ok=True), _det_factory([(0, 0, 9, 9, 0.9)]))
    orch.analyze_frame(frame_gray=object())
    assert len(calls) == 1
    version, input_arr, summary = calls[0][0]
    assert version == "best.pt"
    assert summary["deferred"] is False and summary["finding"] == "coronary_stenosis"


def test_router_deferred_frame_never_reaches_model_or_audit(monkeypatch):
    calls = []
    monkeypatch.setattr(orch_mod, "record", lambda *a, **kw: calls.append((a, kw)))
    ran = []
    router = FakeRouter(ModalityDecision("unknown", None, True, 0.4, True, "router-uncertain"))

    def factory(entry):
        def _fail(frame):
            ran.append(True)
            raise AssertionError("model must not run when the router already deferred")
        return _fail

    orch = DiagnosticOrchestrator(router, _reg(floor_ok=True), factory)
    orch.analyze_frame(frame_gray=object())
    assert ran == [] and calls == []


def test_unsupported_modality_does_not_run_model_or_audit(monkeypatch):
    calls = []
    monkeypatch.setattr(orch_mod, "record", lambda *a, **kw: calls.append((a, kw)))
    ran = []
    router = FakeRouter(ModalityDecision("cerebral_dsa", None, True, 0.9, False, "confident"))

    def factory(entry):
        ran.append(True)
        return lambda frame: {"boxes": []}

    orch = DiagnosticOrchestrator(router, _reg(floor_ok=True), factory)
    orch.analyze_frame(frame_gray=object())
    assert ran == [] and calls == []


# --- router-unavailable fail-safe: an undeployed/unloadable router must defer, never crash -------
# (The video path -- analyze_video, temporal voting, tracking -- was deleted per the 2026-08-03
# audit P3 decision: Model One screens a single still frame only. Its tests went with it.)

class UnavailableRouter:
    """A decision source whose own load/run fails: classify raises RouterUnavailable. Today's
    `ValidityGate` is numpy-only and cannot fail this way, but the contract must stay pinned --
    any future gate with weights (e.g. the learned OOD head) can, and the orchestrator must defer
    rather than let the exception escape as a generic analysis-error."""
    weights = "gates/does-not-exist.pt"

    def classify(self, frame):
        raise orch_mod.RouterUnavailable("weights missing")


def test_analyze_frame_router_unavailable_defers_with_distinct_reason():
    # A router that cannot load/run must not escape as a raw exception (the old escape path was
    # ModuleNotFoundError -> the endpoint's generic 'analysis-error'); it must defer with its OWN
    # reason so operators can tell "router not deployed" apart from "genuine bug".
    orch = DiagnosticOrchestrator(UnavailableRouter(), _reg(floor_ok=True),
                                  _det_factory([(0, 0, 9, 9, 0.9)]))
    r = orch.analyze_frame(frame_gray=object())
    assert r.deferred is True and r.defer_reason == "router-unavailable"
    assert r.findings == [] and r.modality == "unknown"
    assert r.frames_analyzed == 1


def test_analyze_frame_router_unavailable_skips_model_and_audit(monkeypatch):
    # No modality decision -> no task model may run and nothing genuine exists to audit-log
    # (mirrors the model-unavailable path).
    calls = []
    monkeypatch.setattr(orch_mod, "record", lambda *a, **kw: calls.append((a, kw)))
    ran = []

    def factory(entry):
        def _fail(frame):
            ran.append(True)
            raise AssertionError("model must not run when the router is unavailable")
        return _fail

    orch = DiagnosticOrchestrator(UnavailableRouter(), _reg(floor_ok=True), factory)
    r = orch.analyze_frame(frame_gray=object())
    assert r.deferred is True and ran == [] and calls == []


def test_analyze_frame_real_gate_rejects_bad_input_without_running_a_model():
    # REAL ValidityGate, no fake: the gate is the orchestrator's live decision source, so at least
    # one test must exercise it unmocked. A degenerate frame must defer with the gate's OWN reason
    # and never reach the model. (This replaces the old real-ModalityRouter fail-safe test, which
    # went with the router in 2026-08-16; the RouterUnavailable contract itself is still pinned
    # above by UnavailableRouter, since any future gate can still fail to load.)
    import numpy as np
    ran = []

    def factory(entry):
        def _fail(frame):
            ran.append(True)
            raise AssertionError("model must not run on a frame the gate rejected")
        return _fail

    orch = DiagnosticOrchestrator(ValidityGate("coronary_angiography"), _reg(floor_ok=True), factory)
    r = orch.analyze_frame(np.full((512, 512), 128, np.uint8))
    assert r.deferred is True and r.defer_reason == "degenerate-contrast"
    assert r.modality == "unknown" and r.findings == [] and ran == []


# --- build_orchestrator: real gate+registry+infer wiring (D3) -------------------------------------

def test_build_orchestrator_wires_validity_gate_and_registry(tmp_path, monkeypatch):
    # Model One is single-modality (Dialygo B3), so the decision source is a validity GATE, not a
    # modality router: it vouches for the input or defers, it does not choose between modalities.
    cfg = tmp_path / "orch.yaml"
    cfg.write_text(
        "validity: {modality: coronary_angiography, accept_score: 0.5}\n"
        "modalities:\n"
        "  coronary_angiography: {task: seg, model_path: student.mlpackage, display_name: Coronary,\n"
        "    finding_label: coronary_vessels, finding_display: Vessel map, floor_ok: true}\n")
    monkeypatch.setattr(orch_mod, "_load_det", lambda p: (lambda f: {"boxes": [], "deferred": False}))
    monkeypatch.setattr(orch_mod, "_load_seg", lambda p: (lambda f: {"deferred": False, "confidence": 0.0}))
    orch = orch_mod.build_orchestrator(str(cfg))
    assert "coronary_angiography" in orch.registry
    assert isinstance(orch.gate, ValidityGate)
    assert orch.gate.modality == "coronary_angiography"


def test_build_orchestrator_rejects_a_config_with_no_validity_block(tmp_path):
    # Fail closed at wiring time: a config with no gate would leave every input unvouched-for.
    cfg = tmp_path / "orch.yaml"
    cfg.write_text(
        "modalities:\n"
        "  coronary_angiography: {task: seg, model_path: m, display_name: C,\n"
        "    finding_label: v, finding_display: V, floor_ok: false}\n")
    with pytest.raises(KeyError):
        orch_mod.build_orchestrator(str(cfg))


# --- model-unavailable fail-safe: missing/unloadable weights or an unknown task type must defer,
# never crash and never silently report a confident/clean result. ---------------------------------

def test_load_det_missing_weights_yields_callable_that_raises_model_unavailable():
    # No monkeypatch: the REAL DetModel constructor runs and fails on a nonexistent weights path --
    # proving the fail-safe wrapper (not a test double) is what defers.
    model = orch_mod._load_det("this/path/does/not/exist.mlpackage")
    raised = None
    try:
        model(object())
    except orch_mod.ModelUnavailable as e:
        raised = e
    assert raised is not None


def test_load_seg_missing_weights_yields_callable_that_raises_model_unavailable():
    model = orch_mod._load_seg("this/path/does/not/exist.mlpackage")
    raised = None
    try:
        model(object())
    except orch_mod.ModelUnavailable as e:
        raised = e
    assert raised is not None


def test_model_factory_unknown_task_type_yields_callable_that_raises_model_unavailable():
    entry = TaskEntry("weird_modality", "cls", "some.pt", "Weird",
                      "weird_finding", "Weird finding", floor_ok=True)
    model = orch_mod._model_factory(entry)
    raised = None
    try:
        model(object())
    except orch_mod.ModelUnavailable as e:
        raised = e
    assert raised is not None


def test_analyze_frame_defers_whole_study_on_missing_weights_without_crash():
    # Real _model_factory/_load_det, no monkeypatch: "best.pt" doesn't exist on disk, so DetModel
    # construction fails -- analyze_frame must defer the study, never crash the endpoint or report
    # a silently clean/confident result.
    router = FakeRouter(ModalityDecision("coronary_angiography", None, True, 0.95, False, "confident"))
    orch = DiagnosticOrchestrator(router, _reg(floor_ok=True), orch_mod._model_factory)
    r = orch.analyze_frame(frame_gray=object())
    assert r.deferred is True and r.defer_reason == "model-unavailable" and r.findings == []


def test_analyze_frame_model_unavailable_skips_audit(monkeypatch):
    calls = []
    monkeypatch.setattr(orch_mod, "record", lambda *a, **kw: calls.append((a, kw)))
    router = FakeRouter(ModalityDecision("coronary_angiography", None, True, 0.95, False, "confident"))
    orch = DiagnosticOrchestrator(router, _reg(floor_ok=True), orch_mod._model_factory)
    orch.analyze_frame(frame_gray=object())
    assert calls == []


# --- Model One: the cls task path -------------------------------------------------------------


def _cls_entry(floor_ok=True, model_path="head.pt"):
    return TaskEntry("avf_fistulography", "cls", model_path, "AVF fistulography",
                     "avf_ja_stenosis", "Possible juxta-anastomotic stenosis", floor_ok=floor_ok)


def test_cls_modality_flows_through_analyze_frame(monkeypatch):
    import numpy as np
    monkeypatch.setattr(orch_mod, "record", lambda *a, **k: None)
    router = FakeRouter(ModalityDecision("avf_fistulography", None, True, 0.9, False, "confident"))

    def cls_factory(e):
        return lambda frame: {"prob": 0.9, "confidence": 0.9, "deferred": False,
                              "reason": "confident", "threshold": 0.5}
    orch = DiagnosticOrchestrator(router, {"avf_fistulography": _cls_entry()}, cls_factory)
    report = orch.analyze_frame(np.zeros((16, 16), np.uint8))
    assert report.findings[0].label == "avf_ja_stenosis" and not report.deferred


def test_cls_result_is_not_routed_through_the_seg_branch(monkeypatch):
    """Before the cls branch existed, `else: seg_to_finding(...)` swallowed every non-det task. A
    cls dict has no 'confidence'-plus-'deferred' seg contract quirk to catch it, so a mis-branch
    would silently mislabel the reason -- pin the cls reason, which only cls_to_finding produces."""
    import numpy as np
    monkeypatch.setattr(orch_mod, "record", lambda *a, **k: None)
    router = FakeRouter(ModalityDecision("avf_fistulography", None, True, 0.9, False, "confident"))

    def cls_factory(e):
        return lambda frame: {"prob": 0.45, "confidence": 0.55, "deferred": True,
                              "reason": "defer-band", "threshold": 0.5}
    orch = DiagnosticOrchestrator(router, {"avf_fistulography": _cls_entry()}, cls_factory)
    report = orch.analyze_frame(np.zeros((16, 16), np.uint8))
    assert report.deferred and report.findings[0].reason == "defer-band"


def test_unknown_cls_checkpoint_defers_model_unavailable():
    import numpy as np
    import pytest as _pytest
    from src.serve.orchestrator import _model_factory
    model = _model_factory(_cls_entry(model_path="definitely/absent/head.pt"))
    with _pytest.raises(orch_mod.ModelUnavailable):
        model(np.zeros((8, 8), dtype=np.uint8))


# --- import-safety guardrail: importing this module must never pull in torch/ultralytics/coremltools

def test_orchestrator_module_imports_without_torch():
    # Run in a FRESH interpreter (subprocess) so torch already loaded into sys.modules by an
    # EARLIER test file (e.g. test_clgeodice imports torch at module level) can't defeat the check --
    # the property under test is orchestrator's OWN import, not global state. Mirrors
    # tests/test_router.py::test_router_module_imports_without_torch.
    import os, subprocess, sys, textwrap
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = textwrap.dedent("""
        import sys, importlib
        importlib.import_module("src.serve.orchestrator")
        for mod in ("torch", "timm", "cv2", "coremltools", "ultralytics", "transformers"):
            assert mod not in sys.modules, f"orchestrator import pulled in {mod}"
    """)
    r = subprocess.run([sys.executable, "-c", code], cwd=repo_root,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_cls_finding_is_mirrored_on_the_event_bus(monkeypatch):
    """The publish site is task-agnostic, so `model.inferred` should carry task='cls'. Asserted,
    not assumed -- the event stream is the only observability the hosted deployment has (B8)."""
    import numpy as np
    from src.serve.events import EventBus
    monkeypatch.setattr(orch_mod, "record", lambda *a, **k: None)
    seen = []
    bus = EventBus()
    bus.subscribe("model.inferred", seen.append)
    router = FakeRouter(ModalityDecision("avf_fistulography", None, True, 0.9, False, "confident"))

    def cls_factory(e):
        return lambda frame: {"prob": 0.9, "confidence": 0.9, "deferred": False,
                              "reason": "confident", "threshold": 0.5}
    orch = DiagnosticOrchestrator(router, {"avf_fistulography": _cls_entry()}, cls_factory, bus=bus)
    orch.analyze_frame(np.zeros((16, 16), np.uint8))
    assert [e["data"]["task"] for e in seen] == ["cls"]
    assert seen[0]["data"]["finding_label"] == "avf_ja_stenosis"
