"""Test the diagnostic orchestrator: route -> resolve -> infer -> typed findings -> StudyReport.

Router and model are both fakes/injected -- no torch import at module scope anywhere in this file,
matching the import-safe convention the orchestrator itself follows. (One fail-safe test runs a
REAL ModalityRouter whose lazy load is EXPECTED to fail -- torch/timm import is attempted at
runtime there and allowed to be missing.)
"""
from src.serve.registry import TaskEntry
from src.serve.router import ModalityDecision
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
    """Mimics ModalityRouter's fail-safe contract: classify raises RouterUnavailable when the
    router itself (weights file / timm / torch) could not be loaded or run."""
    weights = "runs/router/does-not-exist.pt"

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


def test_analyze_frame_real_router_missing_weights_defers_router_unavailable():
    # REAL ModalityRouter, no monkeypatch (the one test in this file that may touch the heavy
    # stack at runtime -- the torch/timm import is attempted and allowed to fail): the weights
    # file does not exist (and/or timm is not installed), so the router's own load fails.
    # analyze_frame must return a deferred report with reason "router-unavailable" -- never let a
    # ModuleNotFoundError/FileNotFoundError escape to the endpoint as a generic analysis-error.
    from src.serve.router import ModalityRouter
    router = ModalityRouter("this/path/does/not/exist.pt",
                            ["coronary_angiography", "other_xray"])
    orch = DiagnosticOrchestrator(router, _reg(floor_ok=True), _det_factory([]))
    r = orch.analyze_frame(frame_gray=object())
    assert r.deferred is True and r.defer_reason == "router-unavailable"
    assert r.findings == []


# --- build_orchestrator: real router+registry+infer wiring (D3) ----------------------------------

def test_build_orchestrator_wires_router_and_registry(tmp_path, monkeypatch):
    cfg = tmp_path / "orch.yaml"
    cfg.write_text(
        "router: {weights: runs/router/student.pt, labels: [coronary_angiography, other_xray],\n"
        "         thresholds: {keep_thr: 0.6, margin: 0.15, quality_thr: 0.5}}\n"
        "modalities:\n"
        "  coronary_angiography: {task: det, model_path: best.pt, display_name: Coronary,\n"
        "    finding_label: coronary_stenosis, finding_display: Possible stenosis, floor_ok: false}\n")
    monkeypatch.setattr(orch_mod, "_load_det", lambda p: (lambda f: {"boxes": [], "deferred": False}))
    monkeypatch.setattr(orch_mod, "_load_seg", lambda p: (lambda f: {"deferred": False, "confidence": 0.0}))
    orch = orch_mod.build_orchestrator(str(cfg))
    assert "coronary_angiography" in orch.registry
    assert orch.router.labels == ["coronary_angiography", "other_xray"]


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
