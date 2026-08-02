"""Test the diagnostic orchestrator: route -> resolve -> infer -> typed findings -> StudyReport.

Router and model are both fakes/injected -- no torch import anywhere in this file, matching the
import-safe convention the orchestrator itself follows.
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


# --- analyze_video: sample -> route each -> temporal-vote aggregate ------------------------------

def test_video_majority_modality_and_temporal_vote(monkeypatch):
    # 5 coronary frames; a stenosis box present in >=2 -> aggregate keeps it, single-frame flicker dropped
    router = FakeRouter(ModalityDecision("coronary_angiography", None, True, 0.95, False, "confident"))
    per_frame_boxes = [
        [(10, 10, 30, 30, 0.8)],           # f0 real
        [(10, 10, 30, 30, 0.82)],          # f1 real (2 hits -> kept)
        [],                                # f2
        [(200, 200, 210, 210, 0.4)],       # f3 flicker (1 hit -> dropped by min_hits=2)
        [(11, 11, 31, 31, 0.79)],          # f4 real
    ]
    calls = {"i": 0}
    def factory(entry):
        def model(frame):
            b = per_frame_boxes[calls["i"]]; calls["i"] += 1
            return {"boxes": b, "top_conf": max([x[4] for x in b], default=0.0), "deferred": False}
        return model
    orch = DiagnosticOrchestrator(router, _reg(floor_ok=True), factory)
    monkeypatch.setattr(orch, "_iter_frames", lambda path, stride, max_frames: iter([object()] * 5))
    r = orch.analyze_video("clip.mp4")
    assert r.modality == "coronary_angiography" and r.frames_analyzed == 5
    assert r.findings[0].label == "coronary_stenosis"
    # the flicker box (only 1 hit) must not survive aggregation into a kept detection
    assert all(not (abs(b[0] - 200) < 5) for b in r.findings[0].boxes)


# --- C3: temporal voting may filter what is REPORTED, but must not silence the DEFER decision ----

def test_video_defers_on_pre_vote_low_confidence_even_if_track_dropped_by_vote(monkeypatch):
    # One box at conf 0.55 on ONE sampled frame: through analyze_frame this defers as
    # 'low-confidence' (the calibrated confidence sits inside the defer band). Through
    # analyze_video, temporal voting drops the 1-hit track (min_hits=2 default) before its
    # confidence ever reaches triage -- silencing the defer and reporting "clean" on a study the
    # model was actually uncertain about. The defer decision must be derived from evidence BEFORE
    # voting; only the reported boxes are allowed to be post-vote.
    router = FakeRouter(ModalityDecision("coronary_angiography", None, True, 0.95, False, "confident"))
    per_frame_boxes = [[], [(0, 0, 10, 10, 0.55)], []]
    calls = {"i": 0}

    def factory(entry):
        def model(frame):
            b = per_frame_boxes[calls["i"]]
            calls["i"] += 1
            return {"boxes": b, "top_conf": max([x[4] for x in b], default=0.0), "deferred": False}
        return model

    orch = DiagnosticOrchestrator(router, _reg(floor_ok=True), factory)
    monkeypatch.setattr(orch, "_iter_frames", lambda path, stride, max_frames: iter([object()] * 3))
    r = orch.analyze_video("clip.mp4")
    assert r.deferred is True
    assert r.findings[0].reason == "low-confidence"


# --- I1: the video path must not run the detector on frames the router refused to route -----------

def test_video_detector_never_runs_on_frames_router_did_not_route(monkeypatch):
    # f1 is router-uncertain and f2 is routed to a DIFFERENT modality -- neither may reach the
    # majority modality's ("coronary_angiography") detector.
    f0, f1, f2, f3 = object(), object(), object(), object()
    coronary = ModalityDecision("coronary_angiography", None, True, 0.95, False, "confident")
    router_deferred = ModalityDecision("unknown", None, True, 0.4, True, "router-uncertain")
    other_modality = ModalityDecision("cerebral_dsa", None, True, 0.9, False, "confident")
    per_frame_decision = {f0: coronary, f1: router_deferred, f2: other_modality, f3: coronary}

    class MultiRouter:
        def classify(self, frame):
            return per_frame_decision[frame]

    ran_on = []

    def factory(entry):
        def model(frame):
            ran_on.append(frame)
            return {"boxes": [], "top_conf": 0.0, "deferred": False}
        return model

    orch = DiagnosticOrchestrator(MultiRouter(), _reg(floor_ok=True), factory)
    monkeypatch.setattr(orch, "_iter_frames", lambda path, stride, max_frames: iter([f0, f1, f2, f3]))
    orch.analyze_video("clip.mp4")
    assert ran_on == [f0, f3]


def test_video_excluded_frames_leave_index_preserving_gap_not_compacted(monkeypatch):
    # Two real detections at f0 and f4 are separated by three router-deferred frames (f1..f3) --
    # a true index gap of 3, which exceeds temporal_vote's default max_gap=1, so they must NOT
    # link into one persistent track. Compacting the excluded frames out of the sequence instead
    # of leaving an empty slot per excluded frame would make f0/f4 adjacent (gap 0) and wrongly
    # link them into a 2-hit "persistent" track.
    f0, f1, f2, f3, f4 = [object() for _ in range(5)]
    coronary = ModalityDecision("coronary_angiography", None, True, 0.95, False, "confident")
    router_deferred = ModalityDecision("unknown", None, True, 0.4, True, "router-uncertain")
    per_frame_decision = {f0: coronary, f1: router_deferred, f2: router_deferred,
                          f3: router_deferred, f4: coronary}

    class MultiRouter:
        def classify(self, frame):
            return per_frame_decision[frame]

    box = (10, 10, 30, 30, 0.8)

    def factory(entry):
        def model(frame):
            return {"boxes": [box], "top_conf": 0.8, "deferred": False}
        return model

    orch = DiagnosticOrchestrator(MultiRouter(), _reg(floor_ok=True), factory)
    monkeypatch.setattr(orch, "_iter_frames",
                        lambda path, stride, max_frames: iter([f0, f1, f2, f3, f4]))
    r = orch.analyze_video("clip.mp4")
    # each detection is a lone hit, too far apart (true gap 3) to link -> dropped by min_hits=2 ->
    # nothing survives temporal voting to be reported.
    assert r.findings[0].boxes == []


def test_video_with_zero_frames_defers_never_reports_clean(monkeypatch):
    # An undecodable clip / empty iterator must defer -- never a clean report on no evidence.
    router = FakeRouter(ModalityDecision("coronary_angiography", None, True, 0.95, False, "confident"))
    orch = DiagnosticOrchestrator(router, _reg(floor_ok=True), _det_factory([(0, 0, 9, 9, 0.9)]))
    monkeypatch.setattr(orch, "_iter_frames", lambda path, stride, max_frames: iter([]))
    r = orch.analyze_video("clip.mp4")
    assert r.deferred is True and r.frames_analyzed == 0 and r.findings == []


def test_video_all_frames_router_deferred_defers_whole_study_and_skips_model(monkeypatch):
    # Every sampled frame is router-uncertain -> defer the whole study; no wasted inference.
    ran = []
    def factory(entry):
        def _fail(frame):
            ran.append(True)
            raise AssertionError("model must not run when every frame's router call deferred")
        return _fail
    router = FakeRouter(ModalityDecision("unknown", None, True, 0.4, True, "router-uncertain"))
    orch = DiagnosticOrchestrator(router, _reg(floor_ok=True), factory)
    monkeypatch.setattr(orch, "_iter_frames", lambda path, stride, max_frames: iter([object()] * 3))
    r = orch.analyze_video("clip.mp4")
    assert r.deferred is True and r.defer_reason == "router-uncertain"
    assert r.frames_analyzed == 3 and r.findings == [] and ran == []


def test_video_unsupported_modality_defers_no_findings(monkeypatch):
    router = FakeRouter(ModalityDecision("cerebral_dsa", None, True, 0.9, False, "confident"))
    orch = DiagnosticOrchestrator(router, _reg(floor_ok=True), _det_factory([]))
    monkeypatch.setattr(orch, "_iter_frames", lambda path, stride, max_frames: iter([object()] * 3))
    r = orch.analyze_video("clip.mp4")
    assert r.deferred is True and r.defer_reason == "no-findings" and r.findings == []


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


def test_analyze_video_defers_whole_study_on_missing_weights_without_crash(monkeypatch):
    router = FakeRouter(ModalityDecision("coronary_angiography", None, True, 0.95, False, "confident"))
    orch = DiagnosticOrchestrator(router, _reg(floor_ok=True), orch_mod._model_factory)
    monkeypatch.setattr(orch, "_iter_frames", lambda path, stride, max_frames: iter([object()] * 3))
    r = orch.analyze_video("clip.mp4")
    assert r.deferred is True and r.defer_reason == "model-unavailable"
    assert r.frames_analyzed == 3 and r.findings == []


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
