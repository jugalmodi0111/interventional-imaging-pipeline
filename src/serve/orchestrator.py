"""Diagnostic orchestrator: gate a frame -> the right task model -> typed findings -> StudyReport.

Ties together the validity gate, registry, and diagnosis layer: `analyze_frame` asks the gate
whether this input may be read at all, resolves the resulting modality to a TaskEntry, runs the
injected model, and turns its raw output into typed Findings, all funneled through `study_defer`
for the final study-level DEFER call. Models are injected via
`model_factory(entry) -> callable(frame_gray) -> dict` so this file is unit-tested with fakes -- no
torch/ultralytics/coremltools import here, ever. `build_orchestrator(cfg_path)` (bottom of this
file) is the one factory that wires a REAL model_factory (`_model_factory`, backed by
`src.serve.infer.DetModel`/`SegModel`) plus a real `ValidityGate` from a YAML config; even it only
imports the heavy stack lazily, inside its own body, so `import src.serve.orchestrator` itself
never touches torch/ultralytics/coremltools -- only *calling* `build_orchestrator` (or a model
callable it built) does.

The decision source was a 4-class `ModalityRouter` until 2026-08-16. Dialygo B3 specifies a
*validity gate* ("reject any input that is not a valid vascular-access angiogram"), not a modality
router, and Model One is single-modality -- so the router was deleted and `src.serve.validity.
ValidityGate` took its place behind the identical `.classify(frame) -> ModalityDecision` protocol.

A model callable built by `_model_factory` never lets a construction failure escape as a raw
exception: a missing weights file, an unloadable/corrupt one, a missing heavy dependency, or an
unrecognized `entry.task` all collapse to a callable that raises `ModelUnavailable` on first use.
`analyze_frame` catches that at the single point a model is actually invoked and defers the WHOLE
study (reason "model-unavailable") -- same fail-safe posture as every other defer path in this
file: never a crash that takes down the endpoint, never a silently empty confident result. The
gate gets the same treatment: a `classify` that collapses its own load/run failures into
`RouterUnavailable` makes `analyze_frame` defer the study (reason "router-unavailable") so an
operator can tell "the gate isn't deployed" apart from a genuine analysis bug -- previously it
escaped as a raw ModuleNotFoundError and surfaced as the endpoint's generic "analysis-error".
Today's `ValidityGate` is numpy-only and cannot fail this way, but the contract stays in place for
the learned OOD gate that will have weights.

Safety default is DEFER, not guess: a gate decision that already deferred (input rejected, quality
too low, uncertain) short-circuits before any model runs -- no wasted inference, no spurious audit
entry, no confident diagnosis the gate itself didn't trust. A resolved modality with no registered TaskEntry
(`registry.resolve` -> None) is deliberately NOT special-cased here: `study_defer` already treats an
empty findings list as an unscreened study (reason "no-findings") -- adding a second, conflicting
"unsupported-modality" guard on top of that would just race the existing one, so we don't.

Every prediction that actually reaches a model is audit-logged (src.eval.audit.record) with the model
version, the input frame, and the resulting finding -- preserving the existing audit-trail convention
at the one place with full context (modality + entry + finding) to summarize it.

Model One (B3) screens a SINGLE STILL FRAME. The former cine/video path (`analyze_video`, temporal
voting, tracking) was deleted per the 2026-08-03 audit (P3): its two remaining criticals lived
entirely inside it, and `/analyze?kind=video` now refuses with HTTP 400 at the endpoint instead of
routing anywhere.
"""
from dataclasses import replace
from src.serve.report import StudyReport, Finding
from src.serve.registry import resolve
from src.serve.diagnosis import cls_to_finding, det_to_findings, seg_to_finding, study_defer
from src.serve.stenosis_triage import triage_decision
from src.eval.audit import input_hash, record


class DiagnosticOrchestrator:
    """gate(input) -> resolve(model) -> infer -> typed findings -> StudyReport, deferring on any
    uncertainty. `model_factory(entry) -> callable(frame_gray) -> dict` is injected so tests supply
    fakes; this class never constructs a real model itself.

    `gate` is any object exposing `.classify(frame) -> ModalityDecision` (today
    `src.serve.validity.ValidityGate`). It is the sole authority on whether a frame may be read at
    all: a decision it defers, or that names modality "unknown", stops the pipeline before any
    disease model runs. NB the event topics (`router.decided`, `router.unavailable`) and the defer
    reason `"router-unavailable"` still carry the pre-B3 name -- they are the published /events and
    report contract, so renaming them is a deliberate versioned change, not a refactor."""

    def __init__(self, gate, registry, model_factory, cfg=None, bus=None):
        self.gate = gate
        self.registry = registry
        self.model_factory = model_factory
        self.cfg = cfg or {}
        self._models = {}
        self.bus = bus                       # observe-only event mirror; None = silent (no-op)

    def _publish(self, topic, **data):
        """Mirror one pipeline step onto the event bus. Never load-bearing: with no bus attached
        this is a no-op, and a crashing subscriber is the bus's problem (counted there), not ours."""
        if self.bus is not None:
            self.bus.publish(topic, **data)

    def _emit_verdict(self, report):
        self._publish("verdict.emitted", modality=report.modality, deferred=report.deferred,
                      defer_reason=report.defer_reason, n_findings=len(report.findings))
        return report

    def _model_for(self, entry):
        """Lazily build (and cache) the model callable for a modality via the injected factory."""
        if entry.modality not in self._models:
            self._models[entry.modality] = self.model_factory(entry)
        return self._models[entry.modality]

    def _report(self, decision, findings, frames_analyzed, versions):
        deferred, reason = study_defer(decision, findings)
        return self._emit_verdict(StudyReport(modality=decision.modality, view=decision.view,
                                              quality_ok=decision.quality_ok, findings=findings,
                                              deferred=deferred, defer_reason=reason,
                                              frames_analyzed=frames_analyzed,
                                              model_versions=versions))

    def _det_findings(self, entry, out):
        """Detector output -> [Finding], honoring BOTH abstention signals a det model can raise:

          * C2 -- a MISSING 'boxes' key is a malformed output, not a confident negative. Reading it
            via `.get("boxes", [])` would silently treat "the model's output schema didn't parse"
            (see infer.py's `_parse_yolo_coreml`, which returns an empty list on exactly that case)
            the same as "the model looked and found nothing" -- those are not the same claim, and
            only the second one is allowed to become a clean report. Missing key -> defer with
            reason 'malformed-det' before triage ever runs.
          * C4 -- the model's OWN 'deferred' flag (e.g. `DetModel`'s `top_conf < defer_below`) must
            be honored even when `triage_decision`, looking at the same boxes, would otherwise call
            it 'clean' or 'confident' -- a low-confidence detection sitting just below triage's own
            defer band must not slip through because triage and the model disagree. A MISSING
            'deferred' key fails safe to True (never silently treated as a confident False).
        """
        if "boxes" not in out:
            return [Finding(label=entry.finding_label, display_name=entry.finding_display,
                            confidence=0.0, deferred=True, reason="malformed-det", boxes=[])]

        triage = triage_decision(out["boxes"], temperature=self.cfg.get("temperature", 1.0))
        finding = det_to_findings(entry, triage)[0]
        if out.get("deferred", True) and not finding.deferred:
            finding = replace(finding, deferred=True, reason="low-confidence")
        return [finding]

    def analyze_frame(self, frame_gray):
        """Classify one frame's modality, run its task model (if any), and return a StudyReport.
        Never raises on router/model uncertainty -- it defers instead."""
        versions = {"gate": getattr(self.gate, "weights", "validity-gate")}
        self._publish("frame.received", input_hash=input_hash(frame_gray))
        try:
            dec = self.gate.classify(frame_gray)
        except RouterUnavailable:
            self._publish("router.unavailable", router=versions["gate"])
            # The router itself could not be loaded or run (weights not deployed, timm/torch
            # missing, corrupt state_dict). Without a modality decision NO task model may run, so
            # defer the whole study -- with a reason an operator can tell apart from both a genuine
            # analysis bug ("analysis-error" at the endpoint) and a missing task model
            # ("model-unavailable"). No audit entry: nothing genuine ran (mirrors the
            # model-unavailable path below).
            return self._emit_verdict(StudyReport(
                modality="unknown", view=None, quality_ok=False,
                findings=[], deferred=True, defer_reason="router-unavailable",
                frames_analyzed=1, model_versions=versions))

        self._publish("router.decided", modality=dec.modality, deferred=dec.deferred,
                      reason=dec.reason, confidence=float(dec.confidence))
        if dec.deferred:
            # Router already distrusts this frame (unknown modality, thin margin, low quality) --
            # don't spend inference or an audit entry on a study that is deferring anyway.
            return self._report(dec, [], 1, versions)

        entry = resolve(self.registry, dec.modality)
        if entry is None:
            # Confidently-classified modality, but nothing is wired up to screen it. No model runs,
            # so findings stays empty; study_defer's own "no-findings" guard covers this case.
            return self._report(dec, [], 1, versions)

        versions[entry.finding_label] = entry.model_path
        try:
            out = self._model_for(entry)(frame_gray)
        except ModelUnavailable:
            # Weights missing/unloadable, or an unrecognized task type: the model never actually
            # ran, so defer the whole study rather than let triage/seg post-processing misread an
            # empty/absent output as a confident negative. No audit entry either -- there is nothing
            # genuine to log (mirrors the router-deferred / unsupported-modality paths above).
            self._publish("model.unavailable", modality=dec.modality,
                          model_path=entry.model_path)
            return self._emit_verdict(StudyReport(
                modality=dec.modality, view=dec.view, quality_ok=dec.quality_ok,
                findings=[], deferred=True, defer_reason="model-unavailable",
                frames_analyzed=1, model_versions=versions))

        # Explicit three-way branch, not det-vs-everything-else: an unrecognized entry.task never
        # reaches here (its model callable raised ModelUnavailable above), so `else` is exactly
        # "seg" and each task keeps its own reason vocabulary -- a cls "defer-band" routed through
        # seg_to_finding would come back relabelled "low-confidence".
        if entry.task == "det":
            findings = self._det_findings(entry, out)
        elif entry.task == "cls":
            findings = [cls_to_finding(entry, out)]
        else:
            findings = [seg_to_finding(entry, out)]

        finding = findings[0]
        self._publish("model.inferred", finding_label=entry.finding_label, task=entry.task,
                      confidence=float(finding.confidence), deferred=finding.deferred,
                      reason=finding.reason)
        record(entry.model_path, frame_gray,
              {"modality": dec.modality, "task": entry.task, "finding": entry.finding_label,
               "confidence": finding.confidence, "deferred": finding.deferred,
               "reason": finding.reason})

        return self._report(dec, findings, 1, versions)


# --- real model_factory: the seam between tested pure orchestration and heavy inference code -----


class ModelUnavailable(Exception):
    """Raised by a model callable -- never by DiagnosticOrchestrator itself -- when the underlying
    task model could not be produced: a missing weights file, an unloadable/corrupt weights file, a
    missing heavy dependency (coremltools/ultralytics/torch), or an unrecognized `entry.task`.
    `analyze_frame` catches this at the single point a model is actually invoked and converts it
    into a deferred study -- never a crash, never a silently empty confident result."""


class RouterUnavailable(RuntimeError):
    """Raised by a decision gate's `classify` when the GATE ITSELF could not be loaded or run: a
    missing weights file, a missing dependency, a corrupt state_dict, an error mid-forward. The
    gate counterpart of `ModelUnavailable`: `analyze_frame` catches it and defers the whole study
    with reason "router-unavailable", so an undeployed gate is operationally distinguishable from
    a genuine bug -- which still surfaces as the endpoint's generic "analysis-error".

    The name predates the 2026-08-16 router -> validity-gate change and is kept deliberately: it is
    part of the published `StudyReport.defer_reason` / `/events` contract, so renaming it is a
    versioned change rather than a refactor (tracked in PROJECT_TRACKER §4.2). No current gate can
    raise it -- `ValidityGate` is numpy-only -- but the learned OOD gate will have weights."""


def _load_det(model_path):
    """Build a real detection-model callable backed by `src.serve.infer.DetModel` (YOLO exported to
    CoreML). `coremltools` is only imported transitively, inside `DetModel.__init__` -- which itself
    only runs when THIS function is called, never at module import time -- so importing
    `src.serve.orchestrator` stays torch/coreml-free.

    If the weights can't be loaded (file missing, corrupt/unloadable, or coremltools itself missing)
    the constructor's exception is caught HERE, at build time, and turned into a callable that always
    raises `ModelUnavailable` when invoked. That keeps the failure scoped to this one modality (a bad
    weights file downgrades it to 'this modality always defers') instead of letting it propagate out
    of `build_orchestrator` -- or, worse, out of `analyze_frame` -- as a live crash.
    """
    from src.serve.infer import DetModel
    try:
        model = DetModel(model_path)
    except Exception as e:
        def _unavailable(frame, _path=model_path, _e=e):
            raise ModelUnavailable(f"det model at {_path!r} failed to load: {_e}") from _e
        return _unavailable
    return lambda frame: model(frame)


def _load_seg(model_path):
    """Segmentation counterpart of `_load_det` -- see its docstring for the lazy-import and
    fail-safe-on-load-failure rationale. Backed by `src.serve.infer.SegModel`."""
    from src.serve.infer import SegModel
    try:
        model = SegModel(model_path)
    except Exception as e:
        def _unavailable(frame, _path=model_path, _e=e):
            raise ModelUnavailable(f"seg model at {_path!r} failed to load: {_e}") from _e
        return _unavailable
    return lambda frame: model(frame)


def _load_cls(model_path):
    """Model One counterpart of `_load_det`/`_load_seg` -- see `_load_det` for the rationale.
    Backed by `src.serve.infer_cls.ClsModel` (hosted torch, B8; NOT the CoreML edge path).

    The import sits INSIDE the try, unlike its two neighbours: this path's heavy dependency is
    torch, which the hosted deployment can be missing outright, and `_load_det`'s own docstring
    says the goal is to scope such a failure to one modality. An ImportError here therefore
    downgrades AVF to 'always defers' rather than taking `build_orchestrator` down with it.
    """
    try:
        from src.serve.infer_cls import ClsModel
        model = ClsModel(model_path)
    except Exception as e:
        def _unavailable(frame, _path=model_path, _e=e):
            raise ModelUnavailable(f"cls model at {_path!r} failed to load: {_e}") from _e
        return _unavailable
    return lambda frame: model(frame)


def _model_factory(entry):
    """The real `model_factory(entry) -> callable(frame_gray) -> dict` that `build_orchestrator`
    wires in. `entry.task` selects `DetModel` vs `SegModel` (via `_load_det`/`_load_seg`, looked up
    by module-level name so tests can monkeypatch either loader without needing real weights). Any
    other `entry.task` value -- a registry typo, a future task type nothing here understands yet --
    is NOT a crash: it returns a callable that raises `ModelUnavailable` on first use, exactly like a
    missing/unloadable weights file, so `analyze_frame` defers the study the same way for both
    failure modes."""
    if entry.task == "det":
        return _load_det(entry.model_path)
    if entry.task == "seg":
        return _load_seg(entry.model_path)
    if entry.task == "cls":
        return _load_cls(entry.model_path)

    def _unknown_task(frame, _entry=entry):
        raise ModelUnavailable(f"unknown task type {_entry.task!r} for modality {_entry.modality!r}")
    return _unknown_task


def build_orchestrator(cfg_path):
    """Wire a real `DiagnosticOrchestrator` from a YAML config: a real `ValidityGate` (Dialygo B3
    input gate) and the real `_model_factory` (real `DetModel`/`SegModel` per registry entry,
    fail-safe on a bad weights file -- see `ModelUnavailable`). `yaml`/`ValidityGate`/
    `load_registry` are all imported here, inside the function body, not at module scope: calling
    `build_orchestrator` is the one place this module touches the heavy stack, and even then only
    through lazy imports several calls deep (`_model_factory` doesn't load coremltools until a model
    is actually invoked) -- importing the module itself never does. The gate itself is numpy-only.

    The `validity:` block is REQUIRED and raises `KeyError` if absent: a config with no gate would
    leave every input unvouched-for, which is exactly the failure B3 exists to prevent, so this
    fails closed at wiring time rather than at the first request.

    Config shape (see tests/test_orchestrator.py for a minimal example):
        validity: {modality: <the one deployed modality>, accept_score: 0.5}
        modalities: {<modality>: {task, model_path, display_name, finding_label, finding_display,
                                   floor_ok}, ...}
        runtime: {temperature}   # optional, passed through as cfg
    """
    import yaml
    from src.serve.events import EventBus, JsonlSink
    from src.serve.validity import ValidityGate
    from src.serve.registry import load_registry

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    vc = cfg["validity"]
    gate = ValidityGate(vc["modality"], **{k: v for k, v in vc.items() if k != "modality"})
    registry = load_registry(cfg_path)

    # Observe-only event mirror: everything the pipeline does is published here, persisted next to
    # the audit trail (runs/events.jsonl) and replayable live via GET /events. What was registered
    # is itself the first thing on the bus.
    bus = EventBus()
    bus.subscribe("*", JsonlSink("runs/events.jsonl"))
    for entry in registry.values():
        bus.publish("registry.loaded", modality=entry.modality, task=entry.task,
                    model_path=entry.model_path, floor_ok=entry.floor_ok)
    return DiagnosticOrchestrator(gate, registry, _model_factory, cfg=cfg.get("runtime", {}),
                                  bus=bus)
