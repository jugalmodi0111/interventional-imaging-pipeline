"""Diagnostic orchestrator: route a frame -> the right task model -> typed findings -> StudyReport.

Ties together the router, registry, and diagnosis layer: `analyze_frame` classifies the modality,
resolves it to a TaskEntry, runs the injected model, and turns its raw output into typed Findings,
all funneled through `study_defer` for the final study-level DEFER call. Models are injected via
`model_factory(entry) -> callable(frame_gray) -> dict` so this file is unit-tested with fakes -- no
torch/ultralytics/coremltools import here, ever. `build_orchestrator(cfg_path)` (bottom of this
file) is the one factory that wires a REAL model_factory (`_model_factory`, backed by
`src.serve.infer.DetModel`/`SegModel`) plus a real `ModalityRouter` from a YAML config; even it only
imports the heavy stack lazily, inside its own body, so `import src.serve.orchestrator` itself
never touches torch/ultralytics/coremltools -- only *calling* `build_orchestrator` (or a model
callable it built) does.

A model callable built by `_model_factory` never lets a construction failure escape as a raw
exception: a missing weights file, an unloadable/corrupt one, a missing heavy dependency, or an
unrecognized `entry.task` all collapse to a callable that raises `ModelUnavailable` on first use.
`analyze_frame`/`analyze_video` catch that at the single point a model is actually invoked and defer
the WHOLE study (reason "model-unavailable") -- same fail-safe posture as every other defer path in
this file: never a crash that takes down the endpoint, never a silently empty confident result.

Safety default is DEFER, not guess: a router that already deferred (unknown modality, low quality,
thin margin) short-circuits before any model runs -- no wasted inference, no spurious audit entry, no
confident diagnosis the router itself didn't trust. A resolved modality with no registered TaskEntry
(`registry.resolve` -> None) is deliberately NOT special-cased here: `study_defer` already treats an
empty findings list as an unscreened study (reason "no-findings") -- adding a second, conflicting
"unsupported-modality" guard on top of that would just race the existing one, so we don't.

Every prediction that actually reaches a model is audit-logged (src.eval.audit.record) with the model
version, the input frame, and the resulting finding -- preserving the existing audit-trail convention
at the one place with full context (modality + entry + finding) to summarize it.

`analyze_video` extends the same route -> resolve -> infer -> typed-findings pipeline to a cine clip:
sample frames, classify + screen each with the SAME machinery `analyze_frame` uses, then fold the
per-frame detections across the window with temporal-vote aggregation
(src.serve.temporal_vote.aggregate_sequence) so a one-frame flicker cannot become a finding while a
lesion that persists across frames does. Safety default carries over unchanged: an undecodable clip,
a window with zero usable frames, or a window where no frame was confidently routed all defer the
WHOLE study rather than ever reporting a clean result on partial/absent evidence.
"""
from dataclasses import replace
from src.serve.report import StudyReport, Finding
from src.serve.registry import resolve
from src.serve.diagnosis import det_to_findings, seg_to_finding, study_defer
from src.serve.stenosis_triage import triage_decision
from src.serve.temporal_vote import aggregate_sequence
from src.eval.audit import record


# --- box-format adapter between the serve-layer detector and temporal_vote.aggregate_sequence ----
#
# temporal_vote.py's own docstring describes its box as a "normalized YOLO box (cx, cy, w, h) in
# [0, 1]", but the serve-layer detector (see stenosis_triage._conf_of and the existing det_factory
# fixtures in tests/test_orchestrator.py) emits (x1, y1, x2, y2, conf) PIXEL-space corner tuples --
# the two do not match, exactly as flagged in the task brief. Reading iou_xywhn (the only place the
# box's numbers are actually read) shows it computes intersection/union straight from the box's own
# w * h with no reference to a frame size or a [0, 1] range: it is scale-invariant. We also have no
# frame dimensions to normalize against here (frames may not even be arrays -- see the injected-
# iterator tests), so the adapter below converts corner-form -> center-form IN THE SAME PIXEL UNITS
# the detector emits, which is exactly what the IoU linking needs and nothing more.

def _xyxy_to_cxcywh(box):
    """(x1, y1, x2, y2) -> (cx, cy, w, h), same units as the input (no [0, 1] normalization)."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1)


def _cxcywh_to_xyxy(box):
    """(cx, cy, w, h) -> (x1, y1, x2, y2); inverse of `_xyxy_to_cxcywh`."""
    cx, cy, w, h = box
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def _boxes_to_track_dets(boxes):
    """Adapt one frame's raw (x1, y1, x2, y2, conf) detector boxes into the
    ``{"box": (cx, cy, w, h), "conf": float}`` shape `aggregate_sequence` consumes."""
    return [{"box": _xyxy_to_cxcywh(b[:4]), "conf": float(b[4])} for b in boxes]


def _flatten_voted(voted):
    """Collapse `aggregate_sequence`'s per-frame stabilized output back into ONE detection per
    surviving track, as a (x1, y1, x2, y2, conf) tuple -- the shape `stenosis_triage.triage_decision`
    and `diagnosis.det_to_findings` both expect (`det_to_findings` does ``tuple(p)`` on each kept
    prediction, which on a dict would yield its KEYS, not its values, so a dict cannot be passed
    through as-is).

    `_densify` (temporal_vote.py) assigns ONE aggregated `conf` to every frame a surviving track
    spans, so grouping the flattened output by that conf value and keeping the first frame we see it
    on recovers exactly one row per track without having to re-derive track identity from the
    already-flattened per-frame output.
    """
    reps = {}
    for frame_dets in voted:
        for d in frame_dets:
            reps.setdefault(d["conf"], d)
    return [(*_cxcywh_to_xyxy(d["box"]), d["conf"]) for d in reps.values()]


class DiagnosticOrchestrator:
    """route(modality) -> resolve(model) -> infer -> typed findings -> StudyReport, deferring on any
    uncertainty. `model_factory(entry) -> callable(frame_gray) -> dict` is injected so tests supply
    fakes; this class never constructs a real model itself."""

    def __init__(self, router, registry, model_factory, cfg=None):
        self.router = router
        self.registry = registry
        self.model_factory = model_factory
        self.cfg = cfg or {}
        self._models = {}

    def _model_for(self, entry):
        """Lazily build (and cache) the model callable for a modality via the injected factory."""
        if entry.modality not in self._models:
            self._models[entry.modality] = self.model_factory(entry)
        return self._models[entry.modality]

    def _report(self, decision, findings, frames_analyzed, versions):
        deferred, reason = study_defer(decision, findings)
        return StudyReport(modality=decision.modality, view=decision.view,
                           quality_ok=decision.quality_ok, findings=findings,
                           deferred=deferred, defer_reason=reason,
                           frames_analyzed=frames_analyzed, model_versions=versions)

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
        dec = self.router.classify(frame_gray)
        versions = {"router": getattr(self.router, "weights", "router")}

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
            return StudyReport(modality=dec.modality, view=dec.view, quality_ok=dec.quality_ok,
                               findings=[], deferred=True, defer_reason="model-unavailable",
                               frames_analyzed=1, model_versions=versions)

        if entry.task == "det":
            findings = self._det_findings(entry, out)
        else:
            findings = [seg_to_finding(entry, out)]

        finding = findings[0]
        record(entry.model_path, frame_gray,
              {"modality": dec.modality, "task": entry.task, "finding": entry.finding_label,
               "confidence": finding.confidence, "deferred": finding.deferred,
               "reason": finding.reason})

        return self._report(dec, findings, 1, versions)

    def _iter_frames(self, path, stride, max_frames):
        """Yield grayscale frames sampled every `stride` frames, up to `max_frames`. cv2 is
        imported lazily here (not at module scope) so this file stays import-safe on a machine
        without opencv, matching the lazy-heavy-import convention the rest of the module follows."""
        import cv2
        cap = cv2.VideoCapture(path)
        try:
            i, kept = 0, 0
            while kept < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % stride == 0:
                    yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                    kept += 1
                i += 1
        finally:
            cap.release()

    def _det_findings_video(self, entry, model, frames, decisions, modality):
        """Detector path for a cine window.

        I1 -- run the detector ONLY on frames the router actually routed to `modality`: a
        router-deferred frame, or one routed to a different modality, must never reach THIS
        modality's detector. Excluded frames get an EMPTY slot rather than being dropped from the
        sequence -- `temporal_vote.link_tracks` reads its gap size straight off slot position, so
        compacting the list would silently shrink the true frame-to-frame distance and let two
        genuinely far-apart (and therefore unrelated) detections wrongly link into one "persistent"
        track.

        C2 -- carried over from `_det_findings`: any evaluated frame whose output is missing the
        'boxes' key entirely is a malformed output, not zero detections -- defer the whole finding
        rather than triage/vote over a silently-substituted empty list.

        C3 -- temporal voting may filter what is REPORTED as a box (`_flatten_voted(voted)`, the
        post-vote survivors), but it must never be the only thing standing between a low-confidence
        detection and a defer: a box that fails to persist across enough frames (dropped by
        `min_hits`) never reaches triage post-vote, silencing exactly the 'low-confidence' /
        'no-detection-uncertain' defers that same evidence would trigger through `analyze_frame`.
        So the defer decision is derived from the UNION of: triage over the PRE-vote evidence (every
        box the detector actually produced on an evaluated frame, before voting drops anything),
        triage over the post-vote survivors, and (C4) each evaluated frame's own 'deferred' flag
        (missing key fails safe to True). Only the post-vote survivors are used for the boxes a
        kept finding reports.
        """
        routed = [not d.deferred and d.modality == modality for d in decisions]
        outs = [model(f) if use else None for f, use in zip(frames, routed)]

        if any(o is not None and "boxes" not in o for o in outs):
            return [Finding(label=entry.finding_label, display_name=entry.finding_display,
                            confidence=0.0, deferred=True, reason="malformed-det", boxes=[])]

        seq = [_boxes_to_track_dets(o["boxes"]) if o is not None else [] for o in outs]
        voted = aggregate_sequence(seq, iou_thr=self.cfg.get("iou_thr", 0.3),
                                   min_hits=self.cfg.get("min_hits", 2),
                                   conf_agg=self.cfg.get("conf_agg", "mean"))

        temperature = self.cfg.get("temperature", 1.0)
        pre_vote_boxes = [b for o in outs if o is not None for b in o["boxes"]]
        pre_triage = triage_decision(pre_vote_boxes, temperature=temperature)
        post_triage = triage_decision(_flatten_voted(voted), temperature=temperature)
        model_deferred = any(o.get("deferred", True) for o in outs if o is not None)

        if pre_triage["deferred"] and not post_triage["deferred"]:
            deferred, reason = True, pre_triage["reason"]
        elif post_triage["deferred"]:
            deferred, reason = True, post_triage["reason"]
        elif model_deferred:
            deferred, reason = True, "low-confidence"
        else:
            deferred, reason = post_triage["deferred"], post_triage["reason"]

        triage = {"prediction": post_triage["prediction"],
                 "calibrated_confs": post_triage["calibrated_confs"],
                 "deferred": deferred, "reason": reason}
        return det_to_findings(entry, triage)

    def analyze_video(self, path, stride=5, max_frames=400):
        """Sample frames from a cine clip, route+screen the window the same way `analyze_frame`
        screens one frame, then fold per-frame detections across the window with temporal-vote
        aggregation so a one-frame flicker cannot become a finding while a persistent lesion
        survives even through a frame or two the per-frame detector missed.

        Safety default is DEFER, not guess: an undecodable clip / zero usable frames, a window
        where no sampled frame was confidently routed, and an unsupported modality all defer the
        WHOLE study -- never a clean report built on partial or absent evidence."""
        frames = list(self._iter_frames(path, stride, max_frames))
        versions = {"router": getattr(self.router, "weights", "router")}

        if not frames:
            # Nothing to screen -- an empty/undecodable clip must defer, never report clean.
            return StudyReport(modality="unknown", view=None, quality_ok=False, findings=[],
                               deferred=True, defer_reason="no-frames",
                               frames_analyzed=0, model_versions=versions)

        decisions = [self.router.classify(f) for f in frames]
        kept = [d for d in decisions if not d.deferred and d.modality not in ("unknown", "")]
        if not kept:
            # No frame in the window was confidently routed -- defer the whole study rather than
            # guess a modality from noise. Surface the first frame's own defer reason.
            return self._report(decisions[0], [], len(frames), versions)

        # Majority modality across the confidently-routed frames: a handful of ambiguous or
        # differently-routed frames inside an otherwise-consistent window shouldn't derail it.
        counts = {}
        for d in kept:
            counts[d.modality] = counts.get(d.modality, 0) + 1
        modality = max(counts, key=counts.get)
        dec = next(d for d in kept if d.modality == modality)

        entry = resolve(self.registry, modality)
        if entry is None:
            # Confidently-routed modality, nothing registered to screen it -- leave findings empty
            # and let study_defer's own "no-findings" guard defer it (see analyze_frame for why we
            # don't invent a second, conflicting "unsupported-modality" reason on top of that here).
            return self._report(dec, [], len(frames), versions)

        versions[entry.finding_label] = entry.model_path
        model = self._model_for(entry)

        # Representative frame for this modality: the highest-confidence confidently-routed frame.
        # Used by the seg path (seg runs on ONE frame, not a sequence) and for the audit log.
        candidate_idxs = [i for i, d in enumerate(decisions)
                          if d.modality == modality and not d.deferred]
        best_idx = max(candidate_idxs, key=lambda i: decisions[i].confidence)
        best_frame = frames[best_idx]

        try:
            if entry.task == "det":
                findings = self._det_findings_video(entry, model, frames, decisions, modality)
            else:
                findings = [seg_to_finding(entry, model(best_frame))]
        except ModelUnavailable:
            # Same fail-safe as analyze_frame: weights missing/unloadable, or an unrecognized task
            # type -- defer the WHOLE study rather than ever report on partial/absent evidence.
            return StudyReport(modality=dec.modality, view=dec.view, quality_ok=dec.quality_ok,
                               findings=[], deferred=True, defer_reason="model-unavailable",
                               frames_analyzed=len(frames), model_versions=versions)

        finding = findings[0]
        record(entry.model_path, best_frame,
              {"modality": dec.modality, "task": entry.task, "finding": entry.finding_label,
               "confidence": finding.confidence, "deferred": finding.deferred,
               "reason": finding.reason, "frames_analyzed": len(frames)})

        return self._report(dec, findings, len(frames), versions)


# --- real model_factory: the seam between tested pure orchestration and heavy inference code -----


class ModelUnavailable(Exception):
    """Raised by a model callable -- never by DiagnosticOrchestrator itself -- when the underlying
    task model could not be produced: a missing weights file, an unloadable/corrupt weights file, a
    missing heavy dependency (coremltools/ultralytics/torch), or an unrecognized `entry.task`.
    `analyze_frame`/`analyze_video` catch this at the single point a model is actually invoked and
    convert it into a deferred study -- never a crash, never a silently empty confident result."""


def _load_det(model_path):
    """Build a real detection-model callable backed by `src.serve.infer.DetModel` (YOLO exported to
    CoreML). `coremltools` is only imported transitively, inside `DetModel.__init__` -- which itself
    only runs when THIS function is called, never at module import time -- so importing
    `src.serve.orchestrator` stays torch/coreml-free.

    If the weights can't be loaded (file missing, corrupt/unloadable, or coremltools itself missing)
    the constructor's exception is caught HERE, at build time, and turned into a callable that always
    raises `ModelUnavailable` when invoked. That keeps the failure scoped to this one modality (a bad
    weights file downgrades it to 'this modality always defers') instead of letting it propagate out
    of `build_orchestrator` -- or, worse, out of `analyze_frame`/`analyze_video` -- as a live crash.
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


def _model_factory(entry):
    """The real `model_factory(entry) -> callable(frame_gray) -> dict` that `build_orchestrator`
    wires in. `entry.task` selects `DetModel` vs `SegModel` (via `_load_det`/`_load_seg`, looked up
    by module-level name so tests can monkeypatch either loader without needing real weights). Any
    other `entry.task` value -- a registry typo, a future task type nothing here understands yet --
    is NOT a crash: it returns a callable that raises `ModelUnavailable` on first use, exactly like a
    missing/unloadable weights file, so `analyze_frame`/`analyze_video` defer the study the same way
    for both failure modes."""
    if entry.task == "det":
        return _load_det(entry.model_path)
    if entry.task == "seg":
        return _load_seg(entry.model_path)

    def _unknown_task(frame, _entry=entry):
        raise ModelUnavailable(f"unknown task type {_entry.task!r} for modality {_entry.modality!r}")
    return _unknown_task


def build_orchestrator(cfg_path):
    """Wire a real `DiagnosticOrchestrator` from a YAML config: a real `ModalityRouter` (edge
    classifier weights + labels + thresholds) and the real `_model_factory` (real `DetModel`/
    `SegModel` per registry entry, fail-safe on a bad weights file -- see `ModelUnavailable`).
    `yaml`/`ModalityRouter`/`load_registry` are all imported here, inside the function body, not at
    module scope: calling `build_orchestrator` is the one place this module actually touches the
    heavy stack, and even then only through lazy imports several calls deep (`ModalityRouter` itself
    doesn't load torch until `.classify()` runs; `_model_factory` doesn't load coremltools until a
    model is actually invoked) -- importing the module itself never does.

    Config shape (see tests/test_orchestrator.py for a minimal example):
        router: {weights: ..., labels: [...], thresholds: {keep_thr, margin, quality_thr}}
        modalities: {<modality>: {task, model_path, display_name, finding_label, finding_display,
                                   floor_ok}, ...}
        runtime: {temperature, iou_thr, min_hits, conf_agg}   # optional, passed through as cfg
    """
    import yaml
    from src.serve.router import ModalityRouter
    from src.serve.registry import load_registry

    cfg = yaml.safe_load(open(cfg_path)) or {}
    rc = cfg["router"]
    router = ModalityRouter(rc["weights"], rc["labels"], rc.get("thresholds"))
    registry = load_registry(cfg_path)
    return DiagnosticOrchestrator(router, registry, _model_factory, cfg=cfg.get("runtime", {}))
