"""Minimal local inference service (FastAPI) wrapping a CoreML edge model.

For the network-API topology (another app POSTs a frame). Air-gapped cath labs should bind to
localhost only. Model One (B3) screens a SINGLE STILL FRAME: there is no video path (deleted per
the 2026-08-03 audit, P3) — `/analyze?kind=video` is a deliberate 400 refusal.

    uvicorn src.serve.app:app --host 127.0.0.1 --port 8000
    MODEL=outputs/coronary_student_clgeodice/student.mlpackage TASK=seg uvicorn src.serve.app:app
"""
import os
import numpy as np

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
except Exception:                                         # keep import-safe without fastapi
    FastAPI = None

# Default seg weights: the gate-passed CLGeoDice student run (the old default,
# runs/coronary/student.mlpackage, no longer exists on disk).
MODEL_PATH = os.environ.get("MODEL", "outputs/coronary_student_clgeodice/student.mlpackage")
TASK = os.environ.get("TASK", "seg")
_model = None


def _get_model():
    global _model
    if _model is None:
        from src.serve.infer import SegModel, DetModel
        _model = SegModel(MODEL_PATH) if TASK == "seg" else DetModel(MODEL_PATH)
    return _model


def _read_gray(raw):
    import cv2
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


if FastAPI is not None:
    app = FastAPI(title="Interventional edge inference")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": os.path.basename(MODEL_PATH), "task": TASK}

    @app.post("/infer")
    async def infer(file: UploadFile = File(...)):
        res = _get_model()(_read_gray(await file.read()))
        if TASK == "seg":
            return {"deferred": res["deferred"], "confidence": res["confidence"],
                    "vessel_pixels": int(res["mask"].sum())}
        return {"deferred": res["deferred"], "top_conf": res["top_conf"],
                "boxes": [[round(v, 2) for v in b] for b in res["boxes"]]}

    # --- /analyze: diagnostic-orchestrator-backed endpoint ---------------------------------------
    #
    # Reachability layer for `DiagnosticOrchestrator` (src/serve/orchestrator.py): one route, one
    # single still frame in, always a JSON `StudyReport` out. `build_orchestrator` is imported here
    # (module scope of this `if` block, not top-of-file) and only actually invoked lazily inside
    # `_get_orch` -- it in turn only touches torch/ultralytics/coremltools several calls deeper,
    # still lazily -- so this module stays import-safe with no heavy deps installed (see
    # test_router.py / test_orchestrator.py's subprocess import-safety guardrails for the pattern
    # this follows).
    #
    # Safety default is DEFER, not guess (same posture the orchestrator itself takes): a corrupt or
    # undecodable upload, or a model/config that fails to load, resolves to a deferred `StudyReport`;
    # an unsupported `kind` (video included -- Model One has no video path) is a clean 400 contract
    # refusal -- never an unhandled 500 that hides the outcome, and never a confident result built
    # on bytes we couldn't actually read.
    from src.serve.orchestrator import build_orchestrator
    from src.serve.report import StudyReport

    ORCH_CONFIG = os.environ.get("ORCH_CONFIG", "configs/orchestrator.yaml")
    _orch = None

    def _get_orch():
        """Lazily build (and cache) the real orchestrator singleton. Left as a module-level function
        (rather than inlined) so tests can monkeypatch it directly to simulate a build failure
        (missing/broken `configs/orchestrator.yaml`) without needing a real router/registry."""
        global _orch
        if _orch is None:
            _orch = build_orchestrator(ORCH_CONFIG)
        return _orch

    def _decode_image(raw):
        """Raw upload bytes -> grayscale ndarray, or None if the bytes aren't a decodable image.
        `cv2.imdecode` fails by returning None rather than raising -- callers must check for it."""
        import cv2
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)

    def _deferred_report(reason):
        """A study-level DEFER with no modality/findings yet resolved -- used when we can't even get
        as far as routing (undecodable upload, model/config unavailable). Always JSON-shaped like a
        real `StudyReport` so callers have exactly one response contract to parse."""
        return StudyReport(modality="unknown", view=None, quality_ok=False, findings=[],
                           deferred=True, defer_reason=reason, frames_analyzed=0,
                           model_versions={}).to_dict()

    @app.get("/events")
    async def events(replay: int = 50, max_events: int = 0, timeout: float = 0.0):
        """Live `text/event-stream` mirror of the orchestrator's event bus: replays the last
        `replay` buffered events, then streams new ones as they happen. `max_events`/`timeout`
        bound the stream (0 = unbounded) -- handy for curl and deterministic tests:

            curl -N 'localhost:8000/events'                       # watch live
            curl -N 'localhost:8000/events?max_events=20'         # first 20 then close

        Observe-only by construction: this endpoint subscribes like any other observer and cannot
        influence a verdict. Events carry hashes/ids/reasons, never pixels (see events.py)."""
        import asyncio
        import json as _json
        import queue as _queue
        import time as _time
        from fastapi.responses import StreamingResponse

        try:
            orch = _orch if _orch is not None else _get_orch()
            bus = getattr(orch, "bus", None)
        except Exception:
            bus = None
        if bus is None:
            raise HTTPException(status_code=503,
                                detail="event bus unavailable (orchestrator failed to build)")

        q = _queue.Queue()
        unsubscribe = bus.subscribe("*", q.put)
        replayed = bus.ring.snapshot()[-replay:] if replay > 0 else []

        async def gen():
            sent = 0
            deadline = _time.monotonic() + timeout if timeout > 0 else None
            try:
                for e in replayed:
                    yield f"data: {_json.dumps(e, default=str)}\n\n"
                    sent += 1
                    if max_events and sent >= max_events:
                        return
                while deadline is None or _time.monotonic() < deadline:
                    try:
                        e = await asyncio.to_thread(q.get, True, 0.25)
                    except _queue.Empty:
                        continue
                    yield f"data: {_json.dumps(e, default=str)}\n\n"
                    sent += 1
                    if max_events and sent >= max_events:
                        return
            finally:
                unsubscribe()

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/analyze")
    async def analyze(file: UploadFile = File(...), kind: str = "image"):
        """POST a single still frame (`?kind=image`, the default) and get back a `StudyReport`
        JSON body -- possible finding + clinician-review flag, never an autonomous diagnosis claim.
        `kind=video` is a deliberate 400 contract refusal: Model One (B3) screens a single still
        frame; the cine/video path was deleted (2026-08-03 audit, P3). Never a raw 500: every other
        failure mode (bad `kind`, corrupt/undecodable upload, an orchestrator/model that fails to
        load, or any other unexpected error) resolves to either a clean 4xx or a 200 carrying a
        deferred report."""
        if kind == "video":
            # Contract refusal, not a defer: nothing about a video request is analyzable here, and
            # a deferred 200 would wrongly imply a clip was screened. Refused before the
            # orchestrator is even built -- this is a property of the endpoint, not of a healthy
            # model stack.
            raise HTTPException(
                status_code=400,
                detail="video analysis is not part of Model One; submit a single frame")
        if kind != "image":
            raise HTTPException(status_code=400,
                                detail=f"unsupported kind={kind!r}; expected 'image'")

        raw = await file.read()

        try:
            orch = _orch if _orch is not None else _get_orch()
        except Exception:
            # Router/registry config missing or unloadable -- defer rather than crash the request.
            return _deferred_report("model-unavailable")

        frame = _decode_image(raw)
        if frame is None:
            # Corrupt/undecodable image bytes: cv2.imdecode returns None instead of raising, so this
            # must be checked explicitly -- never hand None to the orchestrator as if it were real
            # pixel data.
            return _deferred_report("undecodable-image")
        try:
            report = orch.analyze_frame(frame)
        except Exception:
            return _deferred_report("analysis-error")
        return report.to_dict()
else:
    app = None   # install fastapi + uvicorn to serve
