"""Minimal local inference service (FastAPI) wrapping a CoreML edge model.

For the network-API topology (another app POSTs a frame). For real-time per-frame overlay use
`realtime.py` in-process instead — an HTTP hop per frame won't keep up. Air-gapped cath labs
should bind to localhost only.

    uvicorn src.serve.app:app --host 127.0.0.1 --port 8000
    MODEL=runs/coronary/student.mlpackage TASK=seg uvicorn src.serve.app:app
"""
import os
import numpy as np

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
except Exception:                                         # keep import-safe without fastapi
    FastAPI = None

MODEL_PATH = os.environ.get("MODEL", "runs/coronary/student.mlpackage")
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
    # Reachability layer for `DiagnosticOrchestrator` (src/serve/orchestrator.py): one route, image
    # or video, always a JSON `StudyReport`. `build_orchestrator` is imported here (module scope of
    # this `if` block, not top-of-file) and only actually invoked lazily inside `_get_orch` -- it in
    # turn only touches torch/ultralytics/coremltools several calls deeper, still lazily -- so this
    # module stays import-safe with no heavy deps installed (see test_router.py /
    # test_orchestrator.py's subprocess import-safety guardrails for the pattern this follows).
    #
    # Safety default is DEFER, not guess (same posture the orchestrator itself takes): a corrupt or
    # undecodable upload, an unsupported `kind`, or a model/config that fails to load all resolve to
    # either a clean 4xx or a deferred `StudyReport` -- never an unhandled 500 that hides the outcome,
    # and never a confident result built on bytes we couldn't actually read.
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

    @app.post("/analyze")
    async def analyze(file: UploadFile = File(...), kind: str = "image"):
        """POST an image frame or a cine clip (`?kind=image|video`, default image) and get back a
        `StudyReport` JSON body -- possible finding + clinician-review flag, never an autonomous
        diagnosis claim. Never a raw 500: every failure mode (bad `kind`, corrupt/undecodable upload,
        an orchestrator/model that fails to load, or any other unexpected error) resolves to either a
        clean 4xx or a 200 carrying a deferred report."""
        if kind not in ("image", "video"):
            raise HTTPException(status_code=400,
                                detail=f"unsupported kind={kind!r}; expected 'image' or 'video'")

        raw = await file.read()

        try:
            orch = _orch if _orch is not None else _get_orch()
        except Exception:
            # Router/registry config missing or unloadable -- defer rather than crash the request.
            return _deferred_report("model-unavailable")

        if kind == "video":
            import tempfile
            suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
            fd, path = tempfile.mkstemp(suffix=suffix)
            try:
                try:
                    with os.fdopen(fd, "wb") as f:
                        f.write(raw)
                    report = orch.analyze_video(path)
                except Exception:
                    # analyze_video already defers on every documented failure (undecodable clip,
                    # zero usable frames, missing weights); this also nets a failure writing the
                    # tempfile itself, so a genuine unanticipated bug still can't take the endpoint
                    # down as a raw 500.
                    return _deferred_report("analysis-error")
            finally:
                os.remove(path)
            return report.to_dict()

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
