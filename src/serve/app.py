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
    from fastapi import FastAPI, UploadFile, File
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
else:
    app = None   # install fastapi + uvicorn to serve
