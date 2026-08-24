"""Hosted torch inference for Model One (B8: central serving; this is NOT the CoreML edge path).

Loads the trainer's head.pt (backbone name + head weights + temperature + threshold + defer band)
and answers one de-identified still frame at a time. B3 posture is enforced here, closest to the
model: a calibrated probability inside the defer band is returned deferred -- downstream layers
may defer MORE, never less. Construction failures raise; src.serve.orchestrator._load_cls turns
them into ModelUnavailable so one bad checkpoint defers studies instead of crashing the service.
"""
import numpy as np


class ClsModel:
    def __init__(self, path):
        import torch
        from src.models.frozen_backbone import FrozenBackboneClassifier
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        self._torch = torch
        self.imgsz = int(ckpt["imgsz"])
        self.temperature = float(ckpt["temperature"])
        self.threshold = float(ckpt["threshold"])
        self.defer_band = tuple(ckpt["defer_band"])
        self.model = FrozenBackboneClassifier(ckpt["backbone"], imgsz=self.imgsz)
        self.model.head.load_state_dict(ckpt["head_state"])
        self.model.eval()

    def _prep(self, frame_gray):
        import cv2
        img = cv2.resize(np.asarray(frame_gray), (self.imgsz, self.imgsz),
                         interpolation=cv2.INTER_AREA).astype("float32") / 255.0
        return self._torch.from_numpy(img[None, None])

    def __call__(self, frame_gray):
        with self._torch.no_grad():
            logit = float(self.model(self._prep(frame_gray))[0])
        prob = float(1.0 / (1.0 + np.exp(-logit / self.temperature)))
        lo, hi = self.defer_band
        deferred = lo <= prob <= hi
        return {"prob": prob, "confidence": float(max(prob, 1.0 - prob)),
                "deferred": bool(deferred), "reason": "defer-band" if deferred else "confident",
                "threshold": self.threshold}
