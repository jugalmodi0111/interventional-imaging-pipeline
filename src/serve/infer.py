"""On-device inference wrappers for the exported CoreML edge models (run on the Mac).

Per-frame: CLAHE preprocess -> CoreML predict -> postprocess -> abstention flag -> audit log.
'Wrong but confident' is the dangerous mode, so every result carries a `deferred` flag and every
call is logged to the audit trail.
"""
import os
import numpy as np
from src.data_prep.preprocess import clahe_unsharp
from src.eval.audit import record


class _CoreMLBase:
    def __init__(self, mlpackage, model_version=None, size=512):
        import coremltools as ct
        self.model = ct.models.MLModel(mlpackage)
        self.input_name = self.model.get_spec().description.input[0].name
        self.size = size
        self.version = model_version or os.path.basename(mlpackage)

    def _prep(self, frame_gray):
        import cv2
        x = clahe_unsharp(frame_gray)
        x = cv2.resize(x, (self.size, self.size)).astype(np.float32) / 255.0
        return x


class SegModel(_CoreMLBase):
    """Binary vessel segmentation. Returns dict(mask, foreground_prob, deferred)."""
    def __init__(self, mlpackage, model_version=None, size=512, defer_below=0.55, audit=True):
        super().__init__(mlpackage, model_version, size)
        self.defer_below, self.audit = defer_below, audit

    def __call__(self, frame_gray):
        x = self._prep(frame_gray)
        out = self.model.predict({self.input_name: x[None, None]})
        logits = np.asarray(list(out.values())[0]).squeeze()
        prob = 1.0 / (1.0 + np.exp(-logits))
        mask = (prob >= 0.5).astype(np.uint8)
        conf = float(prob[mask == 1].mean()) if mask.any() else 0.0     # mean fg confidence
        deferred = conf < self.defer_below
        res = {"mask": mask, "foreground_prob": prob, "confidence": conf, "deferred": deferred}
        if self.audit:
            record(self.version, x, {"task": "coronary_seg", "confidence": round(conf, 3),
                                     "deferred": deferred})
        return res


class DetModel(_CoreMLBase):
    """YOLO stenosis/catheter detection exported to CoreML (NMS baked in).
    Returns dict(boxes=[(x1,y1,x2,y2,conf)], deferred)."""
    def __init__(self, mlpackage, model_version=None, size=640, conf=0.25, defer_below=0.4, audit=True):
        super().__init__(mlpackage, model_version, size)
        self.conf, self.defer_below, self.audit = conf, defer_below, audit

    def __call__(self, frame_gray):
        x = self._prep(frame_gray)
        rgb = np.repeat(x[None, :, :, None], 3, axis=3)                 # CoreML YOLO expects 3ch
        out = self.model.predict({self.input_name: rgb})
        boxes = _parse_yolo_coreml(out, self.conf, self.size)
        top = max((b[4] for b in boxes), default=0.0)
        deferred = top < self.defer_below
        res = {"boxes": boxes, "top_conf": top, "deferred": deferred}
        if self.audit:
            record(self.version, x, {"task": "detection", "n": len(boxes),
                                     "top_conf": round(top, 3), "deferred": deferred})
        return res


def _parse_yolo_coreml(out, conf_th, size):
    """Ultralytics CoreML NMS export yields 'confidence'[N,C] + 'coordinates'[N,4] (cx,cy,w,h norm)."""
    vals = {k: np.asarray(v) for k, v in out.items()}
    confs = vals.get("confidence"); coords = vals.get("coordinates")
    if confs is None or coords is None:
        return []                                                      # raw-tensor export: add decoder
    boxes = []
    for c, (cx, cy, w, h) in zip(confs, coords):
        s = float(c.max())
        if s < conf_th:
            continue
        x1, y1 = (cx - w / 2) * size, (cy - h / 2) * size
        x2, y2 = (cx + w / 2) * size, (cy + h / 2) * size
        boxes.append((x1, y1, x2, y2, s))
    return boxes
