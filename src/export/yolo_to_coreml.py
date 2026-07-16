"""YOLO -> CoreML (Mac side). Ultralytics has first-class CoreML export with NMS baked in.

One call; run on macOS. Unlike the seg student, no manual trace/convert is needed.
"""
import argparse


def _ckpt_imgsz(model):
    """Read the training imgsz Ultralytics stored in the checkpoint (train_args -> model.args)."""
    ckpt = getattr(model, "ckpt", None) or {}
    ta = ckpt.get("train_args") or {}
    if ta.get("imgsz"):
        return int(ta["imgsz"])
    margs = getattr(getattr(model, "model", None), "args", None)
    if isinstance(margs, dict) and margs.get("imgsz"):
        return int(margs["imgsz"])
    return None


def export(weights, imgsz=None, nms=True, int8=False):
    from ultralytics import YOLO
    model = YOLO(weights)
    if imgsz is None:                                  # match training res (stenosis trains @768, not the 640 default)
        imgsz = _ckpt_imgsz(model)
        if imgsz is None:
            imgsz = 640
            print("WARNING: could not read train imgsz from checkpoint; defaulting to 640 — pass --imgsz to be safe")
        else:
            print(f"auto-detected train imgsz={imgsz} from checkpoint")
    out = model.export(format="coreml", nms=nms, imgsz=imgsz, int8=int8)
    print("wrote", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="runs/stenosis/**/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=None, help="default: read training imgsz from the checkpoint")
    ap.add_argument("--int8", action="store_true")
    a = ap.parse_args()
    export(a.weights, imgsz=a.imgsz, int8=a.int8)
