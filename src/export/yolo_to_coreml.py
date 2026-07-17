"""YOLO -> CoreML (Mac side). Ultralytics has first-class CoreML export with NMS baked in.

One call; run on macOS. Unlike the seg student, no manual trace/convert is needed.
"""
import argparse
import os


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


def _check_coreml_output(path):
    """Pure, stdlib-only sanity check on an export() output path -> (ok, msg).

    No torch/ultralytics/coremltools needed: a CoreML export is either a `.mlpackage`
    directory or a legacy `.mlmodel` file, and "did it actually write something non-empty"
    is answerable from os.path/os.walk alone.
    """
    if not path or not os.path.exists(path):
        return False, f"no output at {path!r}"

    if os.path.isdir(path):
        kind = "mlpackage" if path.endswith(".mlpackage") else "directory"
        total_bytes = 0
        for root, _dirs, files in os.walk(path):
            for f in files:
                total_bytes += os.path.getsize(os.path.join(root, f))
    else:
        kind = "mlmodel" if path.endswith(".mlmodel") else "file"
        total_bytes = os.path.getsize(path)

    mb = total_bytes / (1024 * 1024)
    msg = f"{kind} at {path} ({mb:.2f} MB)"
    if total_bytes == 0:
        msg += " -- WARNING: size is 0 bytes"
    return True, msg


def smoketest(weights, imgsz=None, int8=False):
    """Export + sanity-check the output, without loading the result back into CoreML.

    Meant as an early, cheap gate before spending GPU time on a bigger detector (e.g. YOLO11m
    or imgsz=1024): confirms the ultralytics -> CoreML export path still runs cleanly and
    produces a non-empty .mlpackage/.mlmodel, catching unsupported-layer or size surprises
    up front. Heavy deps (ultralytics/coremltools) are still only imported lazily, inside
    export().
    """
    path = export(weights, imgsz=imgsz, nms=True, int8=int8)
    ok, msg = _check_coreml_output(path)
    status = "PASS" if ok else "FAIL"
    print(f"CoreML smoke-test {status}: {msg}")
    return ok, path, msg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="runs/stenosis/**/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=None, help="default: read training imgsz from the checkpoint")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--smoketest", action="store_true", help="export + sanity-check the output, don't just export")
    a = ap.parse_args()
    if a.smoketest:
        smoketest(a.weights, imgsz=a.imgsz, int8=a.int8)
    else:
        export(a.weights, imgsz=a.imgsz, int8=a.int8)
