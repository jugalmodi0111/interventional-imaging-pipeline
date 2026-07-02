"""YOLO -> CoreML (Mac side). Ultralytics has first-class CoreML export with NMS baked in.

One call; run on macOS. Unlike the seg student, no manual trace/convert is needed.
"""
import argparse


def export(weights, imgsz=640, nms=True, int8=False):
    from ultralytics import YOLO
    out = YOLO(weights).export(format="coreml", nms=nms, imgsz=imgsz, int8=int8)
    print("wrote", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="runs/stenosis/**/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--int8", action="store_true")
    a = ap.parse_args()
    export(a.weights, imgsz=a.imgsz, int8=a.int8)
