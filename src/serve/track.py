"""Catheter/guidewire tracking = per-frame detection + ByteTrack. ByteTrack runs as Python on top
of the detector (it is NOT inside the CoreML/ONNX graph), so it composes with either backend.

- GPU/dev:   Ultralytics YOLO.track(..., tracker='bytetrack.yaml') on a .pt model (fast to iterate).
- Edge/Mac:  detect per frame with the CoreML DetModel, feed boxes to a lightweight ByteTrack.

Reports fps + ID switches (the metric that matters for guidewire continuity) and logs the audit trail.
"""
import argparse, os


def track_yolo(weights, source, out=None, conf=0.25, tracker="bytetrack.yaml"):
    """Dev-path tracking with Ultralytics' built-in ByteTrack. Returns (n_frames, n_ids)."""
    from ultralytics import YOLO
    from src.eval.audit import record
    model = YOLO(weights)
    ids, n = set(), 0
    writer = None
    for res in model.track(source=source, conf=conf, tracker=tracker, stream=True, persist=True, verbose=False):
        n += 1
        if res.boxes is not None and res.boxes.id is not None:
            ids.update(int(i) for i in res.boxes.id.tolist())
        record(os.path.basename(weights), res.orig_img[..., 0],
               {"task": "catheter_track", "tracks": 0 if res.boxes is None else len(res.boxes)})
        if out:
            import cv2
            vis = res.plot()
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
            writer.write(vis)
    if writer:
        writer.release()
    print(f"frames {n} | unique track ids {len(ids)} | audit -> runs/audit.jsonl")
    return n, len(ids)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="catheter detector .pt (dev) ")
    ap.add_argument("--source", required=True, help="video | image dir")
    ap.add_argument("--out"); ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    track_yolo(a.weights, a.source, out=a.out, conf=a.conf)
