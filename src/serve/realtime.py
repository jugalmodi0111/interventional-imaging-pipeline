"""Real-time overlay loop (Mac). Reads a video/image-dir/camera, runs a CoreML edge model
per frame, overlays mask/boxes, marks deferred frames, logs the audit trail.

    python -m src.serve.realtime --model runs/coronary/student.mlpackage --task seg --source clip.mp4
    python -m src.serve.realtime --model runs/stenosis/best.mlpackage   --task det --source frames/
"""
import argparse, glob, os
import numpy as np


def _frames(source):
    import cv2
    if os.path.isdir(source):
        for p in sorted(glob.glob(os.path.join(source, "*"))):
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if g is not None:
                yield g
    else:
        cap = cv2.VideoCapture(0 if source == "camera" else source)
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        cap.release()


def _overlay_seg(gray, res):
    import cv2
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    m = cv2.resize(res["mask"], (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    base[m == 1] = (0.4 * base[m == 1] + np.array([0, 0, 160])).astype(np.uint8)
    return base


def _overlay_det(gray, res, size):
    import cv2
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    sx, sy = gray.shape[1] / size, gray.shape[0] / size
    for x1, y1, x2, y2, s in res["boxes"]:
        cv2.rectangle(base, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), (0, 200, 0), 2)
        cv2.putText(base, f"{s:.2f}", (int(x1 * sx), int(y1 * sy) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    return base


def run(a):
    import cv2
    from src.serve.infer import SegModel, DetModel
    model = SegModel(a.model, size=a.size) if a.task == "seg" else DetModel(a.model, size=a.size)
    writer, n, deferred = None, 0, 0
    for gray in _frames(a.source):
        res = model(gray)
        deferred += int(res["deferred"])
        vis = _overlay_seg(gray, res) if a.task == "seg" else _overlay_det(gray, res, a.size)
        if res["deferred"]:
            cv2.putText(vis, "DEFER -> human", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if a.out:
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
            writer.write(vis)
        if a.show:
            cv2.imshow("interventional", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        n += 1
    if writer:
        writer.release()
    if a.show:
        cv2.destroyAllWindows()
    print(f"frames {n} | deferred {deferred} ({deferred / max(1, n):.1%}) | audit -> runs/audit.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", choices=["seg", "det"], required=True)
    ap.add_argument("--source", required=True, help="video file | image dir | 'camera'")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", help="write overlay mp4")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--show", action="store_true")
    run(ap.parse_args())
