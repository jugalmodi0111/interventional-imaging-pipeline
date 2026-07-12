"""Catheter/guidewire tracking = per-frame detection + ByteTrack. ByteTrack runs as Python on top
of the detector (it is NOT inside the CoreML/ONNX graph), so it composes with either backend.

- GPU/dev:   Ultralytics YOLO.track(..., tracker='bytetrack.yaml') on a .pt model (fast to iterate).
- Edge/Mac:  detect per frame with the CoreML DetModel, feed boxes to a lightweight ByteTrack.

Reports fps + ID metrics (what matters for guidewire continuity) and logs the audit trail.

Two eval modes:
  track_yolo  — one source (mp4 OR a list of frames) = ONE continuous sequence.
  track_clips — a frame DIR that concatenates many clips (e.g. CathAction img/, 149 clips):
                split by filename prefix, track each clip with a FRESH tracker, aggregate.
                Unique-ID count over the raw concatenation is invalid (every clip cut mints new
                IDs); per-clip is the only honest signal.
"""
import argparse, os, re
from collections import defaultdict

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def group_frames_by_clip(frame_dir):
    """Return {prefix: [sorted frame paths]} by stripping the trailing frame number from each stem.
    Matches src.data_prep.verify_sequence so clip boundaries agree across tools."""
    clips = defaultdict(list)
    for f in os.listdir(frame_dir):
        if f.lower().endswith(IMG_EXTS):
            stem = os.path.splitext(f)[0]
            m = re.match(r"^(.*?)(\d+)$", stem)
            clips[m.group(1) if m else stem].append(os.path.join(frame_dir, f))
    for k in clips:
        clips[k].sort()
    return dict(clips)


def track_yolo(weights, source, out=None, conf=0.25, tracker="bytetrack.yaml", device=0, audit=True):
    """Track ONE continuous sequence. `source` = mp4 path, image dir, or list of frame paths.
    Returns dict(frames, ids, max_tracks, fps). fps = pure model+track wall-clock (no video write)."""
    from ultralytics import YOLO
    from src.eval.audit import record
    model = YOLO(weights)
    ids, n, max_tracks, infer_s = set(), 0, 0, 0.0
    writer = None
    for res in model.track(source=source, conf=conf, tracker=tracker, stream=True,
                           persist=True, verbose=False, device=device):
        infer_s += (res.speed.get("preprocess", 0) + res.speed.get("inference", 0)
                    + res.speed.get("postprocess", 0)) / 1000.0
        n += 1
        cur = 0 if res.boxes is None else len(res.boxes)
        max_tracks = max(max_tracks, cur)
        if res.boxes is not None and res.boxes.id is not None:
            ids.update(int(i) for i in res.boxes.id.tolist())
        if audit:
            record(os.path.basename(weights), res.orig_img[..., 0],
                   {"task": "catheter_track", "tracks": cur})
        if out:
            import cv2
            vis = res.plot()
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
            writer.write(vis)
    if writer:
        writer.release()
    fps = n / infer_s if infer_s > 0 else 0.0
    return {"frames": n, "ids": len(ids), "max_tracks": max_tracks, "fps": fps}


def track_clips(weights, frame_dir, conf=0.25, tracker="bytetrack.yaml", device=0,
                out_json="runs/catheter/track_clips.json"):
    """Split a concatenated frame dir into clips (by filename prefix) and track each separately
    with a fresh tracker. The honest metrics:
      fragmentation = unique_ids - max_simultaneous_tracks per clip
        (0 = every instrument kept ONE ID for the whole clip; >0 = tracks were dropped/re-acquired).
      fps = frames / (model+track time), averaged over clips.
    Writes a per-clip JSON report. Returns the aggregate dict."""
    import json
    clips = group_frames_by_clip(frame_dir)
    if not clips:
        raise SystemExit(f"no image frames under {frame_dir!r}")
    rows, tot_frames, tot_frag, fps_sum = [], 0, 0, 0.0
    for prefix, frames in sorted(clips.items()):
        r = track_yolo(weights, frames, out=None, conf=conf, tracker=tracker,
                       device=device, audit=False)     # skip audit: 5k rows/clip drowns the trail
        frag = max(0, r["ids"] - r["max_tracks"])
        r.update(clip=prefix, fragmentation=frag)
        rows.append(r)
        tot_frames += r["frames"]; tot_frag += frag; fps_sum += r["fps"]
    agg = {"clips": len(rows), "frames": tot_frames,
           "total_fragmentation": tot_frag,
           "frag_per_clip": round(tot_frag / len(rows), 3),
           "mean_fps": round(fps_sum / len(rows), 1),
           "per_clip": sorted(rows, key=lambda x: -x["fragmentation"])}
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(agg, f, indent=2)
    worst = agg["per_clip"][0] if agg["per_clip"] else {}
    print(f"clips {agg['clips']} | frames {agg['frames']} | mean fps {agg['mean_fps']} | "
          f"total fragmentation {tot_frag} ({agg['frag_per_clip']}/clip) | "
          f"worst clip {worst.get('clip','-')} frag={worst.get('fragmentation',0)} "
          f"(ids={worst.get('ids',0)}, max_tracks={worst.get('max_tracks',0)}) | report -> {out_json}")
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="catheter detector .pt (dev)")
    ap.add_argument("--source", required=True, help="video | image dir")
    ap.add_argument("--out", help="overlay mp4 (single-sequence mode only)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=0, help="0 = GPU (default), 'cpu' to override")
    ap.add_argument("--per-clip", action="store_true",
                    help="treat --source as a dir of MANY concatenated clips; split by filename "
                         "prefix and report per-clip fps + fragmentation (the valid metric)")
    a = ap.parse_args()
    if a.per_clip:
        track_clips(a.weights, a.source, conf=a.conf, device=a.device)
    else:
        r = track_yolo(a.weights, a.source, out=a.out, conf=a.conf, device=a.device)
        print(f"frames {r['frames']} | unique track ids {r['ids']} | max simultaneous {r['max_tracks']} "
              f"| {r['fps']:.1f} fps | audit -> runs/audit.jsonl")
