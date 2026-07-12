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
import argparse, os, re, time
from collections import defaultdict

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# --- pure, ultralytics-free metric helpers (unit-testable) --------------------------

def concurrent_tracks(box_ids):
    """Number of ASSIGNED track IDs in one frame = count of non-None entries in `box_ids`.

    `box_ids` is None when the tracker produced no output, otherwise an iterable of ids where
    untracked detections are None (e.g. res.boxes.id.tolist()). Counting ONLY assigned IDs -- not
    raw detections (len(res.boxes)) -- stops spurious/untracked detections from inflating the
    concurrent-track count, which is what let the old fragmentation metric read a false-good."""
    if box_ids is None:
        return 0
    return sum(1 for i in box_ids if i is not None)


def fragmentation(total_unique_ids, max_concurrent):
    """ID fragmentation over a clip = unique_ids - max simultaneous tracks, floored at 0.
    0 = every instrument kept ONE id the whole clip; >0 = a track was dropped and re-acquired
    under a new id somewhere. Honest only when max_concurrent counts assigned IDs (see above)."""
    return max(0, total_unique_ids - max_concurrent)


def weighted_fps(rows):
    """Frame-weighted throughput across clips = total frames / total time.
    Each row is a mapping carrying 'frames' and a time in seconds ('time_s', falling back to
    'track_s'). An unweighted mean of per-clip fps biases toward tiny clips, so weight by frames."""
    tot_frames, tot_time = 0.0, 0.0
    for r in rows:
        tot_frames += r.get("frames", 0) or 0
        tot_time += r.get("time_s", r.get("track_s", 0.0)) or 0.0
    return tot_frames / tot_time if tot_time > 0 else 0.0


def suspect_flat_numbering(clips, min_frames=50):
    """True when the prefix-grouping collapsed a MANY-clip concatenation with flat/global frame
    numbering (0001.png..0300.png spanning several clips) into ONE group.

    Signature of the trap: exactly one group, an EMPTY prefix (purely numeric filenames, so the
    regex found no non-numeric prefix to split on) and a large frame count. Treating that as one
    continuous sequence silently invalidates unique-ID / fragmentation metrics, so callers must
    demand a clip manifest instead."""
    if len(clips) != 1:
        return False
    (prefix, frames), = clips.items()
    return prefix == "" and len(frames) >= min_frames


def group_frames_by_clip(frame_dir):
    """Return {prefix: [sorted frame paths]} by stripping the trailing frame number from each stem.
    Matches src.data_prep.verify_sequence so clip boundaries agree across tools.

    Loudly warns (does NOT silently continue) when the result looks like a flat-numbered
    concatenation of many clips collapsed into one group -- that needs an explicit clip manifest."""
    clips = defaultdict(list)
    for f in os.listdir(frame_dir):
        if f.lower().endswith(IMG_EXTS):
            stem = os.path.splitext(f)[0]
            m = re.match(r"^(.*?)(\d+)$", stem)
            clips[m.group(1) if m else stem].append(os.path.join(frame_dir, f))
    for k in clips:
        clips[k].sort()
    clips = dict(clips)
    if suspect_flat_numbering(clips):
        n = len(next(iter(clips.values())))
        print(f"WARNING: {frame_dir!r} grouped into ONE clip from {n} flatly-numbered frames "
              f"(no non-numeric prefix). This is almost certainly MANY clips with global numbering; "
              f"treating it as one continuous sequence invalidates unique-ID/fragmentation metrics. "
              f"Provide a clip manifest / per-clip subdirs instead.")
    return clips


def track_yolo(weights, source, out=None, conf=0.25, tracker="bytetrack.yaml", device=0, audit=True):
    """Track ONE continuous sequence. `source` = mp4 path, image dir, or list of frame paths.
    Returns dict(frames, ids, max_tracks, fps, det_fps, time_s).
      fps     = full detect+track wall-clock throughput (includes ByteTrack association).
      det_fps = detector-only throughput from res.speed (preprocess+inference+postprocess).
    Both exclude the optional video write."""
    from ultralytics import YOLO
    from src.eval.audit import record
    model = YOLO(weights)
    ids, n, max_tracks, det_s, track_s = set(), 0, 0, 0.0, 0.0
    writer = None
    # Time the detect+track step end-to-end: the ByteTrack association happens inside the
    # streaming generator's __next__, so wall-clock across it + box handling is the honest fps.
    # res.speed captures ONLY the detector, which is why it's reported separately as det_fps.
    t_mark = time.perf_counter()
    for res in model.track(source=source, conf=conf, tracker=tracker, stream=True,
                           persist=True, verbose=False, device=device):
        det_s += (res.speed.get("preprocess", 0) + res.speed.get("inference", 0)
                  + res.speed.get("postprocess", 0)) / 1000.0
        n += 1
        box_ids = None if (res.boxes is None or res.boxes.id is None) else res.boxes.id.tolist()
        cur = concurrent_tracks(box_ids)     # assigned track IDs, NOT raw detection count
        max_tracks = max(max_tracks, cur)
        if box_ids is not None:
            ids.update(int(i) for i in box_ids if i is not None)
        track_s += time.perf_counter() - t_mark    # detect+track+box handling; audit/write excluded
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
        t_mark = time.perf_counter()
    if writer:
        writer.release()
    fps = n / track_s if track_s > 0 else 0.0
    det_fps = n / det_s if det_s > 0 else 0.0
    return {"frames": n, "ids": len(ids), "max_tracks": max_tracks,
            "fps": fps, "det_fps": det_fps, "time_s": track_s}


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
    rows, tot_frames, tot_frag = [], 0, 0
    for prefix, frames in sorted(clips.items()):
        r = track_yolo(weights, frames, out=None, conf=conf, tracker=tracker,
                       device=device, audit=False)     # skip audit: 5k rows/clip drowns the trail
        frag = fragmentation(r["ids"], r["max_tracks"])
        r.update(clip=prefix, fragmentation=frag)
        rows.append(r)
        tot_frames += r["frames"]; tot_frag += frag
    agg = {"clips": len(rows), "frames": tot_frames,
           "total_fragmentation": tot_frag,
           "frag_per_clip": round(tot_frag / len(rows), 3),
           "mean_fps": round(weighted_fps(rows), 1),   # frame-weighted, not mean-of-per-clip-fps
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
              f"| {r['fps']:.1f} fps (detect+track), {r['det_fps']:.1f} det_fps | audit -> runs/audit.jsonl")
