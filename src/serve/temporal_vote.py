"""Temporal voting = post-process a per-frame stenosis detector over a SHORT CINE WINDOW.

A stenosis is a physical lesion; it does not blink in and out between consecutive cine frames.
The single-frame detector, however, does: it MISSES the lesion on some frames (motion blur, a
faint fill, a bad diastole phase) and it FIRES spuriously on others (a catheter tip, a vessel
crossing) for exactly one frame. Both errors are temporally incoherent, so an ordered window of
detections carries information a single frame cannot:

  * persistence  -> real.  A box that recurs across several frames (linked by IoU) is the lesion.
  * flicker      -> noise. A box present in ONE frame and gone the next is almost always a false
                   positive; dropping it RAISES PRECISION.
  * gap recovery -> recall. Inside a surviving track's span, frames the detector missed are filled
                   by interpolating between the neighbouring hits; recovering those misses RAISES
                   RECALL versus the raw per-frame detector.

This module is pure post-processing over an ORDERED sequence of per-frame detections: no model,
no torch, no numpy, no IO. A "detection" is a plain dict ``{"box": (cx, cy, w, h), "conf": float}``
where the box is a normalized YOLO box (center-x, center-y, width, height in [0, 1]). Everything
here is a pure function so the recall/precision behaviour is unit-testable without a GPU.
"""


# --- IoU on normalized YOLO boxes (pure) --------------------------------------------------------

def iou_xywhn(a, b):
    """Intersection-over-union of two normalized YOLO boxes ``(cx, cy, w, h)``.

    Boxes are center/width/height in [0, 1]; convert each to corner form, intersect, and divide by
    the union area. Returns 0.0 for disjoint boxes and for any degenerate (zero-area) box, so it is
    safe to feed detector output directly without pre-filtering.
    """
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


# --- greedy IoU linking across a short window ---------------------------------------------------

def link_tracks(frames, iou_thr=0.3, max_gap=1):
    """Greedily link per-frame detections into tracks by IoU across (nearly) adjacent frames.

    ``frames`` is a list, ORDERED by frame, of lists of detections ``{"box": (cx,cy,w,h), "conf"}``.
    Returns a list of tracks, each a list of ``(frame_idx, det)`` tuples sorted by frame.

    Linking rule, applied frame by frame:
      * A track is eligible to extend into frame ``f`` while it has skipped at most ``max_gap``
        frames since its last hit (``f - last_frame - 1 <= max_gap``). ``max_gap=0`` links strictly
        adjacent frames; ``max_gap=1`` (default) bridges a single missed frame, which is the common
        one-frame detector drop-out in cine and is what lets a lesion seen on frames 0,2,4 form ONE
        track instead of three fragments.
      * All (eligible-track, current-detection) pairs with ``IoU >= iou_thr`` are matched GREEDILY,
        highest IoU first, each track and detection used at most once. Greedy-by-IoU is deterministic
        and cheap (windows are a handful of frames with a handful of boxes), and avoids a spurious
        crossing detection stealing a track from the box that actually continues it.
      * Any detection left unmatched in frame ``f`` starts a new track.

    Note this only links; it does NOT drop short tracks or fill gaps -- that is
    :func:`aggregate_sequence`'s job. Tracks are returned in creation order (first-seen first).
    """
    tracks = []   # each: {"dets": [(fi, det), ...], "last_frame": int, "last_box": (cx,cy,w,h)}
    for fi, dets in enumerate(frames):
        active = [t for t in tracks if fi - t["last_frame"] - 1 <= max_gap]
        # Score every eligible (track, det) pair, then assign greedily by descending IoU.
        pairs = []
        for ti, t in enumerate(active):
            for di, det in enumerate(dets):
                iou = iou_xywhn(t["last_box"], det["box"])
                if iou >= iou_thr:
                    pairs.append((iou, ti, di))
        pairs.sort(key=lambda p: -p[0])
        used_t, used_d = set(), set()
        for iou, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t, det = active[ti], dets[di]
            t["dets"].append((fi, det))
            t["last_frame"] = fi
            t["last_box"] = det["box"]
        for di, det in enumerate(dets):        # unmatched detections seed new tracks
            if di not in used_d:
                tracks.append({"dets": [(fi, det)], "last_frame": fi, "last_box": det["box"]})
    return [t["dets"] for t in tracks]


# --- confidence aggregation + gap interpolation -------------------------------------------------

def _agg_conf(confs, mode):
    """Aggregate a track's per-hit confidences. 'mean' is robust to a single weak/strong frame;
    'max' is optimistic (report the best evidence); 'min' is conservative (weakest link)."""
    if not confs:
        return 0.0
    if mode == "mean":
        return sum(confs) / len(confs)
    if mode == "max":
        return max(confs)
    if mode == "min":
        return min(confs)
    raise ValueError(f"unknown conf_agg={mode!r}; use 'mean' | 'max' | 'min'")


def _densify(dets, conf):
    """Emit one stabilized detection per frame the track spans, interpolating skipped frames.

    ``dets`` is a track's ``[(frame_idx, det), ...]``. Observed frames are emitted with
    ``interpolated=False``; each gap frame strictly between two consecutive hits gets a box linearly
    interpolated (per coordinate) between the bracketing hits -- these carry ``interpolated=True`` so
    a caller can tell a recovered miss from a real detection. Every emitted box carries the same
    track-level aggregated ``conf``.
    """
    dets = sorted(dets, key=lambda x: x[0])
    out = []
    for idx, (fi, det) in enumerate(dets):
        out.append((fi, {"box": tuple(det["box"]), "conf": conf, "interpolated": False}))
        if idx + 1 < len(dets):
            fj, detj = dets[idx + 1]
            b0, b1 = det["box"], detj["box"]
            for fg in range(fi + 1, fj):       # skipped frames -> interpolate to recover the miss
                alpha = (fg - fi) / (fj - fi)
                box = tuple(b0[k] + alpha * (b1[k] - b0[k]) for k in range(4))
                out.append((fg, {"box": box, "conf": conf, "interpolated": True}))
    return out


def aggregate_sequence(frames, iou_thr=0.3, min_hits=2, conf_agg="mean"):
    """Stabilize an ordered sequence of per-frame detections by temporal persistence voting.

    Pipeline: link detections into tracks (:func:`link_tracks`), keep only tracks that persist for
    at least ``min_hits`` frames, then emit one stabilized detection per surviving track for every
    frame it spans -- interpolating boxes for frames the detector skipped inside that span -- each
    carrying the track's aggregated confidence (``conf_agg`` in {'mean','max','min'}).

    Returns per-frame stabilized detections in the SAME shape as ``frames``: a list (indexed by
    frame) of lists of dicts. Each dict keeps ``box`` and ``conf`` like the input and adds
    ``interpolated`` (True for a box synthesized to recover a missed frame).

    ``min_hits`` is THE recall/precision knob:
      * min_hits = 1  -> keep every track. No flicker is rejected; you get the raw detector's recall
        with no precision gain (and no persistence guarantee).
      * min_hits = 2 (default) -> a detection must be corroborated by at least one other frame in its
        track to survive. This drops single-frame flicker (a big PRECISION win) while the surviving
        tracks still recover their internally-missed frames via interpolation (a RECALL win over the
        raw per-frame detector). One corroborating frame is usually enough for a real lesion.
      * larger min_hits -> demands longer persistence: rejects more transient false positives (higher
        precision) but also discards genuinely short-lived / briefly-visible lesions (lower recall).
      In short: raising ``min_hits`` trades recall for precision; lowering it trades precision for
      recall. The interpolation step adds recall independently of ``min_hits`` for tracks that survive.
    """
    tracks = link_tracks(frames, iou_thr=iou_thr)
    out = [[] for _ in range(len(frames))]
    for dets in tracks:
        if len(dets) < min_hits:               # not persistent -> flicker/false-positive, drop it
            continue
        conf = _agg_conf([d["conf"] for _, d in dets], conf_agg)
        for fi, det in _densify(dets, conf):
            out[fi].append(det)
    return out


# --- tiny synthetic demo ------------------------------------------------------------------------

def _demo():
    """Synthetic 5-frame window: a persistent lesion the detector misses on frames 1 & 3, plus a
    one-frame flicker. Shows min_hits=2 recovering the misses and dropping the flicker."""
    lesion = lambda cx: {"box": (cx, 0.5, 0.2, 0.2), "conf": 0.8}
    frames = [
        [lesion(0.50)],                                   # f0: lesion
        [{"box": (0.1, 0.1, 0.1, 0.1), "conf": 0.4}],     # f1: flicker only (detector missed lesion)
        [lesion(0.52)],                                   # f2: lesion
        [],                                               # f3: detector missed everything
        [lesion(0.54)],                                   # f4: lesion
    ]
    stab = aggregate_sequence(frames, min_hits=2)
    for fi, dets in enumerate(stab):
        tags = [("interp" if d["interpolated"] else "obs", round(d["box"][0], 3)) for d in dets]
        print(f"frame {fi}: raw={len(frames[fi])} det -> stabilized={len(dets)} {tags}")


if __name__ == "__main__":
    _demo()
