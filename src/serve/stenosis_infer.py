"""Sequence-level stenosis inference: detector -> temporal voting -> triage-with-abstention.

The single-frame stenosis detector is BELOW the ship floor, so on its own it is neither precise
nor recall-safe enough to run unattended. This module composes the two Stage-2 post-processors that
make it shippable, in the order that protects RECALL first and then abstains safely:

  1. detector  -- run YOLO over an ORDERED cine window at a DELIBERATELY LOW ``conf`` so a faint but
     real lesion is not thresholded away before the smarter stages can see it. Recall is cheap to
     recover downstream but impossible to invent, so we let the detector over-fire and filter later.
  2. temporal voting (:func:`temporal_vote.aggregate_sequence`) -- a lesion is a physical narrowing;
     it does not blink between adjacent frames. Persistence linking keeps boxes that recur across
     frames (RECALL: frames the detector missed inside a surviving track are interpolated back) and
     drops one-frame flicker (PRECISION), turning the noisy per-frame stream into a stable one.
  3. triage (:func:`stenosis_triage.triage_decision`) -- per frame, keep only confident detections
     and DEFER the uncertain / boundary / possibly-missed frames to a human instead of being
     silently wrong. A missed stenosis is the deadly error, so 'looks clean' must be earned.

The composition is kept PURE-testable: :func:`predict_sequence` takes an injectable ``detect_fn`` so
the temporal-voting + triage wiring can be exercised on synthetic detections with no ultralytics,
torch or GPU. Only :func:`run_detector_sequence` touches the model, and it lazy-imports the heavy
deps inside the function per house style.

A per-frame detection is the plain dict the post-processors speak:
``{"box": (cx, cy, w, h), "conf": float}`` with a normalized YOLO box (center-x/-y, width, height
in [0, 1]).
"""


def run_detector_sequence(detector, frames, conf=0.001, imgsz=768, device=None):
    """Run the YOLO stenosis detector over an ORDERED list of frame paths.

    ``detector`` is a weights path (or an already-loaded ultralytics model); ``frames`` is a list of
    image paths in CINE ORDER (the ordering is what temporal voting relies on). Each frame is read
    grayscale, CLAHE-preprocessed to match training, and predicted; detections are returned as the
    post-processor's dict shape ``{"box": (cx, cy, w, h), "conf": float}`` in normalized ``xywhn``.

    ``conf`` defaults VERY LOW (0.001) on purpose: recall must not be throttled at the detector: the
    downstream temporal voting (persistence) and triage (calibrated keep/defer) are what filter false
    positives, and they can only act on boxes the detector actually emitted. Heavy imports
    (ultralytics/cv2) are lazy so importing this module stays torch-free.

    Returns a per-frame list (same length/order as ``frames``) of detection lists.
    """
    from ultralytics import YOLO
    import cv2
    from src.data_prep.preprocess import clahe_unsharp

    model = detector if hasattr(detector, "predict") else YOLO(detector)
    sequence = []
    for path in frames:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        assert gray is not None, f"cannot read frame: {path}"
        clahe_bgr = cv2.cvtColor(clahe_unsharp(gray), cv2.COLOR_GRAY2BGR)   # detector trained on CLAHE
        res = model.predict(clahe_bgr, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        dets = []
        for box, c in zip(res.boxes.xywhn, res.boxes.conf):
            cx, cy, w, h = (float(v) for v in box.tolist())
            dets.append({"box": (cx, cy, w, h), "conf": float(c)})
        sequence.append(dets)
    return sequence


def predict_sequence(detector, frames, *, temporal=True, min_hits=2, iou_thr=0.3,
                     triage=True, keep_conf=0.5, temperature=1.0, defer_band=(0.3, 0.6),
                     conf=0.001, imgsz=768, device=None, detect_fn=run_detector_sequence):
    """Detector -> temporal voting -> triage over an ordered cine window. Recall-protective.

    Runs ``detect_fn`` over ``frames`` (default :func:`run_detector_sequence`), optionally stabilizes
    the per-frame detections with :func:`temporal_vote.aggregate_sequence` (persistence voting that
    RECOVERS detector-missed frames by interpolation and DROPS one-frame flicker), then runs
    :func:`stenosis_triage.triage_decision` per frame to keep confident detections and DEFER the
    uncertain ones to a human.

    ``detect_fn`` is injectable so the whole composition is unit-testable with a fake sequence and no
    ultralytics/torch. It is called ``detect_fn(detector, frames, conf=conf, imgsz=imgsz,
    device=device)`` and must return a per-frame list of ``{"box": (cx,cy,w,h), "conf": float}``.

    Knobs:
      * ``temporal`` / ``min_hits`` / ``iou_thr`` -- persistence voting. ``min_hits=2`` (default)
        drops single-frame flicker while surviving tracks still recover their internally-missed
        frames; ``temporal=False`` passes the raw per-frame detections straight to triage.
      * ``triage`` / ``keep_conf`` / ``temperature`` / ``defer_band`` -- abstention. ``triage=False``
        surfaces the (stabilized) detections with no keep/defer decision.

    Returns a list, one entry PER FRAME, of ``{"detections", "deferred", "reason"}`` where
    ``detections`` are the triage-surfaced detections for that frame (a possible stenosis is never
    hidden from the human, even on a deferred frame), ``deferred`` flags a frame routed to a human,
    and ``reason`` is the triage reason (``confident`` | ``clean`` | ``low-confidence`` | ``ood`` |
    ``no-detection-uncertain``; ``no-triage`` when ``triage=False``).
    """
    from src.serve.temporal_vote import aggregate_sequence
    from src.serve.stenosis_triage import triage_decision

    raw = detect_fn(detector, frames, conf=conf, imgsz=imgsz, device=device)
    stabilized = aggregate_sequence(raw, iou_thr=iou_thr, min_hits=min_hits) if temporal else raw

    results = []
    for dets in stabilized:
        if triage:
            d = triage_decision(dets, temperature=temperature, keep_conf=keep_conf,
                                defer_band=defer_band)
            results.append({"detections": d["prediction"], "deferred": d["deferred"],
                            "reason": d["reason"]})
        else:
            results.append({"detections": dets, "deferred": False, "reason": "no-triage"})
    return results


if __name__ == "__main__":
    # Synthetic cine window (no model): a lesion detected on frames 0 & 2 (missed 1), a one-frame
    # flicker on frame 3, and a boundary-confidence lesion on frames 3 & 4. Temporal voting recovers
    # frame 1, drops the flicker; triage keeps the confident lesion and DEFERS the boundary one.
    def _fake_detect(detector, frames, **_):
        return [
            [{"box": (0.50, 0.5, 0.2, 0.2), "conf": 0.85}],                        # f0 lesion A
            [],                                                                    # f1 A missed
            [{"box": (0.52, 0.5, 0.2, 0.2), "conf": 0.85}],                        # f2 lesion A
            [{"box": (0.80, 0.5, 0.2, 0.2), "conf": 0.55},                         # f3 lesion B (band)
             {"box": (0.10, 0.1, 0.1, 0.1), "conf": 0.40}],                        # f3 flicker
            [{"box": (0.82, 0.5, 0.2, 0.2), "conf": 0.55}],                        # f4 lesion B
        ]

    for fi, r in enumerate(predict_sequence("<weights>", ["f%d" % i for i in range(5)],
                                            detect_fn=_fake_detect)):
        print(f"frame {fi}: n={len(r['detections'])} deferred={r['deferred']} reason={r['reason']}")
