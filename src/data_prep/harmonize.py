"""Box-size harmonization (Stage 2, P2.1) — clamp every YOLO box up to a common minimum w/h floor.

Why: the three merged sources annotate stenosis at very different box sizes (annotation QA §3c:
median box area arcade 0.0108 / cadica 0.0058 / danilov 0.0029; danilov tiny_frac 0.36). The model
can't learn a consistent target size from mixed conventions, which caps the IoU-sensitive metric
(mAP50 0.209 vs mAP50-95 0.080). Clamping sub-floor boxes up to a minimum w/h gives one consistent
minimum target WITHOUT dropping any positive (recall-preserving), unlike simply deleting tiny boxes.

TRAIN-ONLY IS NOT FREE (audit B5). The default `splits=("train",)` keeps val scored against the
ORIGINAL boxes, so the *metric* stays comparable to the un-harmonized baseline — but the *model* is
then trained to emit boxes systematically LARGER than the evaluation target. At IoU 0.5 an
over-sized prediction loses IoU against a small GT box, so train-only harmonization can LOWER mAP50
and will hit mAP50-95 harder still, even when it genuinely fixes the source convention mismatch. A
drop therefore does NOT mean the convention fix was wrong, and a rise is not clean evidence it was
right: the lever's effect is confounded with the train/eval target mismatch it introduces.

So before trusting any number from this lever, score the SAME weights on BOTH the original val and a
harmonized val (`splits=("train","val")`); only then is the mAP change attributable to the
harmonization rather than to the moved target. `harmonization_warning()` states this at runtime the
moment the lever is switched on. `splits=("train","val")` moves the eval target and removes the
mismatch — that is a different, also-legitimate choice, not a safer one.

Pure stdlib (os/glob/argparse) — no cv2/torch. It only rewrites the YOLO .txt label files in place.
"""
import argparse, glob, os

OUT = "data/processed/stenosis"


def clamp_box_wh(cx, cy, w, h, min_wh):
    """Expand a normalized YOLO box so w,h >= min_wh, keeping the center where possible.

    If expanding pushes an edge out of [0,1], the center is shifted inward so the box stays in frame;
    a box wider/taller than the frame is centered and set to full extent. A box already >= min_wh in a
    dimension is left untouched in that dimension. Returns (cx, cy, w, h)."""
    w2 = w if w >= min_wh else min_wh
    h2 = h if h >= min_wh else min_wh
    if w2 >= 1.0:
        cx, w2 = 0.5, 1.0
    else:
        cx = min(max(cx, w2 / 2), 1.0 - w2 / 2)
    if h2 >= 1.0:
        cy, h2 = 0.5, 1.0
    else:
        cy = min(max(cy, h2 / 2), 1.0 - h2 / 2)
    return cx, cy, w2, h2


def _fmt(v):
    """6-dp like the converters, but drop trailing zeros so 0.04 stays '0.04' (stable test/output)."""
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def harmonize_label_lines(lines, min_wh):
    """Clamp every 5-field YOLO line's box to min_wh. Returns (new_lines, n_changed).

    Malformed / blank lines pass through unchanged. A line is counted changed only if a dimension
    was actually below the floor."""
    out, changed = [], 0
    for ln in lines:
        parts = ln.split()
        if len(parts) != 5:
            out.append(ln)
            continue
        try:
            cls = parts[0]
            cx, cy, w, h = (float(p) for p in parts[1:])
        except ValueError:
            out.append(ln)
            continue
        if w >= min_wh and h >= min_wh:
            out.append(ln)                       # nothing to clamp -> preserve original text
            continue
        ncx, ncy, nw, nh = clamp_box_wh(cx, cy, w, h, min_wh)
        out.append(f"{cls} {_fmt(ncx)} {_fmt(ncy)} {_fmt(nw)} {_fmt(nh)}")
        changed += 1
    return out, changed


def harmonization_warning(min_box_wh, splits=("train",)):
    """Return the train/eval mismatch warning text for this setting, or None if there is none. Pure.

    A mismatch exists exactly when ONE of train/val is harmonized and the other is not: the model's
    predicted box-size distribution then no longer matches the distribution it is scored against.
    Returns None when the lever is off (min_box_wh <= 0) and when train and val move together."""
    try:
        m = float(min_box_wh or 0.0)
    except (TypeError, ValueError):
        return None
    if m <= 0:
        return None
    s = {str(x).strip().lower() for x in (splits or ())}
    if ("train" in s) == ("val" in s):
        return None            # both harmonized (targets agree) or neither (nothing was clamped)
    shown = list(splits or ())
    if "train" in s:
        return (
            f"[harmonize] WARNING: min_box_wh={m} clamps the TRAIN split only (splits={shown}).\n"
            "  The MODEL will learn to emit boxes at least this wide/tall, but val is still scored\n"
            "  against the ORIGINAL (smaller) boxes -- so predictions become systematically LARGER\n"
            "  than the evaluation target. An over-sized prediction loses IoU against a small GT\n"
            "  box, so this lever can LOWER mAP50 and will hit mAP50-95 harder still, even when it\n"
            "  genuinely fixes the source annotation-convention mismatch. Keeping val un-harmonized\n"
            "  makes the metric COMPARABLE to the baseline; it does not make it a FAIR test of the\n"
            "  lever -- the effect is confounded with the train/eval target mismatch.\n"
            "  RECOMMENDED: score the SAME weights on BOTH the ORIGINAL val and a HARMONIZED val\n"
            "  (harmonize.splits: [train, val]) and report the pair. Only then is a change in\n"
            "  mAP50 / mAP50-95 attributable to the harmonization rather than to the moved target."
        )
    return (
        f"[harmonize] WARNING: min_box_wh={m} clamps the VAL split only (splits={shown}).\n"
        "  The evaluation target moved but the model was trained on the ORIGINAL box sizes, so its\n"
        "  predictions are systematically SMALLER than the val boxes they are matched against, and\n"
        "  mAP50 / mAP50-95 will move for a reason unrelated to model quality. Harmonize train too\n"
        "  (harmonize.splits: [train, val]) or neither, and score BOTH the ORIGINAL and HARMONIZED\n"
        "  val with the same weights so the effect is attributable rather than confounded."
    )


def harmonize_labels(proc=OUT, min_wh=0.0, splits=("train",)):
    """Rewrite YOLO labels under proc/labels/<split>, clamping boxes to min_wh. min_wh<=0 -> no-op.

    Returns {"files": n_files_seen, "boxes_clamped": total}. Default TRAIN-ONLY, which leaves val
    scored against the ORIGINAL boxes: comparable to the baseline, but it trains the model to emit
    boxes LARGER than the eval target and so can LOWER mAP50 / mAP50-95 (see module docstring and
    audit B5). Prints `harmonization_warning()` when that mismatch is live."""
    rep = {"files": 0, "boxes_clamped": 0}
    if not min_wh or min_wh <= 0:
        return rep
    warn = harmonization_warning(min_wh, splits)
    if warn:
        print(warn)
    for sp in splits:
        for lp in sorted(glob.glob(os.path.join(proc, "labels", sp, "*.txt"))):
            rep["files"] += 1
            lines = open(lp).read().splitlines()
            new, changed = harmonize_label_lines(lines, min_wh)
            if changed:
                open(lp, "w").write("\n".join(new) + ("\n" if new else ""))
                rep["boxes_clamped"] += changed
    return rep


def main(cfg, proc=OUT):
    h = (cfg or {}).get("harmonize") or {}
    min_wh = float(h.get("min_box_wh", 0.0) or 0.0)
    splits = tuple(h.get("splits", ["train"]))
    if min_wh <= 0:
        print("[harmonize] min_box_wh <= 0 -> skipped (set harmonize.min_box_wh, e.g. 0.04, to enable)")
        return {"files": 0, "boxes_clamped": 0}
    rep = harmonize_labels(proc, min_wh, splits)
    print(f"[harmonize] min_box_wh={min_wh} splits={list(splits)} -> "
          f"clamped {rep['boxes_clamped']} boxes across {rep['files']} label files")
    return rep


if __name__ == "__main__":
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--proc", default=OUT)
    a = ap.parse_args()
    main(yaml.safe_load(open(a.config)), proc=a.proc)
