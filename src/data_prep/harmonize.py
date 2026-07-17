"""Box-size harmonization (Stage 2, P2.1) — clamp every YOLO box up to a common minimum w/h floor.

Why: the three merged sources annotate stenosis at very different box sizes (annotation QA §3c:
median box area arcade 0.0108 / cadica 0.0058 / danilov 0.0029; danilov tiny_frac 0.36). The model
can't learn a consistent target size from mixed conventions, which caps the IoU-sensitive metric
(mAP50 0.209 vs mAP50-95 0.080). Clamping sub-floor boxes up to a minimum w/h gives one consistent
minimum target WITHOUT dropping any positive (recall-preserving), unlike simply deleting tiny boxes.

Default is TRAIN-ONLY (`splits=("train",)`): the model learns the harmonized size, but val is scored
against the ORIGINAL boxes so the metric stays comparable to the un-harmonized baseline. Pass
`splits=("train","val")` only if you deliberately want to move the eval target too.

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


def harmonize_labels(proc=OUT, min_wh=0.0, splits=("train",)):
    """Rewrite YOLO labels under proc/labels/<split>, clamping boxes to min_wh. min_wh<=0 -> no-op.

    Returns {"files": n_files_seen, "boxes_clamped": total}. Default TRAIN-ONLY (keeps val honest)."""
    rep = {"files": 0, "boxes_clamped": 0}
    if not min_wh or min_wh <= 0:
        return rep
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
