"""Per-source annotation QA for the merged stenosis YOLO dataset (Stage 2 Phase 2, P2.1).

The train-run diagnostic found mAP50 0.209 vs mAP50-95 0.080 -- a 2.6x collapse that only
happens when boxes are in roughly the right place but loosely localized. That pattern points
at a box-*convention* mismatch across the three merged sources (ARCADE / CADICA / Danilov):
different upstream tooling drawing/sizing boxes differently around the same lesions. This
module quantifies that by computing per-source box-geometry stats (width/height/area
percentiles, tiny-box fraction) straight from the YOLO label files, so the mismatch is visible
without needing a trained model.

stdlib-only (os, glob, argparse, statistics import only). Reuses the source classifier from
val_by_source (`source_of`), which is itself torch-free -- ultralytics is only imported lazily
inside val_by_source.main(), never at module import time, so importing it here does not pull in
torch/ultralytics/cv2 (repo invariant: src/* and tests/* import without those heavy deps).
"""
import os, glob, argparse

from src.eval.val_by_source import source_of

_TINY_SQRT_AREA = 0.05  # ~38px at 768px side


def parse_yolo_label(text):
    """Parse the text of a YOLO label file into a list of (cls, cx, cy, w, h) float tuples.

    Lines that don't have exactly 5 whitespace-separated fields (blank lines, malformed rows)
    are skipped rather than raising, since real label files in the wild occasionally have stray
    blank lines.
    """
    boxes = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, w, h = parts
        boxes.append((int(float(cls)), float(cx), float(cy), float(w), float(h)))
    return boxes


def _pct(values, q):
    """The q-th percentile (q in [0, 100]) of `values` via linear interpolation on the sorted
    list (same convention as numpy's default 'linear' method). Returns None for an empty input.
    """
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    if n == 1:
        return float(s[0])
    rank = (q / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def box_stats(boxes):
    """Geometry stats for a pooled list of (cls, cx, cy, w, h) boxes.

    area is normalized w*h; tiny_frac is the fraction of boxes whose sqrt(area) is below
    _TINY_SQRT_AREA (~38px at a 768px image side) -- i.e. boxes small enough that a mAP50-95
    style overlap requirement gets punishing even for a near-correct prediction.
    """
    n = len(boxes)
    if n == 0:
        return {
            "n_boxes": 0,
            "w_p10": None, "w_p50": None, "w_p90": None,
            "h_p50": None,
            "area_p10": None, "area_p50": None, "area_p90": None,
            "tiny_frac": 0.0,
        }
    widths = [b[3] for b in boxes]
    heights = [b[4] for b in boxes]
    areas = [b[3] * b[4] for b in boxes]
    tiny = sum(1 for a in areas if a ** 0.5 < _TINY_SQRT_AREA)
    return {
        "n_boxes": n,
        "w_p10": round(_pct(widths, 10), 4),
        "w_p50": round(_pct(widths, 50), 4),
        "w_p90": round(_pct(widths, 90), 4),
        "h_p50": round(_pct(heights, 50), 4),
        "area_p10": round(_pct(areas, 10), 4),
        "area_p50": round(_pct(areas, 50), 4),
        "area_p90": round(_pct(areas, 90), 4),
        "tiny_frac": round(tiny / n, 4),
    }


def summarize(proc, split="train"):
    """Walk `<proc>/labels/<split>/*.txt`, classify each label stem by source, and pool box
    geometry per source. Returns {source: {**box_stats(pooled_boxes), n_images, boxes_per_img}}.

    n_images counts label files seen for that source, including empty files (background
    frames with no annotated stenosis) -- those still count as images for that source.
    """
    buckets = {}   # source -> list of boxes
    n_images = {}  # source -> count of label files
    pattern = os.path.join(proc, "labels", split, "*.txt")
    for lp in sorted(glob.glob(pattern)):
        stem = os.path.splitext(os.path.basename(lp))[0]
        src = source_of(stem)
        n_images[src] = n_images.get(src, 0) + 1
        boxes = parse_yolo_label(open(lp).read())
        buckets.setdefault(src, []).extend(boxes)

    out = {}
    for src in n_images:
        stats = box_stats(buckets.get(src, []))
        ni = n_images[src]
        stats["n_images"] = ni
        stats["boxes_per_img"] = round(stats["n_boxes"] / ni, 4) if ni else 0.0
        out[src] = stats
    return out


def _format_table(summary):
    header = f"{'source':10s} {'n_img':>6s} {'n_box':>6s} {'box/img':>8s} {'w_p50':>7s} {'h_p50':>7s} {'area_p50':>9s} {'tiny_frac':>9s}"
    lines = [header, "-" * len(header)]
    for src in sorted(summary):
        s = summary[src]
        lines.append(
            f"{src:10s} {s['n_images']:6d} {s['n_boxes']:6d} {s['boxes_per_img']:8.4f} "
            f"{_fmt(s['w_p50']):>7s} {_fmt(s['h_p50']):>7s} {_fmt(s['area_p50']):>9s} {s['tiny_frac']:9.4f}"
        )
    return "\n".join(lines)


def _fmt(v):
    return "n/a" if v is None else f"{v:.4f}"


def main(proc="data/processed/stenosis", split="train"):
    summary = summarize(proc, split=split)
    table = _format_table(summary)
    print(table)
    if os.path.isdir("/kaggle/working"):
        with open("/kaggle/working/phase2_annotation_qa.txt", "w") as f:
            f.write(table + "\n")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", default="data/processed/stenosis")
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    main(a.proc, a.split)
