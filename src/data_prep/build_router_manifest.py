"""Scan the per-modality dataset roots already on disk and emit a (path,label) manifest CSV for
router training (B2 `train_router.py` consumes it). Labels come from substring rules on the path
(dataset dir name -> modality), so adding a modality to the router = adding one rule to
`configs/router.yaml`'s `rules` map -- no code change needed. A frame whose path matches no rule is
DROPPED (counted, never guessed at) so the manifest never mislabels a source it doesn't recognize.

Pure stdlib walk + the repo's IMG_EXTS convention (src/data_prep/preprocess.py) -- no cv2, no
torch/ultralytics/coremltools/transformers, so this stays importable on a laptop with no GPU deps.
`label_for_path` is pure (no filesystem) and is the unit-tested core; `build_manifest` is the IO
shell around it.
"""
import argparse
import csv
import glob
import os

import yaml

from src.data_prep.preprocess import IMG_EXTS


def label_for_path(path, rules):
    """First modality whose rule matches `path`; else None.

    `rules` is {label: [substring, ...]}. A label matches if ANY of its substrings appears in the
    lowercased path. Dict order is the priority order: if a path happens to satisfy more than one
    label (e.g. an aliased/nested dataset name), the first matching label in `rules` wins.
    """
    p = path.lower()
    for label, substrings in rules.items():
        if any(s.lower() in p for s in substrings):
            return label
    return None


def _iter_images(root):
    """Yield every image file path under `root` (recursive), reusing the repo's IMG_EXTS convention."""
    for path in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if os.path.isfile(path) and os.path.splitext(path)[1].lower() in IMG_EXTS:
            yield path


def build_manifest(roots, rules, out_csv, per_class_cap=4000):
    """Walk every root in `roots`, label each image via `label_for_path`, write a (path,label) CSV
    to `out_csv`. Unmatched frames are dropped, never mislabeled.

    `per_class_cap` (default 4000, matching configs/router.yaml): once a label hits the cap no more
    of its rows are written, so one oversized source can't swamp router training. Rows are counted
    in walk order (roots in the given order, then glob's order within a root), so the cap is
    deterministic for a fixed directory tree but is NOT a "most balanced" subsample -- just a ceiling.

    Returns {"counts": {label: n_written}, "unmatched": n_dropped, "rows": total_rows_written}.
    """
    counts = {}
    unmatched = 0
    rows = []
    for root in roots:
        for path in _iter_images(root):
            label = label_for_path(path, rules)
            if label is None:
                unmatched += 1
                continue
            if per_class_cap is not None and counts.get(label, 0) >= per_class_cap:
                continue
            counts[label] = counts.get(label, 0) + 1
            rows.append((path, label))

    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("path", "label"))
        writer.writerows(rows)

    return {"counts": counts, "unmatched": unmatched, "rows": len(rows)}


def main(cfg):
    report = build_manifest(cfg["roots"], cfg["rules"], cfg["manifest_csv"],
                             per_class_cap=cfg.get("per_class_cap", 4000))
    print(f"router manifest -> {cfg['manifest_csv']} : {report['rows']} rows, "
          f"per-class {report['counts']}, unmatched {report['unmatched']}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    main(yaml.safe_load(open(a.config)))
