"""CathAction -> YOLO (2 classes: catheter, guidewire). Feeds the catheter/guidewire detector.

CathAction ships per-frame instance masks (catheter + guidewire labelled separately) and, in some
releases, COCO json. We take the COCO path when present; otherwise derive boxes from per-class masks.
Confirm your CathAction download layout and adjust `_mask_dirs` if the folder names differ.

Heavy deps (cv2, numpy, pycocotools via io_utils) are imported lazily inside the functions that
need them, so the pure helpers below (`classify_mask_dir`, `coco_classmap_by_name`,
`pair_img_mask_dirs`, `mask_value_to_class`) can be imported and unit-tested without cv2 installed.
"""
import argparse, glob, json, os, re, yaml

OUT = "data/processed/catheter"
NAMES = ("catheter", "guidewire")


# --------------------------------------------------------------------------------------------------
# Pure helpers (stdlib only; unit-testable without cv2/numpy/pycocotools)
# --------------------------------------------------------------------------------------------------
def classify_mask_dir(basename, names):
    """Map a mask-folder *basename* to exactly one yolo class index, or None.

    A class name must appear as a whole token (basename split on any non-alphanumeric run) and be
    the ONLY class name present. So ``catheter`` -> 0, ``guidewire_masks`` -> 1, but
    ``catheter_guidewire`` -> None (ambiguous) instead of matching BOTH classes like the old
    ``*catheter*`` / ``*guidewire*`` substring globs did.
    """
    tokens = {t for t in re.split(r"[^a-z0-9]+", basename.lower()) if t}
    present = [i for i, nm in enumerate(names) if nm.lower() in tokens]
    return present[0] if len(present) == 1 else None


def coco_classmap_by_name(categories, names):
    """Build ``{coco_category_id: yolo_idx}`` from COCO categories by matching NAME, not id.

    ``categories`` is the COCO ``categories`` list (dicts with ``id`` + ``name``). Each category's
    name is matched case-insensitively to its position in ``names`` — so 0-indexed, reversed, or
    sparse category ids all map to the right class. Categories whose name is not in ``names`` are
    skipped with a warning (their boxes are dropped on purpose rather than silently mislabelled).
    """
    lut = {nm.lower(): i for i, nm in enumerate(names)}
    cmap = {}
    for c in categories or []:
        try:
            cid, cname = c["id"], str(c["name"]).strip().lower()
        except (KeyError, TypeError):
            continue
        if cname in lut:
            cmap[cid] = lut[cname]
        else:
            print(f"[cathaction] WARNING: COCO category {c.get('name')!r} (id={c.get('id')}) "
                  f"is not one of {list(names)}; skipping its boxes")
    return cmap


def pair_img_mask_dirs(dirs):
    """Pair ``img`` and ``mask`` subdirs that share the SAME parent (one clip/sequence each).

    ``dirs`` is any iterable of directory paths; only those whose basename is exactly ``img`` or
    ``mask`` are considered. Returns ``[(img_dir, mask_dir), ...]`` — one pair per parent that has
    BOTH, sorted by parent for determinism. Pairing is by shared parent, never by list position,
    so every clip is matched (the old code used ``img_dirs[0], mask_dirs[0]`` and dropped the rest).
    """
    imgs, masks = {}, {}
    for d in dirs:
        nd = os.path.normpath(d)
        parent, base = os.path.dirname(nd), os.path.basename(nd)
        if base == "img":
            imgs[parent] = d
        elif base == "mask":
            masks[parent] = d
    return [(imgs[p], masks[p]) for p in sorted(imgs) if p in masks]


def mask_value_to_class(values, names):
    """Map nonzero mask pixel values -> yolo class indices, or raise on an ambiguous encoding.

    ``values`` = the unique pixel values present in a merged mask. Returns ``{value: yolo_idx}``.

    - Per-class value coding (catheter=1, guidewire=2, ...): each value ``v`` in ``1..len(names)``
      maps to ``v - 1``. A guidewire-only frame ({2}) therefore maps to class 1, NOT catheter.
    - A single-class *binary* mask (e.g. 0/255) carries no class information. Rather than defaulting
      the foreground to class 0 (catheter) — which turned every guidewire-only frame into a false
      catheter box — we RAISE ``ValueError`` so the caller fails loudly / skips the frame.
    """
    vals = sorted({int(v) for v in values if int(v) != 0})
    if not vals:
        return {}
    n = len(names)
    if all(1 <= v <= n for v in vals):
        return {v: v - 1 for v in vals}
    raise ValueError(
        f"ambiguous mask encoding {vals}: values are not per-class codes in 1..{n} "
        f"({list(names)}) — refusing to default the foreground to catheter")


# --------------------------------------------------------------------------------------------------
# Mask / COCO discovery + conversion (need cv2 / numpy / io_utils)
# --------------------------------------------------------------------------------------------------
def _mask_dirs(root):
    """Return ``{yolo_idx: [mask_dir, ...]}`` by classifying each folder basename to ONE class."""
    dirs = {}
    for d in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if not os.path.isdir(d):
            continue
        idx = classify_mask_dir(os.path.basename(os.path.normpath(d)), NAMES)
        if idx is not None:
            dirs.setdefault(idx, []).append(d)
    return dirs


def _from_masks(root, size):
    import cv2
    from src.data_prep import io_utils as io
    dirs = _mask_dirs(root)
    if not dirs:
        return 0
    # index masks by frame stem across classes
    frames = {}
    for idx, mask_dirs in dirs.items():
        for d in mask_dirs:
            for mp in glob.glob(os.path.join(d, "*")):
                stem = os.path.splitext(os.path.basename(mp))[0]
                frames.setdefault(stem, []).append((idx, mp))
    n = 0
    for stem, items in frames.items():
        ip = io.resolve_image(root, stem + ".png") or io.resolve_image(root, stem + ".jpg")
        if not ip:
            continue
        class_masks = [(idx, cv2.imread(mp, cv2.IMREAD_GRAYSCALE)) for idx, mp in items]
        class_masks = [(i, m) for i, m in class_masks if m is not None]
        n += io.masks_to_yolo(ip, class_masks, OUT, size)
    return n


def _from_img_mask_pairs(root, size):
    """CathAction human_dataset_train layout: ``<root>/**/<clip>/img/<stem>.<ext>`` +
    ``<root>/**/<clip>/mask/<stem>_mask.png`` (one merged mask per frame, NOT per-class dirs).

    EVERY (img, mask) clip pair sharing a parent is processed (not just the first). Nonzero mask
    values are decoded by ``mask_value_to_class`` so a single-class frame keeps its true class
    (catheter=1->0, guidewire=2->1); a genuinely binary mask (e.g. 0/255) is ambiguous and its
    frame is skipped with a warning rather than mislabelled catheter. Returns frames converted."""
    import cv2
    import numpy as np
    from src.data_prep import io_utils as io
    img_dirs  = [d for d in glob.glob(os.path.join(root, "**", "img"),  recursive=True) if os.path.isdir(d)]
    mask_dirs = [d for d in glob.glob(os.path.join(root, "**", "mask"), recursive=True) if os.path.isdir(d)]
    pairs = pair_img_mask_dirs(img_dirs + mask_dirs)
    if not pairs:
        return 0
    n = 0
    for img_dir, mask_dir in pairs:
        imgs = {os.path.splitext(os.path.basename(p))[0]: p
                for p in glob.glob(os.path.join(img_dir, "*"))}
        for mp in glob.glob(os.path.join(mask_dir, "*")):
            stem = os.path.splitext(os.path.basename(mp))[0]
            if stem.endswith("_mask"):
                stem = stem[:-len("_mask")]                      # mask files are <stem>_mask.png
            ip = imgs.get(stem)
            if not ip:
                continue
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            vals = [int(v) for v in np.unique(m) if int(v) != 0]
            if not vals:
                continue                                         # empty mask, nothing to emit
            try:
                vmap = mask_value_to_class(vals, NAMES)
            except ValueError as e:                              # ambiguous binary mask -> skip loud
                print(f"[cathaction] WARNING: skipping {mp}: {e}")
                continue
            class_masks = [(cls, (m == val).astype("uint8")) for val, cls in sorted(vmap.items())]
            n += io.masks_to_yolo(ip, class_masks, OUT, size)
    return n


def _coco_classmap(root):
    """Build the ``{coco_cat_id: yolo_idx}`` map from category NAMES across all COCO jsons."""
    from src.data_prep import io_utils as io
    cmap = {}
    for jp in io.find_coco_jsons(root):
        try:
            cats = json.load(open(jp)).get("categories", [])
        except Exception:
            continue
        cmap.update(coco_classmap_by_name(cats, NAMES))
    return cmap


def main(cfg):
    from src.data_prep import io_utils as io
    root = cfg["datasets"]["cathaction"]["root"]
    size = cfg.get("detector", {}).get("imgsz", 640)
    cmap = _coco_classmap(root)                                 # COCO cat ids -> yolo idx BY NAME
    n = io.coco_to_yolo(root, OUT, size=size, class_map=cmap) if cmap else 0   # COCO path if present
    if n == 0:
        n = _from_masks(root, size)                            # per-class mask dirs
    if n == 0:
        n = _from_img_mask_pairs(root, size)                  # img/ + mask/ single-mask layout
    if n == 0:
        raise SystemExit(f"No CathAction annotations converted under {root!r}. Check layout.")
    yml = io.write_yolo_datayaml(OUT, names=NAMES)
    print(f"CathAction -> {OUT} : {n} frames ; data cfg {yml}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    a = ap.parse_args(); main(yaml.safe_load(open(a.config)))
