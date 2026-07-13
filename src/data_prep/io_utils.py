"""Shared data-prep IO: standardized (img,msk) PNG pairs, nnU-Net raw layout, COCO->YOLO.

Keeps every converter emitting the SAME on-disk shapes so datasets are interchangeable:
  - segmentation student:  data/processed/<task>/{img,msk}/<stem>.png   (msk 0/255)
  - nnU-Net teacher:       $nnUNet_raw/Dataset0XX_<name>/{imagesTr,labelsTr} + dataset.json
  - YOLO detector:         data/processed/<task>/{images,labels}/<split>/<stem>.{png,txt}
"""
import glob, hashlib, json, os, re
import cv2, numpy as np
from src.data_prep.preprocess import clahe_unsharp

# Video-derived frames are near-identical between consecutive frames, so a per-frame split leaks
# the same sequence into train AND val. Collapse every frame of one source sequence to a single
# group so it lands entirely on one side (honest holdout):
#   Danilov    <site>_<patient>_<seq>_<frame>  (e.g. 14_002_5_0016)          -> <site>_<patient>
#   CathAction <clip>_img-<seg>-<frame>        (e.g. JFQ_j3383201_img-00000-0042) -> <clip>
_PATIENT_RE = re.compile(r"^(\d+_\d+)_\d+_\d+$")
_CLIP_RE = re.compile(r"^(.+?)_img-\d+-\d+$")


def group_key(name):
    """Split-group key: collapse a source sequence's frames to one key; else the name itself."""
    m = _PATIENT_RE.match(name)
    if m:
        return m.group(1)
    m = _CLIP_RE.match(name)
    if m:
        return m.group(1)
    return name


def ensure(*ds):
    for d in ds:
        os.makedirs(d, exist_ok=True)


def split_of(name, val_frac=0.15):
    """Deterministic, patient-grouped train/val split (stable across runs/processes).
    Hashes group_key(name) so all frames of a Danilov patient share a split (no frame leakage)."""
    h = int(hashlib.md5(group_key(name).encode()).hexdigest(), 16) % 1000
    return "val" if h < val_frac * 1000 else "train"


def cap_frames_per_patient(stems, k, key_fn=group_key):
    """Keep at most ``k`` stems per patient group, EVENLY SPACED across each group's sorted frames.

    Danilov ships ~8325 near-identical frames from only 64 patients; keeping every frame dilutes the
    honest per-patient metric with redundant almost-duplicates. This caps each ``key_fn(stem)`` group
    to ``k`` frames chosen at evenly-spaced indices from ``0`` to ``m-1`` (both endpoints included),
    so the retained frames still span the whole clip's temporal range — strictly better coverage than
    first-``k``, which biases to the start of a sequence. Fully deterministic (sorted + arithmetic
    index selection, no RNG). ``k=None`` -> no cap (all stems returned, sorted). Returns the kept
    stems as a sorted list.
    """
    if k is None:
        return sorted(stems)
    groups = {}
    for s in stems:
        groups.setdefault(key_fn(s), []).append(s)
    kept = []
    for key in sorted(groups):
        frames = sorted(groups[key])
        m = len(frames)
        if m <= k:
            kept.extend(frames)
            continue
        idxs = [m // 2] if k == 1 else [round(i * (m - 1) / (k - 1)) for i in range(k)]
        seen = set()
        for j in idxs:                       # dedupe defensively (indices are distinct for k<=m)
            if j not in seen:
                seen.add(j)
                kept.append(frames[j])
    return sorted(kept)


def write_pair(img_gray, mask, stem, out_dir, size=512, clahe=True):
    ensure(os.path.join(out_dir, "img"), os.path.join(out_dir, "msk"))
    im = clahe_unsharp(img_gray) if clahe else img_gray
    im = cv2.resize(im, (size, size))
    mk = cv2.resize((mask > 0).astype(np.uint8) * 255, (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(out_dir, "img", stem + ".png"), im)
    cv2.imwrite(os.path.join(out_dir, "msk", stem + ".png"), mk)


def write_nnunet_case(img_gray, mask, stem, raw_dir, size=512, clahe=True):
    imtr, latr = os.path.join(raw_dir, "imagesTr"), os.path.join(raw_dir, "labelsTr")
    ensure(imtr, latr)
    im = clahe_unsharp(img_gray) if clahe else img_gray
    cv2.imwrite(os.path.join(imtr, f"{stem}_0000.png"), cv2.resize(im, (size, size)))
    mk = cv2.resize((mask > 0).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(latr, f"{stem}.png"), mk)          # label values 0/1


def write_nnunet_datasetjson(raw_dir, n, channel="XCA"):
    json.dump({"channel_names": {"0": channel},
               "labels": {"background": 0, "vessel": 1},
               "numTraining": n, "file_ending": ".png"},
              open(os.path.join(raw_dir, "dataset.json"), "w"), indent=2)


def resolve_image(root, file_name):
    fn = os.path.basename(file_name)
    # fast paths: COCO json usually sits in .../annotations, images in a sibling .../images
    for cand in (os.path.join(root, file_name), os.path.join(root, fn),
                 os.path.join(root, "..", "images", fn), os.path.join(root, "images", fn)):
        if os.path.exists(cand):
            return cand
    hits = glob.glob(os.path.join(root, "**", fn), recursive=True)   # fallback (slow)
    return hits[0] if hits else None


def find_coco_jsons(root):
    """Yield paths of every COCO-shaped json under root."""
    for jp in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
        try:
            d = json.load(open(jp))
        except Exception:
            continue
        if isinstance(d, dict) and "images" in d and "annotations" in d:
            yield jp


def coco_seg_to_pairs(root, out_dir, size=512, raw_dir=None):
    """ARCADE-style: union all polygon anns per image -> binary vessel mask. Returns count."""
    from pycocotools.coco import COCO
    dupes = duplicate_basenames_across_cocos(root)
    n = 0
    for jp in find_coco_jsons(root):
        coco = COCO(jp); base = os.path.dirname(jp)
        for img in coco.loadImgs(coco.getImgIds()):
            ip = resolve_image(base, img["file_name"]) or resolve_image(root, img["file_name"])
            if not ip:
                continue
            g = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            m = np.zeros((img["height"], img["width"]), np.uint8)
            for a in coco.loadAnns(coco.getAnnIds(imgIds=img["id"])):
                try:
                    m = np.maximum(m, coco.annToMask(a))
                except Exception:
                    pass
            stem = _disambiguated_stem(ip, jp, dupes)
            write_pair(g, m, stem, out_dir, size)
            if raw_dir:
                write_nnunet_case(g, m, stem, raw_dir, size)
            n += 1
    return n


def coco_to_yolo(root, out_dir, size=512, class_id=0, class_map=None):
    """COCO bbox -> YOLO txt. class_map: {coco_cat_id: yolo_idx} for multi-class; else class_id."""
    from pycocotools.coco import COCO
    dupes = duplicate_basenames_across_cocos(root)
    n = 0
    for jp in find_coco_jsons(root):
        coco = COCO(jp); base = os.path.dirname(jp)
        for img in coco.loadImgs(coco.getImgIds()):
            ip = resolve_image(base, img["file_name"]) or resolve_image(root, img["file_name"])
            if not ip:
                continue
            W, H = img["width"], img["height"]
            lines = []
            for a in coco.loadAnns(coco.getAnnIds(imgIds=img["id"])):
                cid = class_map.get(a["category_id"]) if class_map else class_id
                if cid is None:
                    continue
                x, y, w, h = a["bbox"]
                lines.append(f"{cid} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} "
                             f"{w / W:.6f} {h / H:.6f}")
            stem = _disambiguated_stem(ip, jp, dupes)
            sp = split_of(stem)
            ensure(os.path.join(out_dir, "images", sp), os.path.join(out_dir, "labels", sp))
            g = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(os.path.join(out_dir, "images", sp, stem + ".png"),
                        cv2.resize(clahe_unsharp(g), (size, size)))
            open(os.path.join(out_dir, "labels", sp, stem + ".txt"), "w").write("\n".join(lines))
            n += 1
    return n


def masks_to_yolo(img_path, class_masks, out_dir, size=512):
    """class_masks: list of (yolo_idx, binary_mask). Emit one YOLO box per connected component."""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    g = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return 0
    H, W = g.shape
    lines = []
    for idx, m in class_masks:
        num, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype("uint8"))
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < 8:
                continue
            lines.append(f"{idx} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
    if not lines:
        return 0
    sp = split_of(stem)
    ensure(os.path.join(out_dir, "images", sp), os.path.join(out_dir, "labels", sp))
    cv2.imwrite(os.path.join(out_dir, "images", sp, stem + ".png"), cv2.resize(clahe_unsharp(g), (size, size)))
    open(os.path.join(out_dir, "labels", sp, stem + ".txt"), "w").write("\n".join(lines))
    return 1


def write_yolo_datayaml(out_dir, names=("stenosis",)):
    p = os.path.join(out_dir, "data.yaml")
    with open(p, "w") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\ntrain: images/train\nval: images/val\n")
        f.write(f"nc: {len(names)}\nnames: {list(names)}\n")
    return p


def duplicate_basenames_across_cocos(root):
    """Detect image basenames that appear in MORE THAN ONE COCO json under ``root``.

    ARCADE task-2 ships train/val/test as separate folders that each renumber images ``1..N``,
    so ``5.png`` exists in all three. The YOLO converters key output files by *basename* stem, so
    pooling the three splits collapses three different physical images onto one output path
    (last-write-wins) — silent data loss AND a train/test contamination. Returns
    ``{basename: [json_paths]}`` for the colliding names only (empty dict = safe to pool).
    """
    seen = {}
    for jp in find_coco_jsons(root):
        try:
            d = json.load(open(jp))
        except Exception:
            continue
        for im in d.get("images", []):
            bn = os.path.basename(im.get("file_name", ""))
            if bn:
                seen.setdefault(bn, set()).add(jp)
    return {bn: sorted(js) for bn, js in seen.items() if len(js) > 1}


# COCO jsons commonly live in a generic container dir next to the images (ARCADE ships
# .../<split>/annotations/<split>.json). The split folder ('train'/'val'/'test') is the real
# disambiguator, so when tagging a collision we skip these container names and take the first
# meaningful ancestor dir instead.
_GENERIC_JSON_DIRS = {"annotations", "annotation", "anns", "ann", "labels", "json", "jsons", "coco"}


def _split_tag(json_path):
    """Deterministic disambiguation tag for a COCO json = its split-folder name.

    Walks up from the json, skipping generic container dirs, so both ``<root>/train/x.json``
    and ``<root>/train/annotations/x.json`` yield ``'train'``. Falls back to the json's own
    stem if no non-generic ancestor exists.
    """
    d = os.path.dirname(os.path.abspath(json_path))
    while d and d != os.path.dirname(d):
        base = os.path.basename(d)
        if base and base.lower() not in _GENERIC_JSON_DIRS:
            return base
        d = os.path.dirname(d)
    return os.path.splitext(os.path.basename(json_path))[0]


def _disambiguated_stem(basename, json_path, dupes):
    """OUTPUT stem for one image, resolving ARCADE cross-split basename collisions.

    ``dupes`` is the ``duplicate_basenames_across_cocos`` map (basename -> [json paths]). If this
    image's basename collides across COCO jsons (ARCADE's ``5.png`` in train/val/test), prefix the
    source json's split tag so the three physical images map to three DISTINCT stems
    (``train_5``/``val_5``/``test_5``) instead of clobbering one path. Non-colliding basenames keep
    their bare stem unchanged, so Danilov ``<site>_<patient>_<seq>_<frame>`` names survive intact and
    ``group_key`` can still collapse them.
    """
    bn = os.path.basename(basename)
    stem = os.path.splitext(bn)[0]
    if bn in dupes:
        return f"{_split_tag(json_path)}_{stem}"
    return stem


def _split_stems(out_dir, split):
    """Set of image stems in a YOLO split's images/<split> dir (strip extension)."""
    d = os.path.join(out_dir, "images", split)
    if not os.path.isdir(d):
        return set()
    return {os.path.splitext(f)[0] for f in os.listdir(d)
            if os.path.splitext(f)[1].lower() in
            (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")}


# SSL rounds inject frames named 'gd_<stem>' / 'pl_<stem>' into images/train. Strip the prefix
# before grouping so a self-labeled copy of a VAL patient (gd_14_002_5_0016) still collides with
# that patient in val — otherwise the prefix hides the re-leak from the auditor.
_SSL_PREFIXES = ("gd_", "pl_")


def _audit_group(stem):
    for p in _SSL_PREFIXES:
        if stem.startswith(p):
            stem = stem[len(p):]
            break
    return group_key(stem)


def audit_split_leakage(out_dir, danilov_stems=None, max_ungrouped_frac=0.5, cathaction_stems=None):
    """Post-conversion honesty gate for a YOLO train/val split. Returns a report dict;
    RAISES AssertionError the moment the split could leak. Call it AFTER conversion and
    BEFORE training so a leaked run can never silently report an inflated metric.

    Two independent failure modes are checked:

    1. Group leakage — no ``group_key`` (and no bare stem) may appear in BOTH train and val.
       This is the direct guard: near-identical consecutive video frames of one patient must
       land entirely on one side, or val F1 is inflated (the 2026-07-12 F1 0.885 signature).

    2. Silent grouping no-op — ``group_key`` only collapses Danilov frames whose names match
       ``<site>_<patient>_<seq>_<frame>``. If the real files are named differently the collapse
       silently does nothing and the split degrades back to per-frame. Pass ``danilov_stems``
       (the true set of Danilov image stems, e.g. from walking data/raw/danilov) so this can be
       detected *independently of the regex*: if more than ``max_ungrouped_frac`` of them are
       ungrouped (``group_key(stem) == stem``), the grouping is untrustworthy -> raise, because
       we cannot prove the split is honest.
    """
    train, val = _split_stems(out_dir, "train"), _split_stems(out_dir, "val")
    assert train and val, (
        f"empty split (train={len(train)}, val={len(val)}) — conversion produced no data")

    # (1a) exact-stem overlap: the same image file must never be in both splits.
    stem_overlap = train & val
    assert not stem_overlap, (
        f"LEAKAGE: {len(stem_overlap)} identical stems in BOTH train and val, "
        f"e.g. {sorted(stem_overlap)[:5]}")

    # (1b) group overlap: no patient/clip sequence may straddle the split (SSL prefixes stripped,
    #      so a self-labeled copy of a val patient injected into train is still caught).
    gtrain, gval = {_audit_group(s) for s in train}, {_audit_group(s) for s in val}
    group_overlap = gtrain & gval
    assert not group_overlap, (
        f"LEAKAGE: {len(group_overlap)} patient/clip groups span BOTH train and val, "
        f"e.g. {sorted(group_overlap)[:5]} — frames of one patient leaked across the split")

    # (2) prove the Danilov video frames were actually collapsed (not a silent regex no-op).
    danilov_report = None
    if danilov_stems is not None:
        dset = set(danilov_stems)
        d_in_split = (train | val) & dset
        ungrouped = {s for s in d_in_split if group_key(s) == s}
        frac = len(ungrouped) / max(1, len(d_in_split))
        danilov_report = {"danilov_frames": len(d_in_split),
                          "ungrouped": len(ungrouped), "ungrouped_frac": round(frac, 3),
                          "patient_groups": len({group_key(s) for s in d_in_split})}
        assert d_in_split and frac <= max_ungrouped_frac, (
            f"UNGROUPED DANILOV: {len(ungrouped)}/{len(d_in_split)} "
            f"({frac:.0%}) Danilov frames were NOT collapsed by group_key — their filenames do "
            f"not match the assumed '<site>_<patient>_<seq>_<frame>' pattern, so the split is "
            f"per-frame and WILL leak. Inspect the names and update group_key()/_PATIENT_RE "
            f"before trusting any F1. Example ungrouped: {sorted(ungrouped)[:5]}")

    # (2b) same silent-grouping-no-op guard for CathAction video frames. group_key only collapses
    #      names matching '<clip>_img-<seg>-<frame>'; if the real files are named otherwise the
    #      collapse silently does nothing and the split degrades to per-frame. Pass the true set of
    #      CathAction image stems to prove they were actually grouped by clip (independent of regex).
    cathaction_report = None
    if cathaction_stems is not None:
        cset = set(cathaction_stems)
        c_in_split = (train | val) & cset
        ungrouped = {s for s in c_in_split if group_key(s) == s}
        frac = len(ungrouped) / max(1, len(c_in_split))
        cathaction_report = {"cathaction_frames": len(c_in_split),
                             "ungrouped": len(ungrouped), "ungrouped_frac": round(frac, 3),
                             "clip_groups": len({group_key(s) for s in c_in_split})}
        assert c_in_split and frac <= max_ungrouped_frac, (
            f"UNGROUPED CATHACTION: {len(ungrouped)}/{len(c_in_split)} "
            f"({frac:.0%}) CathAction frames were NOT collapsed by group_key — their filenames do "
            f"not match the assumed '<clip>_img-<seg>-<frame>' pattern, so the split is per-frame "
            f"and WILL leak. Inspect the names and update group_key()/_CLIP_RE before trusting any "
            f"F1. Example ungrouped: {sorted(ungrouped)[:5]}")

    return {"train_imgs": len(train), "val_imgs": len(val),
            "train_groups": len(gtrain), "val_groups": len(gval),
            "val_frac_by_group": round(len(gval) / max(1, len(gtrain) + len(gval)), 3),
            "danilov": danilov_report, "cathaction": cathaction_report}
