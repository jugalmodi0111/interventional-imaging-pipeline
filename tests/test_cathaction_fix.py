"""Regression tests for the four silently-wrong-catheter-data bugs in cathaction_to_yolo.

Pure helpers are tested in isolation (stdlib only). The two mask-path bugs are also exercised
end-to-end on tiny synthetic PNGs written with cv2 (small arrays, no real dataset needed).
"""
import glob
import os

import cv2
import numpy as np
import pytest

from src.data_prep import cathaction_to_yolo as c

NAMES = ("catheter", "guidewire")


# ---- Bug 4: _mask_dirs substring matching (classify_mask_dir) -------------------------------------
def test_classify_mask_dir_single_class():
    assert c.classify_mask_dir("catheter", NAMES) == 0
    assert c.classify_mask_dir("guidewire", NAMES) == 1
    assert c.classify_mask_dir("Guidewire_masks", NAMES) == 1     # token + case-insensitive
    assert c.classify_mask_dir("catheter-labels", NAMES) == 0


def test_classify_mask_dir_ambiguous_and_none():
    # the reported bug: a dir named for BOTH classes must map to neither (not double-count)
    assert c.classify_mask_dir("catheter_guidewire", NAMES) is None
    assert c.classify_mask_dir("images", NAMES) is None
    assert c.classify_mask_dir("scatheterx", NAMES) is None       # substring, not a whole token


# ---- Bug 2: COCO class map built from ids instead of names (coco_classmap_by_name) ----------------
def test_coco_classmap_by_name_zero_indexed():
    cats = [{"id": 0, "name": "catheter"}, {"id": 1, "name": "guidewire"}]
    # old {i+1:i} map assumed ids 1,2 -> would drop id 0 and mislabel id 1 as guidewire
    assert c.coco_classmap_by_name(cats, NAMES) == {0: 0, 1: 1}


def test_coco_classmap_by_name_reversed_sparse_and_case():
    cats = [{"id": 7, "name": "Guidewire"}, {"id": 3, "name": "CATHETER"}]
    assert c.coco_classmap_by_name(cats, NAMES) == {7: 1, 3: 0}   # by name, not position/id


def test_coco_classmap_by_name_skips_unknown():
    cats = [{"id": 1, "name": "catheter"}, {"id": 2, "name": "vessel"}]
    m = c.coco_classmap_by_name(cats, NAMES)
    assert m == {1: 0}                                            # unknown 'vessel' dropped, not kept
    assert 2 not in m


# ---- Bug 3: binary/merged masks collapse to class 0 (mask_value_to_class) -------------------------
def test_mask_value_to_class_class_coded():
    assert c.mask_value_to_class([1, 2], NAMES) == {1: 0, 2: 1}
    # guidewire-only frame keeps class 1 -- must NOT become catheter(0)
    assert c.mask_value_to_class([2], NAMES) == {2: 1}
    assert c.mask_value_to_class([0], NAMES) == {}                # background only


def test_mask_value_to_class_binary_raises():
    # a 0/255 binary mask carries no class info -> loud failure, never a silent class-0 catheter box
    with pytest.raises(ValueError):
        c.mask_value_to_class([255], NAMES)
    with pytest.raises(ValueError):
        c.mask_value_to_class([1, 2, 255], NAMES)                # mixed/corrupt coding


# ---- Bug 1: only the first img/mask dir pair processed (pair_img_mask_dirs) -----------------------
def test_pair_img_mask_dirs_matches_all_clips_by_parent():
    dirs = [
        "/data/clipB/img", "/data/clipA/mask",
        "/data/clipA/img", "/data/clipB/mask",
    ]
    pairs = c.pair_img_mask_dirs(dirs)
    assert pairs == [("/data/clipA/img", "/data/clipA/mask"),
                     ("/data/clipB/img", "/data/clipB/mask")]     # paired by parent, not by index


def test_pair_img_mask_dirs_drops_unmatched():
    dirs = ["/data/clipA/img", "/data/clipA/mask", "/data/clipB/img", "/data/other/notes"]
    assert c.pair_img_mask_dirs(dirs) == [("/data/clipA/img", "/data/clipA/mask")]


# ---- end-to-end helpers --------------------------------------------------------------------------
def _write_img(path, val=120, size=20):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.full((size, size), val, np.uint8))


def _write_mask(path, blocks, size=20):
    """blocks: list of (value, (y0,y1,x0,x1)) filled rectangles into a single merged mask."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    m = np.zeros((size, size), np.uint8)
    for val, (y0, y1, x0, x1) in blocks:
        m[y0:y1, x0:x1] = val
    cv2.imwrite(path, m)


def _all_label_lines(out_dir):
    lines = {}
    for lp in glob.glob(os.path.join(out_dir, "labels", "*", "*.txt")):
        stem = os.path.splitext(os.path.basename(lp))[0]
        with open(lp) as f:
            lines[stem] = [ln for ln in f.read().splitlines() if ln.strip()]
    return lines


# ---- Bug 1 end-to-end: every clip converted, not just the first ----------------------------------
def test_from_img_mask_pairs_processes_all_clips(tmp_path, monkeypatch):
    root = tmp_path / "cathaction"
    out = tmp_path / "out"
    monkeypatch.setattr(c, "OUT", str(out))
    for clip in ("clipA", "clipB"):
        _write_img(str(root / clip / "img" / f"{clip}_f0.png"))
        _write_mask(str(root / clip / "mask" / f"{clip}_f0_mask.png"),
                    [(1, (2, 10, 2, 10)), (2, (12, 18, 12, 18))])  # catheter + guidewire coded

    n = c._from_img_mask_pairs(str(root), size=20)

    assert n == 2, "both clips must be converted, not only the first"
    labels = _all_label_lines(str(out))
    assert set(labels) == {"clipA_f0", "clipB_f0"}
    for stem, lns in labels.items():
        classes = sorted(int(ln.split()[0]) for ln in lns)
        assert classes == [0, 1], f"{stem}: expected one catheter + one guidewire box"


# ---- Bug 3 end-to-end: guidewire-only value-coded; binary mask NOT defaulted to catheter ----------
def test_from_img_mask_pairs_guidewire_only_and_binary(tmp_path, monkeypatch):
    root = tmp_path / "cathaction"
    out = tmp_path / "out"
    monkeypatch.setattr(c, "OUT", str(out))
    clip = root / "clip"
    _write_img(str(clip / "img" / "gw.png"))
    _write_img(str(clip / "img" / "bin.png"))
    _write_mask(str(clip / "mask" / "gw_mask.png"), [(2, (3, 15, 3, 15))])   # guidewire-only
    _write_mask(str(clip / "mask" / "bin_mask.png"), [(255, (3, 15, 3, 15))])  # binary/merged

    n = c._from_img_mask_pairs(str(root), size=20)

    labels = _all_label_lines(str(out))
    # guidewire-only frame -> exactly one class-1 box (never mislabelled catheter)
    assert labels.get("gw") == ["1 0.450000 0.450000 0.600000 0.600000"]
    # ambiguous binary frame is skipped loudly, NOT emitted as a class-0 catheter box
    assert "bin" not in labels
    assert n == 1
