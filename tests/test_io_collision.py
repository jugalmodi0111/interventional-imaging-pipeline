"""ARCADE cross-split stem collision + nnU-Net numTraining + CathAction split honesty.

Torch-free and pycocotools-free by design: the collision logic is tested through the pure
helpers ``_split_tag`` / ``_disambiguated_stem`` (no images, no COCO runtime), and the
arcade ``numTraining`` fix is tested by monkeypatching ``coco_seg_to_pairs`` so ``main`` runs
without pycocotools/cv2 decode. These build only tiny COCO jsons + empty files on tmp_path.
"""
import json
import os

import pytest

from src.data_prep import io_utils as io


# --- _split_tag / _disambiguated_stem: pure collision-stem logic --------------------

def test_split_tag_uses_split_folder_directly_under_root():
    assert io._split_tag("/root/train/a.json") == "train"
    assert io._split_tag("/root/val/a.json") == "val"
    assert io._split_tag("/root/test/a.json") == "test"


def test_split_tag_skips_generic_annotations_container():
    # ARCADE ships .../<split>/annotations/<split>.json -> the container must be skipped so the
    # tag is the split folder, not the useless 'annotations' (which would collide across splits).
    assert io._split_tag("/root/train/annotations/seg_train.json") == "train"
    assert io._split_tag("/root/val/annotations/seg_val.json") == "val"


def test_disambiguated_stem_keeps_noncolliding_bare_stem():
    # No collision -> stem unchanged, so Danilov '<site>_<patient>_<seq>_<frame>' survives for group_key.
    assert io._disambiguated_stem("14_002_5_0016.png", "/x/train/a.json", {}) == "14_002_5_0016"
    assert io._disambiguated_stem("9.png", "/x/train/a.json", {"5.png": ["/a", "/b"]}) == "9"


def test_disambiguated_stem_prefixes_split_tag_on_collision():
    dupes = {"5.png": ["/root/train/annotations/a.json", "/root/val/annotations/a.json"]}
    assert io._disambiguated_stem("5.png", "/root/train/annotations/a.json", dupes) == "train_5"
    assert io._disambiguated_stem("5.png", "/root/val/annotations/a.json", dupes) == "val_5"


def test_disambiguated_stem_three_colliding_splits_stay_distinct():
    # The core data-loss fix: '5.png' in train/val/test must map to THREE distinct output stems.
    dupes = {"5.png": ["/root/train/a.json", "/root/val/a.json", "/root/test/a.json"]}
    stems = {io._disambiguated_stem("5.png", f"/root/{s}/a.json", dupes)
             for s in ("train", "val", "test")}
    assert stems == {"train_5", "val_5", "test_5"}


def _coco(path, file_names):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"images": [{"id": i, "file_name": fn} for i, fn in enumerate(file_names)],
               "annotations": [], "categories": []}, open(path, "w"))


def test_dupes_map_drives_distinct_stems_end_to_end(tmp_path):
    # Real duplicate_basenames_across_cocos map feeding _disambiguated_stem (no pycocotools/cv2).
    _coco(os.path.join(tmp_path, "train", "annotations", "a.json"), ["1.png", "5.png", "9.png"])
    _coco(os.path.join(tmp_path, "val", "annotations", "a.json"), ["1.png", "5.png"])
    _coco(os.path.join(tmp_path, "test", "annotations", "a.json"), ["5.png"])
    dupes = io.duplicate_basenames_across_cocos(str(tmp_path))

    def stem(bn, split):
        jp = os.path.join(tmp_path, split, "annotations", "a.json")
        return io._disambiguated_stem(bn, jp, dupes)

    # '5.png' collides in all three -> three distinct stems (no last-write-wins).
    assert {stem("5.png", s) for s in ("train", "val", "test")} == {"train_5", "val_5", "test_5"}
    # '1.png' collides in two -> two distinct stems.
    assert stem("1.png", "train") == "train_1" and stem("1.png", "val") == "val_1"
    # '9.png' is unique -> bare stem preserved.
    assert stem("9.png", "train") == "9"


# --- arcade_to_coco.main: numTraining counts disk, not the returned n ---------------

def test_arcade_main_numtraining_globs_disk_not_returned_n(tmp_path, monkeypatch):
    import src.data_prep.arcade_to_coco as A

    # Stand in for coco_seg_to_pairs: land 3 distinct disambiguated cases on disk, but RETURN an
    # inflated 9 (as the pre-fix code did when duplicates overwrote each other yet were counted).
    def fake_pairs(root, out_dir, size=512, raw_dir=None):
        imtr = os.path.join(raw_dir, "imagesTr")
        os.makedirs(imtr, exist_ok=True)
        for st in ("train_5", "val_5", "test_5"):
            open(os.path.join(imtr, f"{st}_0000.png"), "w").close()
        return 9

    monkeypatch.setattr(A.io, "coco_seg_to_pairs", fake_pairs)
    monkeypatch.setenv("nnUNet_raw", str(tmp_path / "raw"))
    monkeypatch.chdir(tmp_path)   # out_dir is relative in main(); keep it under tmp
    A.main({"datasets": {"arcade": {"root": str(tmp_path)}}, "preprocess": {"size": 64}})

    ds = json.load(open(tmp_path / "raw" / "Dataset001_Coronary" / "dataset.json"))
    assert ds["numTraining"] == 3, "numTraining must be the on-disk case count, not the returned n"


# --- audit_split_leakage: CathAction ungrouped-fraction guard ------------------------

def _write_split(tmp, train_stems, val_stems):
    for split, stems in (("train", train_stems), ("val", val_stems)):
        d = os.path.join(tmp, "images", split)
        os.makedirs(d, exist_ok=True)
        for s in stems:
            open(os.path.join(d, s + ".png"), "w").close()
    return tmp


def test_audit_cathaction_default_none_is_backward_compat_noop(tmp_path):
    tmp = _write_split(str(tmp_path),
                       train_stems=["JFQ_a_img-00000-0001", "JFQ_a_img-00000-0002"],
                       val_stems=["JFQ_b_img-00000-0001"])
    rep = io.audit_split_leakage(tmp)   # no cathaction_stems passed
    assert rep["cathaction"] is None


def test_audit_cathaction_grouped_frames_pass_and_report(tmp_path):
    # Whole clip on one side -> group_key collapses each clip -> ungrouped == 0 -> passes.
    train = [f"JFQ_a_img-00000-{i:04d}" for i in range(6)]
    val = [f"JFQ_b_img-00000-{i:04d}" for i in range(6)]
    tmp = _write_split(str(tmp_path), train_stems=train, val_stems=val)
    rep = io.audit_split_leakage(tmp, cathaction_stems=train + val)
    assert rep["cathaction"]["ungrouped"] == 0
    assert rep["cathaction"]["clip_groups"] == 2
    assert rep["cathaction"]["cathaction_frames"] == 12


def test_audit_cathaction_ungrouped_names_raise(tmp_path):
    # Files NOT matching '<clip>_img-<seg>-<frame>' -> group_key can't collapse -> per-frame -> raise.
    bad = [f"clipXframe{i:04d}" for i in range(20)]     # no _img-<seg>-<frame> -> each its own group
    tmp = _write_split(str(tmp_path), train_stems=bad[:14], val_stems=bad[14:])
    with pytest.raises(AssertionError, match="UNGROUPED CATHACTION"):
        io.audit_split_leakage(tmp, cathaction_stems=bad)
