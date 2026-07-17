"""Split-grouping guards: no video/patient leaks across train/val.

A per-frame split leaks near-identical consecutive frames of the same clip/patient into
BOTH train and val, inflating metrics. group_key() must collapse every frame of a source
sequence to one key so split_of() puts the whole sequence on one side.
"""
import json
import os

import pytest

from src.data_prep.cadica_to_yolo import _cap_records
from src.data_prep.io_utils import (audit_split_leakage, duplicate_basenames_across_cocos,
                                     group_key, split_of)


def _write_split(tmp, train_stems, val_stems):
    """Materialize a minimal YOLO images/{train,val} tree (empty files) for the auditor."""
    for split, stems in (("train", train_stems), ("val", val_stems)):
        d = os.path.join(tmp, "images", split)
        os.makedirs(d, exist_ok=True)
        for s in stems:
            open(os.path.join(d, s + ".png"), "w").close()
    return tmp


def test_danilov_frames_of_a_patient_share_a_group():
    # Danilov naming: <site>_<patient>_<seq>_<frame>
    frames = [f"14_002_5_{i:04d}" for i in range(20)] + [f"14_002_8_{i:04d}" for i in range(20)]
    keys = {group_key(f) for f in frames}
    assert keys == {"14_002"}, f"all frames of patient 14_002 must share one group, got {keys}"


def test_cathaction_frames_of_a_clip_share_a_group():
    # CathAction naming: <clip>_img-<seg>-<frame>, e.g. JFQ_j3383201_img-00000-0042
    frames = [f"JFQ_j3383201_img-00000-{i:04d}" for i in range(65)]
    keys = {group_key(f) for f in frames}
    assert keys == {"JFQ_j3383201"}, f"all frames of clip JFQ_j3383201 must share one group, got {keys}"


def test_two_cathaction_clips_get_distinct_groups():
    assert group_key("JFQ_j3383201_img-00000-0000") != group_key("JFQ_j3383206_img-00000-0000")


def test_arcade_numeric_names_unchanged():
    # ARCADE frames are plain integers -> each its own group (no sequence to leak)
    assert group_key("800") == "800"
    assert group_key("1000") == "1000"


# --- CADICA grouping (P1.3): pXX_vYY_NNNNN frames must collapse to patient pXX --------------------

def test_cadica_frame_groups_to_its_patient():
    assert group_key("p12_v3_00045") == "p12"
    assert group_key("p1_v1_0") == "p1"


def test_cadica_frames_of_a_patient_share_a_group():
    frames = [f"p12_v3_{i:05d}" for i in range(20)] + [f"p12_v7_{i:05d}" for i in range(20)]
    keys = {group_key(f) for f in frames}
    assert keys == {"p12"}, f"all frames of patient p12 must share one group, got {keys}"


def test_two_cadica_patients_get_distinct_groups():
    assert group_key("p12_v3_00045") != group_key("p13_v3_00045")


def test_split_of_agrees_for_cadica_patient_and_frame():
    # split_of hashes group_key(name), so a bare patient id and any of its frames must land on the
    # SAME side of the split -- this is what cadica_to_yolo._convert relies on when it calls
    # io.split_of(patient) directly instead of io.split_of(<full frame stem>).
    assert split_of("p12") == split_of("p12_v3_00045")


# --- regression: pre-existing group_key patterns must be unaffected by the CADICA branch ---------

def test_group_key_existing_patterns_unchanged():
    assert group_key("14_002_5_0016") == "14_002"                       # Danilov
    assert group_key("JFQ_j3383201_img-00000-0042") == "JFQ_j3383201"   # CathAction
    assert group_key("train_5") == "train_5"                            # ARCADE-disambiguated stem
    assert group_key("5") == "5"                                        # ARCADE bare stem


def test_no_cathaction_clip_spans_both_splits():
    # Simulate the real corpus shape: contiguous frames per clip. Group by the TRUE clip id
    # (parsed here, NOT via group_key) so this detects a leak even if group_key is wrong.
    clips = {f"JFQ_j{cid}": n for cid, n in [
        ("3383201", 65), ("3383206", 291), ("3383209", 119), ("3383233", 446),
        ("3383690", 100), ("3383752", 29), ("3383784", 36),
    ]}
    true_clip_splits = {}
    for clip, nframes in clips.items():
        for i in range(nframes):
            stem = f"{clip}_img-00000-{i:04d}"
            true_clip_splits.setdefault(clip, set()).add(split_of(stem))
    spanning = {c for c, s in true_clip_splits.items() if len(s) > 1}
    assert not spanning, f"clips leaking across train/val: {spanning}"


# --- audit_split_leakage: the notebook's pre-train honesty gate --------------------

def test_audit_passes_on_a_clean_patient_grouped_split(tmp_path):
    tmp = _write_split(str(tmp_path),
                       train_stems=[f"14_002_5_{i:04d}" for i in range(10)] + ["800", "801"],
                       val_stems=[f"14_050_2_{i:04d}" for i in range(10)] + ["900"])
    danilov = [f"14_002_5_{i:04d}" for i in range(10)] + [f"14_050_2_{i:04d}" for i in range(10)]
    rep = audit_split_leakage(tmp, danilov_stems=danilov)
    assert rep["danilov"]["ungrouped"] == 0
    assert rep["train_imgs"] == 12 and rep["val_imgs"] == 11


def test_audit_raises_when_a_patient_spans_both_splits(tmp_path):
    # Same patient 14_002 in BOTH splits -> group_key collides -> must raise (the F1 0.885 bug).
    tmp = _write_split(str(tmp_path),
                       train_stems=[f"14_002_5_{i:04d}" for i in range(5)],
                       val_stems=[f"14_002_5_{i:04d}" for i in range(5, 10)])
    with pytest.raises(AssertionError, match="span BOTH"):
        audit_split_leakage(tmp, danilov_stems=[f"14_002_5_{i:04d}" for i in range(10)])


def test_audit_catches_ssl_prefixed_val_patient_releaked_into_train(tmp_path):
    # SSL wrote a self-labeled copy of a VAL patient into train as 'pl_<stem>'. The prefix must be
    # stripped before grouping, or the re-leak hides from the auditor.
    val = [f"14_002_5_{i:04d}" for i in range(10)]
    train = [f"14_070_1_{i:04d}" for i in range(8)] + ["pl_14_002_5_0003"]   # leaked val patient
    tmp = _write_split(str(tmp_path), train_stems=train, val_stems=val)
    with pytest.raises(AssertionError, match="span BOTH"):
        audit_split_leakage(tmp)


def test_audit_raises_when_danilov_names_defeat_group_key(tmp_path):
    # Real files named unlike '<site>_<patient>_<seq>_<frame>' -> group_key can't collapse them
    # -> silent per-frame split. The auditor must catch this via the independent danilov_stems set.
    bad = [f"patient14_frame{i:04d}" for i in range(20)]           # no regex match -> each its own group
    tmp = _write_split(str(tmp_path), train_stems=bad[:14], val_stems=bad[14:])
    with pytest.raises(AssertionError, match="UNGROUPED DANILOV"):
        audit_split_leakage(tmp, danilov_stems=bad)


# --- _cap_records (P1.3): CADICA per-patient frame cap, pure selection logic ----------------------

def _cadica_rec(patient, video, i):
    stem = f"{patient}_{video}_{i:05d}"
    return (patient, video, f"/data/{patient}/{video}/input/{stem}.png",
            f"/data/{patient}/{video}/groundtruth/{stem}.txt")


def test_cap_records_limits_frames_per_patient():
    # p1 has 12 frames across 2 videos; p2 has only 3. Capping at k=5 must trim p1 to <=5 while
    # leaving the smaller patient p2 untouched (it never exceeded the cap).
    records = ([_cadica_rec("p1", "v1", i) for i in range(6)]
               + [_cadica_rec("p1", "v2", i) for i in range(6)]
               + [_cadica_rec("p2", "v1", i) for i in range(3)])

    kept = _cap_records(records, 5)

    by_patient = {}
    for patient, video, ip, gp in kept:
        by_patient.setdefault(patient, []).append((video, ip, gp))

    assert len(by_patient["p1"]) <= 5, "patient p1 must be capped to at most k=5 frames"
    assert len(by_patient["p2"]) == 3, "patient p2 has <=k frames and must keep them all"
    assert set(kept) <= set(records), "cap must only SELECT from the given records, never fabricate"


def test_cap_records_k_none_keeps_everything():
    records = [_cadica_rec("p1", "v1", i) for i in range(4)]
    kept = _cap_records(records, None)
    assert set(kept) == set(records)


# --- duplicate_basenames_across_cocos: ARCADE cross-split stem collision -------------

def _coco(path, file_names):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"images": [{"id": i, "file_name": fn} for i, fn in enumerate(file_names)],
               "annotations": [], "categories": []}, open(path, "w"))


def test_flags_arcade_style_cross_split_basename_collision(tmp_path):
    # train/val/test each renumber 1..N -> '5.png' in all three -> collision.
    _coco(os.path.join(tmp_path, "train", "annotations", "a.json"), ["1.png", "5.png", "9.png"])
    _coco(os.path.join(tmp_path, "val", "annotations", "a.json"), ["1.png", "5.png"])
    _coco(os.path.join(tmp_path, "test", "annotations", "a.json"), ["5.png"])
    dupes = duplicate_basenames_across_cocos(str(tmp_path))
    assert set(dupes) == {"1.png", "5.png"}
    assert len(dupes["5.png"]) == 3 and len(dupes["1.png"]) == 2
    assert "9.png" not in dupes


def test_no_collision_when_basenames_are_unique(tmp_path):
    _coco(os.path.join(tmp_path, "train", "a.json"), ["1.png", "2.png"])
    _coco(os.path.join(tmp_path, "val", "a.json"), ["3.png", "4.png"])
    assert duplicate_basenames_across_cocos(str(tmp_path)) == {}
