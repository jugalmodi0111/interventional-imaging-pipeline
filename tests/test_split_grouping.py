"""Split-grouping guards: no video/patient leaks across train/val.

A per-frame split leaks near-identical consecutive frames of the same clip/patient into
BOTH train and val, inflating metrics. group_key() must collapse every frame of a source
sequence to one key so split_of() puts the whole sequence on one side.
"""
from src.data_prep.io_utils import group_key, split_of


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
