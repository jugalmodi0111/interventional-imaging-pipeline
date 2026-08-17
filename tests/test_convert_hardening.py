"""Converter hardening from the 2026-08-16 architecture audit (docs/STENOSIS_ARCHITECTURE_AUDIT.md).

B1 — ``coco_to_yolo`` trusted the COCO json's declared ``width``/``height`` and NEVER compared them
     to the file on disk, so stale/corrupt metadata silently mis-normalized EVERY box of that image.
     The fix fails closed: when the declared dims disagree with the actual ones we cannot know which
     coordinate system the annotations are in, so the image is SKIPPED and the drop recorded — a
     mis-normalized box is worse than a dropped image.

B2 — the converters dropped unresolvable / unreadable images with a bare ``continue``; the only
     signal was a printed count nobody diffs. This mirrors the ingest hardening
     (``src/ingest/index_dicom.py`` writes ``index_errors.jsonl`` with ``{path, reason}`` rows):
     every dropped image is now a row in ``<out_dir>/convert_errors.jsonl`` with a reason that
     distinguishes ``image_unresolved`` / ``unreadable`` / ``dim_mismatch``, and the converters
     report the drop count in their printed summary.

B4 — the ARCADE output stem depended on WHICH COCO jsons happened to be attached (the split tag was
     added only on a cross-json basename collision), so the same physical image changed stem, hence
     ``split_of`` hash, hence SPLIT, between dataset configurations. The fix applies the split tag
     iff ``group_key(stem) == stem``: ARCADE's bare numeric stems stabilise, while the sequence
     stems (Danilov / CADICA / CathAction / AVF) are left untouched so ``group_key`` still collapses
     them to a patient/clip. Prefixing those would give every frame its own group -> per-frame split
     -> the project's signature leak (F1 0.885 -> 0.214, PROJECT_TRACKER 2026-07-12(a)).

pycocotools is not installed locally, so the COCO runtime is faked through ``sys.modules``
(``coco_to_yolo`` imports it inside the function, so the fake is picked up at call time).
"""
import glob
import json
import os
import sys
import types

import cv2
import numpy as np
import pytest

from src.data_prep import cadica_to_yolo as c2y
from src.data_prep import danilov_to_yolo as dan
from src.data_prep import io_utils as io
from src.data_prep.io_utils import group_key, split_of


# ==================================================================================================
# fakes / fixtures
# ==================================================================================================
class _FakeCOCO:
    """Minimal stand-in for ``pycocotools.coco.COCO`` (not installed locally)."""

    def __init__(self, json_path):
        d = json.load(open(json_path))
        self._imgs = {im["id"]: im for im in d.get("images", [])}
        self._anns = {a["id"]: a for a in d.get("annotations", [])}

    def getImgIds(self):
        return sorted(self._imgs)

    def loadImgs(self, ids):
        return [self._imgs[i] for i in ids]

    def getAnnIds(self, imgIds=None):
        return sorted(i for i, a in self._anns.items() if a["image_id"] == imgIds)

    def loadAnns(self, ids):
        return [self._anns[i] for i in ids]


@pytest.fixture
def fake_coco(monkeypatch):
    coco_mod = types.ModuleType("pycocotools.coco")
    coco_mod.COCO = _FakeCOCO
    pkg = types.ModuleType("pycocotools")
    pkg.coco = coco_mod
    monkeypatch.setitem(sys.modules, "pycocotools", pkg)
    monkeypatch.setitem(sys.modules, "pycocotools.coco", coco_mod)
    return _FakeCOCO


def _png(path, w, h, value=128):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.full((h, w), value, np.uint8))


def _garbage(path):
    """A file with an image extension that cv2 cannot decode -> imread returns None."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"not an image at all")


def _write_coco(root, split, images, annotations=()):
    """ARCADE-shaped tree: ``<root>/<split>/annotations/<split>.json`` + ``<root>/<split>/images/``.

    ``images`` items: ``{"file_name", "declared": (W, H)}`` plus ONE of
    ``"actual": (W, H)`` (write a real PNG of that size), ``"garbage": True`` (undecodable file),
    or neither (no file on disk at all -> unresolvable).
    """
    ann_dir = os.path.join(root, split, "annotations")
    img_dir = os.path.join(root, split, "images")
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    recs = []
    for i, spec in enumerate(images, start=1):
        fn = spec["file_name"]
        dw, dh = spec["declared"]
        recs.append({"id": i, "file_name": fn, "width": dw, "height": dh})
        if spec.get("garbage"):
            _garbage(os.path.join(img_dir, fn))
        elif spec.get("actual"):
            aw, ah = spec["actual"]
            _png(os.path.join(img_dir, fn), aw, ah)
    jp = os.path.join(ann_dir, f"{split}.json")
    json.dump({"images": recs, "annotations": list(annotations),
               "categories": [{"id": 1, "name": "stenosis"}]}, open(jp, "w"))
    return jp


def _drops(out_dir):
    p = os.path.join(out_dir, "convert_errors.jsonl")
    if not os.path.isfile(p):
        return []
    return [json.loads(ln) for ln in open(p) if ln.strip()]


def _written(out_dir):
    """{stem: split} for every image the converter actually wrote."""
    return {os.path.splitext(os.path.basename(p))[0]: os.path.basename(os.path.dirname(p))
            for p in glob.glob(os.path.join(out_dir, "images", "*", "*.png"))}


def _labels(out_dir):
    return {os.path.splitext(os.path.basename(p))[0]: open(p).read().strip()
            for p in glob.glob(os.path.join(out_dir, "labels", "*", "*.txt"))}


# ==================================================================================================
# B1 — the COCO json's declared dimensions must be corroborated against the file on disk
# ==================================================================================================
def test_coco_declared_dims_that_disagree_with_the_file_drop_the_image(tmp_path, fake_coco):
    # The json claims 512x384; the file on disk is 128x64. Every box of this image would have been
    # normalized by the LIE. Fail closed: nothing written, nothing guessed.
    root, out = str(tmp_path / "raw"), str(tmp_path / "proc")
    _write_coco(root, "train",
                [{"file_name": "1.png", "declared": (512, 384), "actual": (128, 64)}],
                annotations=[{"id": 1, "image_id": 1, "category_id": 1, "bbox": [32, 16, 32, 8]}])

    n = io.coco_to_yolo(root, out, size=32)

    assert n == 0, "an image whose declared dims are unverifiable must not be counted as converted"
    assert _written(out) == {}, "no image may be written for a dimension mismatch"
    assert _labels(out) == {}, "a mis-normalized label is worse than a dropped image"


def test_coco_dim_mismatch_records_both_declared_and_actual_dims(tmp_path, fake_coco):
    root, out = str(tmp_path / "raw"), str(tmp_path / "proc")
    _write_coco(root, "train",
                [{"file_name": "1.png", "declared": (512, 384), "actual": (128, 64)}])

    io.coco_to_yolo(root, out, size=32)

    rows = _drops(out)
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["reason"] == "dim_mismatch"
    assert row["path"].endswith(os.path.join("images", "1.png"))
    # BOTH coordinate systems recorded: an operator must be able to see which one is wrong without
    # re-opening the file, and the converter must not silently prefer either.
    assert (row["declared_w"], row["declared_h"]) == (512, 384)
    assert (row["actual_w"], row["actual_h"]) == (128, 64)


def test_coco_matching_dims_convert_and_normalize_by_the_true_size(tmp_path, fake_coco):
    # Non-square frame with all four box components distinct, so a swapped axis cannot pass.
    root, out = str(tmp_path / "raw"), str(tmp_path / "proc")
    _write_coco(root, "train",
                [{"file_name": "1.png", "declared": (128, 64), "actual": (128, 64)}],
                annotations=[{"id": 1, "image_id": 1, "category_id": 1, "bbox": [32, 16, 32, 8]}])

    n = io.coco_to_yolo(root, out, size=32)

    assert n == 1
    assert _drops(out) == []
    assert list(_labels(out).values()) == ["0 0.375000 0.312500 0.250000 0.125000"]


def test_coco_one_bad_image_does_not_stop_the_good_ones(tmp_path, fake_coco):
    root, out = str(tmp_path / "raw"), str(tmp_path / "proc")
    _write_coco(root, "train", [
        {"file_name": "1.png", "declared": (64, 64), "actual": (64, 64)},
        {"file_name": "2.png", "declared": (512, 512), "actual": (64, 64)},    # lying metadata
        {"file_name": "3.png", "declared": (64, 64), "actual": (64, 64)},
    ])

    n = io.coco_to_yolo(root, out, size=32)

    assert n == 2
    assert set(_written(out)) == {"train_1", "train_3"}
    assert [r["reason"] for r in _drops(out)] == ["dim_mismatch"]


# ==================================================================================================
# B2 — every dropped image is recorded, mirroring src/ingest/index_dicom.py's index_errors.jsonl
# ==================================================================================================
def test_coco_unresolvable_image_is_recorded(tmp_path, fake_coco):
    root, out = str(tmp_path / "raw"), str(tmp_path / "proc")
    _write_coco(root, "train", [{"file_name": "missing.png", "declared": (64, 64)}])

    n = io.coco_to_yolo(root, out, size=32)

    assert n == 0
    rows = _drops(out)
    assert [r["reason"] for r in rows] == ["image_unresolved"]
    assert "missing.png" in rows[0]["path"]


def test_coco_unreadable_image_is_recorded(tmp_path, fake_coco):
    root, out = str(tmp_path / "raw"), str(tmp_path / "proc")
    _write_coco(root, "train", [{"file_name": "1.png", "declared": (64, 64), "garbage": True}])

    n = io.coco_to_yolo(root, out, size=32)

    assert n == 0
    rows = _drops(out)
    assert [r["reason"] for r in rows] == ["unreadable"]
    assert rows[0]["path"].endswith("1.png")


def test_drop_record_lives_under_the_output_dir_and_every_row_has_path_and_reason(tmp_path,
                                                                                 fake_coco):
    root, out = str(tmp_path / "raw"), str(tmp_path / "proc")
    _write_coco(root, "train", [
        {"file_name": "missing.png", "declared": (64, 64)},
        {"file_name": "2.png", "declared": (64, 64), "garbage": True},
        {"file_name": "3.png", "declared": (99, 99), "actual": (64, 64)},
    ])

    io.coco_to_yolo(root, out, size=32)

    assert os.path.isfile(os.path.join(out, "convert_errors.jsonl"))
    rows = _drops(out)
    assert all(r.get("path") and r.get("reason") for r in rows)
    assert {r["reason"] for r in rows} == {"image_unresolved", "unreadable", "dim_mismatch"}


def test_drop_rows_accumulate_across_converters_sharing_one_out_dir(tmp_path, fake_coco):
    # danilov/cadica/cathaction all write into the SAME processed dir; a converter must never
    # truncate the record another one just wrote.
    root, out = str(tmp_path / "raw"), str(tmp_path / "proc")
    _write_coco(root, "train", [{"file_name": "1.png", "declared": (64, 64), "garbage": True}])
    io.coco_to_yolo(root, out, size=32)

    dan_root = str(tmp_path / "danilov")
    _voc_frame(dan_root, "14_050_2_0001", garbage=True)
    dan._danilov_native(dan_root, out, size=32)

    assert len(_drops(out)) == 2, "the second converter must APPEND, not truncate"


def test_io_drop_helpers_round_trip(tmp_path):
    out = str(tmp_path / "proc")
    collected = []
    io.record_drop(out, "/a/1.png", "unreadable", drops=collected)
    io.record_drop(out, "/a/2.png", "dim_mismatch", drops=collected,
                   declared_w=512, declared_h=512, actual_w=64, actual_h=64)

    assert io.convert_errors_path(out) == os.path.join(out, "convert_errors.jsonl")
    rows = io.read_drops(out)
    assert [r["path"] for r in rows] == ["/a/1.png", "/a/2.png"]
    assert rows[1]["declared_w"] == 512
    assert collected == rows, "the optional collector must mirror what was written"
    assert io.drop_reason_counts(rows) == {"dim_mismatch": 1, "unreadable": 1}


# --- Danilov native readers -----------------------------------------------------------------------
def _voc_frame(root, stem, garbage=False, w=64, h=48, missing_image=False):
    """Danilov Pascal-VOC pair: ``<stem>.bmp`` + ``<stem>.xml`` with one box."""
    os.makedirs(root, exist_ok=True)
    ip = os.path.join(root, stem + ".bmp")
    if garbage:
        _garbage(ip)
    elif not missing_image:
        _png(ip, w, h)
    with open(os.path.join(root, stem + ".xml"), "w") as f:
        f.write(f"<annotation><filename>{stem}.bmp</filename><object><bndbox>"
                f"<xmin>10</xmin><ymin>8</ymin><xmax>30</xmax><ymax>24</ymax>"
                f"</bndbox></object></annotation>")
    return ip


def test_danilov_native_records_unreadable_frame(tmp_path):
    root, out = str(tmp_path / "danilov"), str(tmp_path / "proc")
    _voc_frame(root, "14_050_2_0001")
    bad = _voc_frame(root, "14_050_2_0002", garbage=True)

    n = dan._danilov_native(root, out, size=32)

    assert n == 1, "the unreadable frame must not be counted"
    rows = _drops(out)
    assert [r["reason"] for r in rows] == ["unreadable"]
    assert rows[0]["path"] == bad


def test_danilov_native_records_annotation_with_no_image(tmp_path):
    root, out = str(tmp_path / "danilov"), str(tmp_path / "proc")
    _voc_frame(root, "14_050_2_0001")
    _voc_frame(root, "14_050_2_0002", missing_image=True)

    n = dan._danilov_native(root, out, size=32)

    assert n == 1
    assert [r["reason"] for r in _drops(out)] == ["image_unresolved"]


def test_danilov_native_yolo_branch_records_unreadable_frame(tmp_path):
    # Branch 2 of the native reader: images with a sibling YOLO .txt (no XML anywhere).
    root, out = str(tmp_path / "danilov"), str(tmp_path / "proc")
    for stem, bad in (("14_050_2_0001", False), ("14_050_2_0002", True)):
        ip = os.path.join(root, stem + ".bmp")
        _garbage(ip) if bad else _png(ip, 64, 48)
        with open(os.path.join(root, stem + ".txt"), "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")

    n = dan._danilov_native(root, out, size=32)

    assert n == 1
    assert [r["reason"] for r in _drops(out)] == ["unreadable"]


def test_danilov_native_reports_zero_drops_on_a_clean_tree(tmp_path):
    root, out = str(tmp_path / "danilov"), str(tmp_path / "proc")
    _voc_frame(root, "14_050_2_0001")
    dan._danilov_native(root, out, size=32)
    assert _drops(out) == []


# --- CADICA ---------------------------------------------------------------------------------------
def _cadica_video(root, patient, video, frames=6, gt_frames=(), garbage_frames=()):
    vdir = os.path.join(root, patient, video)
    indir = os.path.join(vdir, "input")
    os.makedirs(indir, exist_ok=True)
    for i in range(frames):
        stem = f"{patient}_{video}_{i:05d}"
        ip = os.path.join(indir, stem + ".png")
        _garbage(ip) if i in garbage_frames else _png(ip, 64, 64)
    if gt_frames:
        gtdir = os.path.join(vdir, "groundtruth")
        os.makedirs(gtdir, exist_ok=True)
        for i in gt_frames:
            with open(os.path.join(gtdir, f"{patient}_{video}_{i:05d}.txt"), "w") as f:
                f.write("10 10 20 20 p20_50\n")


def _cadica_manifests(root, patient, lesion=(), nonlesion=()):
    for name, entries in (("lesionVideos.txt", lesion), ("nonlesionVideos.txt", nonlesion)):
        with open(os.path.join(root, patient, name), "w") as f:
            f.write("\n".join(entries) + "\n")


def test_cadica_convert_records_unreadable_positive_frame(tmp_path):
    root, out = str(tmp_path / "cadica"), str(tmp_path / "proc")
    _cadica_video(root, "p2", "v1", frames=4, gt_frames=(0, 1, 2), garbage_frames=(1,))

    res = c2y._convert(root, out, size=32, negatives_per_positive=0)

    assert len(res["positives"]) == 2
    assert [r["reason"] for r in res["drops"]] == ["unreadable"]
    assert [r["reason"] for r in _drops(out)] == ["unreadable"]


def test_cadica_convert_records_unreadable_negative_frame(tmp_path):
    root, out = str(tmp_path / "cadica"), str(tmp_path / "proc")
    _cadica_video(root, "p2", "v1", frames=4, gt_frames=(0, 1, 2))
    _cadica_video(root, "p2", "v2", frames=4, garbage_frames=(0, 1, 2, 3))
    _cadica_manifests(root, "p2", lesion=["v1"], nonlesion=["v2"])

    res = c2y._convert(root, out, size=32, negatives_per_positive=1.0)

    assert len(res["positives"]) == 3
    assert res["negatives"] == [], "an undecodable background frame must not be written"
    assert len(res["drops"]) == 3 and {r["reason"] for r in res["drops"]} == {"unreadable"}


def test_cadica_convert_reports_zero_drops_on_a_clean_tree(tmp_path):
    root, out = str(tmp_path / "cadica"), str(tmp_path / "proc")
    _cadica_video(root, "p2", "v1", frames=4, gt_frames=(0, 2))
    res = c2y._convert(root, out, size=32, negatives_per_positive=0)
    assert res["drops"] == [] and _drops(out) == []


# --- printed summaries ------------------------------------------------------------------------------
def test_cadica_main_prints_the_drop_count(tmp_path, monkeypatch, capsys):
    root, out = str(tmp_path / "cadica"), str(tmp_path / "proc")
    for p in ("p1", "p2"):                       # p1 -> val, p2 -> train (split_of)
        _cadica_video(root, p, "v1", frames=4, gt_frames=(0, 1, 2), garbage_frames=(1,))
        _cadica_video(root, p, "v2", frames=4)
        _cadica_manifests(root, p, lesion=["v1"], nonlesion=["v2"])
    monkeypatch.setattr(c2y, "OUT", out)

    c2y.main({"datasets": {"cadica": {"root": root}}, "model": {"imgsz": 32}})

    printed = capsys.readouterr().out
    assert "dropped" in printed.lower()
    assert "2" in printed and "unreadable" in printed
    assert "convert_errors.jsonl" in printed


def test_danilov_main_prints_the_drop_count(tmp_path, monkeypatch, capsys, fake_coco):
    root, out = str(tmp_path / "danilov"), str(tmp_path / "proc")
    _voc_frame(root, "14_050_2_0001")            # train side
    _voc_frame(root, "14_002_5_0001")            # val side
    _voc_frame(root, "14_050_2_0002", garbage=True)
    monkeypatch.setattr(dan, "OUT", out)

    dan.main({"datasets": {"danilov": {"root": root}}, "model": {"imgsz": 32}})

    printed = capsys.readouterr().out
    assert "dropped" in printed.lower()
    assert "unreadable" in printed and "convert_errors.jsonl" in printed


# ==================================================================================================
# B4 — the ARCADE stem must not depend on which COCO jsons are attached, and the sequence stems
#      must keep collapsing under group_key (prefixing them re-creates the F1 0.885 -> 0.214 leak)
# ==================================================================================================
_SEQUENCE_STEMS = [
    ("14_002_5_0016", "14_002"),                                    # Danilov
    ("p12_v3_00045", "p12"),                                        # CADICA
    ("JFQ_j3383201_img-00000-0042", "JFQ_j3383201"),                # CathAction
    ("avf_inu_3f9c21b04e_s01_00012", "avf_inu_3f9c21b04e"),         # Dialygo AVF
]


def test_arcade_bare_stem_is_tagged_even_without_a_collision():
    # Pre-fix this returned '1' when no other json was attached and 'train_1' when they were.
    assert io._disambiguated_stem("1.png", "/root/train/annotations/a.json", {}) == "train_1"


def test_arcade_stem_is_identical_whether_or_not_other_jsons_are_attached():
    jp = "/root/train/annotations/a.json"
    alone = io._disambiguated_stem("1.png", jp, {})
    with_siblings = io._disambiguated_stem(
        "1.png", jp, {"1.png": ["/root/train/annotations/a.json", "/root/val/annotations/a.json"]})
    assert alone == with_siblings == "train_1"
    assert split_of(alone) == split_of(with_siblings), "the split must not move with the config"


@pytest.mark.parametrize("stem,key", _SEQUENCE_STEMS)
def test_sequence_stems_are_never_tagged_so_group_key_still_collapses(stem, key):
    dupes = {f"{stem}.png": ["/root/train/a.json", "/root/val/a.json"]}
    for d in ({}, dupes):
        out = io._disambiguated_stem(f"{stem}.png", "/root/train/annotations/a.json", d)
        assert out == stem, "tagging a sequence stem destroys its patient/clip collapse"
        assert group_key(out) == key


@pytest.mark.parametrize("stem,key", _SEQUENCE_STEMS)
def test_all_frames_of_a_sequence_still_land_on_one_split_after_the_stem_change(stem, key):
    # The actual failure mode: per-frame stems -> per-frame groups -> a patient in BOTH splits.
    frames = _frames_of(stem)
    assert len({group_key(f) for f in frames}) == 1, "fixture must be one real sequence"
    stems = [io._disambiguated_stem(f + ".png", "/root/train/annotations/a.json", {})
             for f in frames]
    assert stems == frames, "frames must keep their bare stems"
    assert {group_key(s) for s in stems} == {key}
    assert len({split_of(s) for s in stems}) == 1, f"{key} straddles train and val"


def _frames_of(stem):
    """20 consecutive frames of the same sequence as ``stem`` (frame index is the trailing field)."""
    head, _, tail = stem.rpartition("-" if "_img-" in stem else "_")
    width = len(tail)
    sep = "-" if "_img-" in stem else "_"
    return [f"{head}{sep}{i:0{width}d}" for i in range(20)]


def test_tagging_never_fabricates_a_new_sequence_group():
    # Pathological: the json's split folder is '14' and the image is '002_5_0016.png'. Naively
    # tagging yields '14_002_5_0016', which group_key COLLAPSES to a fake patient '14_002' —
    # inventing a sequence identity the image does not have (and colliding with a real Danilov
    # patient). The bare stem must be kept instead.
    out = io._disambiguated_stem("002_5_0016.png", "/root/14/annotations/a.json", {})
    assert group_key(out) == out, f"tagging fabricated the group {group_key(out)!r}"


def test_arcade_split_is_stable_across_dataset_configurations(tmp_path, fake_coco):
    # End-to-end: the SAME physical train image must get the same stem and the same split whether
    # ARCADE's train json is attached alone or together with val+test.
    alone_root, alone_out = str(tmp_path / "alone"), str(tmp_path / "alone_proc")
    _write_coco(alone_root, "train",
                [{"file_name": f"{i}.png", "declared": (64, 64), "actual": (64, 64)}
                 for i in range(1, 6)])
    io.coco_to_yolo(alone_root, alone_out, size=32)

    all_root, all_out = str(tmp_path / "all"), str(tmp_path / "all_proc")
    for split in ("train", "val", "test"):
        _write_coco(all_root, split,
                    [{"file_name": f"{i}.png", "declared": (64, 64), "actual": (64, 64)}
                     for i in range(1, 6)])
    io.coco_to_yolo(all_root, all_out, size=32)

    alone_written = _written(alone_out)
    all_written = _written(all_out)
    assert set(alone_written) == {f"train_{i}" for i in range(1, 6)}
    assert {s: sp for s, sp in all_written.items() if s.startswith("train_")} == alone_written
    # and the three splits' images stay distinct (the original collision fix still holds)
    assert len(all_written) == 15


def test_main_warns_about_the_collision_the_rule_cannot_disambiguate(tmp_path, monkeypatch, capsys,
                                                                    fake_coco):
    # The rule's accepted residual risk: two SEQUENCE-shaped images sharing a basename across jsons
    # cannot be tagged (a tag would defeat group_key), so they still collide last-write-wins. That
    # must be said out loud — an ARCADE-shaped collision, which IS resolved, must not be.
    root, out = str(tmp_path / "raw"), str(tmp_path / "proc")
    for split in ("train", "val"):
        _write_coco(root, split, [
            {"file_name": "5.png", "declared": (64, 64), "actual": (64, 64)},                # ok
            {"file_name": "14_002_5_0001.png", "declared": (64, 64), "actual": (64, 64)},    # not
        ])
    _write_coco(root, "train", [
        {"file_name": "5.png", "declared": (64, 64), "actual": (64, 64)},
        {"file_name": "14_002_5_0001.png", "declared": (64, 64), "actual": (64, 64)},
        {"file_name": "14_050_2_0001.png", "declared": (64, 64), "actual": (64, 64)},
    ])
    monkeypatch.setattr(dan, "OUT", out)

    dan.main({"datasets": {"danilov": {"root": root}}, "model": {"imgsz": 32}})

    printed = capsys.readouterr().out
    assert "[WARN]" in printed and "14_002_5_0001" in printed
    assert "5.png" not in printed.split("[WARN]")[1], "the tagged ARCADE collision is not a loss"
    assert {"train_5", "val_5"} <= set(_written(out)), "the ARCADE-style collision is still resolved"
    collided = glob.glob(os.path.join(out, "images", "*", "14_002_5_0001.png"))
    assert len(collided) == 1, "the un-taggable collision does clobber — which is why it warns"


def test_danilov_shaped_coco_keeps_its_patient_grouped_split(tmp_path, fake_coco):
    # A Danilov export that ships COCO goes through coco_to_yolo too. Its stems must stay bare or
    # every frame becomes its own group and the patient straddles train/val.
    root, out = str(tmp_path / "raw"), str(tmp_path / "proc")
    stems = [f"14_002_5_{i:04d}" for i in range(6)] + [f"14_050_2_{i:04d}" for i in range(6)]
    _write_coco(root, "train", [{"file_name": s + ".png", "declared": (64, 64),
                                 "actual": (64, 64)} for s in stems])

    n = io.coco_to_yolo(root, out, size=32)

    written = _written(out)
    assert n == 12 and set(written) == set(stems), "stems must not be tagged"
    for key in ("14_002", "14_050"):
        splits = {sp for s, sp in written.items() if group_key(s) == key}
        assert len(splits) == 1, f"patient {key} straddles {splits}"
