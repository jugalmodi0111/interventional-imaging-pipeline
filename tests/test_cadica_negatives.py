"""CADICA negative (background) frame sampling.

2026-08-16 root cause (docs/STENOSIS_ARCHITECTURE_AUDIT.md A1): the stenosis training corpus held
ZERO label-less images -- every frame carried at least one stenosis box -- so the detector had no
gradient signal teaching it NOT to fire and learned "always propose a box" (per-video false-flag
rate ~1.0, negative Youden J). ``cadica_to_yolo._iter_frames`` dropped every non-lesion video whole
(they ship no ``groundtruth`` dir) and CADICA is the only source that actually holds negatives.

The fix samples frames from NON-LESION videos and writes them with EMPTY label files. The hard
correctness constraint (audit A1a) is what most of this file guards: an unannotated frame inside a
LESION video is NOT a negative -- the lesion is physically present, CADICA just annotates keyframes.
Using those as background would teach the model to suppress true positives, i.e. destroy recall.
The partition must therefore be on the VIDEO's lesion/non-lesion label, never on per-frame
annotation presence.
"""
import ast
import os

import cv2
import numpy as np
import pytest
import yaml

from src.data_prep import cadica_to_yolo as c2y

GT_LINE = "100 100 40 40 p20_50"
FRAMES = 9                      # per video; 9 makes evenly-spaced k=3 land exactly on 0/4/8
KEYFRAMES = (0, 4, 8)           # the only annotated frames of a LESION video
PATIENTS = tuple(f"p{i}" for i in range(1, 9))     # p1 -> val, p2..p8 -> train (split_of)


# --------------------------------------------------------------------------------------------
# fixtures: a real-shaped CADICA tree with BOTH lesion and non-lesion videos
# --------------------------------------------------------------------------------------------
def _write_video(root, patient, video, frames=FRAMES, gt_frames=None, gt_dir_name="groundtruth"):
    """One CADICA video: ``<root>/<patient>/<video>/input/<patient>_<video>_<NNNNN>.png``.

    ``gt_frames=None`` -> NO groundtruth dir at all (CADICA's real non-lesion video shape).
    ``gt_frames=()``   -> an EMPTY groundtruth dir (mirror artifact; still not a lesion).
    Otherwise a ``.txt`` is written for each listed frame index (keyframe annotation).
    """
    vdir = os.path.join(root, patient, video)
    indir = os.path.join(vdir, "input")
    os.makedirs(indir, exist_ok=True)
    stems = []
    for f in range(frames):
        stem = f"{patient}_{video}_{f:05d}"
        stems.append(stem)
        cv2.imwrite(os.path.join(indir, stem + ".png"), np.full((512, 512), 128, np.uint8))
    if gt_frames is not None:
        gtdir = os.path.join(vdir, gt_dir_name)
        os.makedirs(gtdir, exist_ok=True)
        for f in gt_frames:
            with open(os.path.join(gtdir, f"{patient}_{video}_{f:05d}.txt"), "w") as fh:
                fh.write(GT_LINE + "\n")
    return stems


def _write_manifest(root, patient, name, entries):
    with open(os.path.join(root, patient, name), "w") as f:
        f.write("\n".join(entries) + "\n")


def _tree(root, patients=PATIENTS, manifests=False, empty_gt_dir=False):
    """8 patients x {v1 = lesion (3 keyframes of 9 frames), v2 = non-lesion (9 frames)}.

    ``manifests=True`` also ships CADICA's authoritative per-patient
    ``lesionVideos.txt`` / ``nonlesionVideos.txt``.
    """
    os.makedirs(root, exist_ok=True)
    for p in patients:
        _write_video(root, p, "v1", gt_frames=KEYFRAMES)
        _write_video(root, p, "v2", gt_frames=() if empty_gt_dir else None)
        if manifests:
            _write_manifest(root, p, "lesionVideos.txt", ["v1"])
            _write_manifest(root, p, "nonlesionVideos.txt", ["v2"])
    return root


def _cfg(root, **cadica):
    d = {"root": root}
    d.update(cadica)
    return {"datasets": {"cadica": d}, "model": {"imgsz": 64}}


def _labels(out, split):
    d = os.path.join(out, "labels", split)
    if not os.path.isdir(d):
        return {}
    out_map = {}
    for f in sorted(os.listdir(d)):
        with open(os.path.join(d, f)) as fh:
            out_map[os.path.splitext(f)[0]] = fh.read()
    return out_map


def _backgrounds(out):
    """{stem: split} for every written label file holding NO boxes (ultralytics' 'background')."""
    return {s: sp for sp in ("train", "val")
            for s, txt in _labels(out, sp).items() if not txt.strip()}


def _positives(out):
    return {s: sp for sp in ("train", "val")
            for s, txt in _labels(out, sp).items() if txt.strip()}


def _images(out):
    return {s: sp for sp in ("train", "val")
            for s in (os.path.splitext(f)[0]
                      for f in os.listdir(os.path.join(out, "images", sp)))}


def _run(tmp_path, name="proc", monkeypatch=None, cfg=None, root=None):
    out = str(tmp_path / name)
    monkeypatch.setattr(c2y, "OUT", out)
    c2y.main(cfg if cfg is not None else _cfg(root))
    return out


# --------------------------------------------------------------------------------------------
# THE correctness constraint: negatives come only from NON-LESION videos
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("manifests", [True, False])
def test_negatives_come_only_from_non_lesion_videos(tmp_path, monkeypatch, manifests):
    src = _tree(str(tmp_path / "raw"), manifests=manifests)
    out = _run(tmp_path, monkeypatch=monkeypatch, root=src)

    bg = _backgrounds(out)
    assert bg, "no background frames written -- the corpus still has zero negatives"
    # every background must come from v2 (the non-lesion video), never from v1
    assert all("_v2_" in s for s in bg), sorted(s for s in bg if "_v2_" not in s)[:5]


def test_unannotated_frames_of_a_lesion_video_are_never_negatives(tmp_path, monkeypatch):
    # v1 has 9 frames but only 3 keyframes: frames 1,2,3,5,6,7 are UNANNOTATED yet the lesion is
    # physically present. Using them as background destroys recall (audit A1a).
    src = _tree(str(tmp_path / "raw"), manifests=True)
    out = _run(tmp_path, monkeypatch=monkeypatch, root=src)

    unannotated = {f"{p}_v1_{f:05d}" for p in PATIENTS for f in range(FRAMES) if f not in KEYFRAMES}
    assert not (set(_backgrounds(out)) & unannotated)
    assert not (set(_images(out)) & unannotated), "unannotated lesion frames must not be written"


def test_manifest_lesion_label_beats_missing_groundtruth_dir(tmp_path, monkeypatch):
    # p2/v2 has NO groundtruth dir (the fallback would call it non-lesion) but the authoritative
    # manifest says it IS a lesion video. The manifest must win, or a lesion clip becomes background.
    src = _tree(str(tmp_path / "raw"), manifests=True)
    _write_manifest(src, "p2", "lesionVideos.txt", ["v1", "v2"])
    _write_manifest(src, "p2", "nonlesionVideos.txt", [])
    out = _run(tmp_path, monkeypatch=monkeypatch, root=src)

    assert not any(s.startswith("p2_v2_") for s in _backgrounds(out))
    assert any(s.startswith("p3_v2_") for s in _backgrounds(out)), "other patients still sampled"


def test_video_listed_in_neither_manifest_is_not_used_as_a_negative(tmp_path, monkeypatch):
    # Manifests are present but do not mention p2/v3 -> label UNKNOWN. Fail closed: an unknown
    # video is not proven lesion-free, so it must not be handed to the model as background.
    src = _tree(str(tmp_path / "raw"), manifests=True)
    _write_video(src, "p2", "v3", gt_frames=None)
    out = _run(tmp_path, monkeypatch=monkeypatch, root=src)

    assert not any(s.startswith("p2_v3_") for s in _backgrounds(out))


# --------------------------------------------------------------------------------------------
# manifest parsing: both entry forms, case-insensitive, and the fallback
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("entry", ["v2", "p3_v2", "V2", "P3_V2"])
def test_manifest_entry_forms_are_honoured(tmp_path, entry):
    root = str(tmp_path / "raw")
    _write_video(root, "p3", "v2", gt_frames=None)
    _write_manifest(root, "p3", "nonlesionVideos.txt", [entry])
    _write_manifest(root, "p3", "lesionVideos.txt", ["v1"])

    label, source = c2y.cadica_video_label(os.path.join(root, "p3"), "p3", "v2", has_gt=False)
    assert (label, source) == ("nonlesion", "manifest")


@pytest.mark.parametrize("fname", ["nonlesionvideos.txt", "NonLesionVideos.txt", "NONLESIONVIDEOS.TXT"])
def test_manifest_filename_lookup_is_case_insensitive(tmp_path, fname):
    root = str(tmp_path / "raw")
    _write_video(root, "p3", "v2", gt_frames=None)
    _write_manifest(root, "p3", fname, ["v2"])

    label, source = c2y.cadica_video_label(os.path.join(root, "p3"), "p3", "v2", has_gt=False)
    assert (label, source) == ("nonlesion", "manifest")


def test_manifest_marks_a_video_lesion_even_without_groundtruth(tmp_path):
    root = str(tmp_path / "raw")
    _write_video(root, "p3", "v2", gt_frames=None)
    _write_manifest(root, "p3", "lesionVideos.txt", ["p3_v2"])

    assert c2y.cadica_video_label(os.path.join(root, "p3"), "p3", "v2", has_gt=False) == \
        ("lesion", "manifest")


def test_fallback_when_manifests_absent_uses_groundtruth_presence(tmp_path):
    root = str(tmp_path / "raw")
    _write_video(root, "p3", "v1", gt_frames=KEYFRAMES)
    _write_video(root, "p3", "v2", gt_frames=None)
    pdir = os.path.join(root, "p3")

    assert c2y.cadica_video_label(pdir, "p3", "v1", has_gt=True) == ("lesion", "groundtruth")
    assert c2y.cadica_video_label(pdir, "p3", "v2", has_gt=False) == ("nonlesion", "groundtruth")


def test_groundtruth_dir_with_no_txt_files_is_non_lesion(tmp_path, monkeypatch):
    # A mirror that ships an EMPTY groundtruth dir must not defeat the fallback.
    src = _tree(str(tmp_path / "raw"), empty_gt_dir=True)
    out = _run(tmp_path, monkeypatch=monkeypatch, root=src)

    assert any(s.startswith("p1_v2_") for s in _backgrounds(out))


def test_label_source_is_reported(tmp_path, monkeypatch, capsys):
    src = _tree(str(tmp_path / "raw"), manifests=True)
    _run(tmp_path, monkeypatch=monkeypatch, root=src)
    assert "manifest" in capsys.readouterr().out.lower()

    src2 = _tree(str(tmp_path / "raw2"))
    _run(tmp_path, name="proc2", monkeypatch=monkeypatch, root=src2)
    printed = capsys.readouterr().out.lower()
    assert "groundtruth" in printed and "manifests absent" in printed


# --------------------------------------------------------------------------------------------
# what gets written: empty labels, same split side, no stem collisions
# --------------------------------------------------------------------------------------------
def test_negative_label_files_are_genuinely_empty(tmp_path, monkeypatch):
    src = _tree(str(tmp_path / "raw"), manifests=True)
    out = _run(tmp_path, monkeypatch=monkeypatch, root=src)

    bg = _backgrounds(out)
    assert bg
    for stem, sp in bg.items():
        p = os.path.join(out, "labels", sp, stem + ".txt")
        assert os.path.getsize(p) == 0, f"{p} is not 0 bytes -- ultralytics may not count it"
        assert os.path.isfile(os.path.join(out, "images", sp, stem + ".png"))


def test_negatives_land_on_the_same_split_side_as_their_patients_positives(tmp_path, monkeypatch):
    src = _tree(str(tmp_path / "raw"), manifests=True)
    out = _run(tmp_path, monkeypatch=monkeypatch, root=src)

    from src.data_prep.io_utils import split_of
    sides = {**_positives(out), **_backgrounds(out)}
    for stem, sp in sides.items():
        patient = stem.split("_")[0]
        assert sp == split_of(patient), f"{stem} landed on {sp}, patient {patient} splits elsewhere"
    # p1 hashes to val: prove a val patient really got negatives (not just train ones)
    assert any(s.startswith("p1_") for s, sp in _backgrounds(out).items() if sp == "val")


def test_negative_stems_never_collide_with_positive_stems(tmp_path, monkeypatch):
    src = _tree(str(tmp_path / "raw"), manifests=True)
    out = _run(tmp_path, monkeypatch=monkeypatch, root=src)

    assert not (set(_positives(out)) & set(_backgrounds(out)))
    # and the stems still match group_key's CADICA pattern, or the audit's grouping degrades
    from src.data_prep.io_utils import group_key
    for s in _backgrounds(out):
        assert group_key(s) == s.split("_")[0], s


def test_backgrounds_reach_the_audit_report(tmp_path, monkeypatch, capsys):
    src = _tree(str(tmp_path / "raw"), manifests=True)
    _run(tmp_path, monkeypatch=monkeypatch, root=src)
    printed = capsys.readouterr().out
    assert "LEAKAGE CHECK PASSED" in printed, printed


# --------------------------------------------------------------------------------------------
# sampling policy: ratio, spread across patients, even spacing, determinism
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("npp,expected", [(1.0, 24), (0.5, 12), (2.0, 48)])
def test_negatives_per_positive_controls_the_count(tmp_path, monkeypatch, npp, expected):
    src = _tree(str(tmp_path / "raw"), manifests=True)
    out = _run(tmp_path, monkeypatch=monkeypatch,
               cfg=_cfg(src, negatives_per_positive=npp))

    assert len(_positives(out)) == 24               # 8 patients x 3 keyframes
    assert len(_backgrounds(out)) == expected


def test_negatives_are_spread_across_patients_not_drawn_from_one(tmp_path, monkeypatch):
    src = _tree(str(tmp_path / "raw"), manifests=True)
    out = _run(tmp_path, monkeypatch=monkeypatch, root=src)

    per_patient = {}
    for s in _backgrounds(out):
        per_patient[s.split("_")[0]] = per_patient.get(s.split("_")[0], 0) + 1
    assert set(per_patient) == set(PATIENTS), per_patient
    assert max(per_patient.values()) - min(per_patient.values()) <= 1, per_patient


def test_negatives_are_evenly_spaced_within_each_video(tmp_path, monkeypatch):
    src = _tree(str(tmp_path / "raw"), manifests=True)
    out = _run(tmp_path, monkeypatch=monkeypatch, root=src)

    picked = sorted(int(s.split("_")[-1]) for s in _backgrounds(out) if s.startswith("p1_"))
    assert picked == [0, 4, 8], picked          # spans the clip, not the first three frames


def test_selection_is_deterministic_across_runs(tmp_path, monkeypatch):
    src = _tree(str(tmp_path / "raw"), manifests=True)
    a = _run(tmp_path, name="a", monkeypatch=monkeypatch, root=src)
    b = _run(tmp_path, name="b", monkeypatch=monkeypatch, root=src)

    assert _backgrounds(a) == _backgrounds(b)
    assert _positives(a) == _positives(b)


def test_converter_uses_no_rng_and_imports_no_cv2_at_module_level(tmp_path):
    """Repo invariants: converters are RNG-free, and importing this module must not need cv2."""
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "src", "data_prep", "cadica_to_yolo.py")
    tree = ast.parse(open(src).read())
    top = set()
    for node in tree.body:                       # MODULE level only; nested imports are the pattern
        if isinstance(node, ast.Import):
            top |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            top.add((node.module or "").split(".")[0])
    assert not (top & {"cv2", "numpy", "torch", "random", "ultralytics"}), top
    assert "io_utils" not in "".join(
        n.module or "" for n in tree.body if isinstance(n, ast.ImportFrom))


# --------------------------------------------------------------------------------------------
# backward compatibility + config wiring
# --------------------------------------------------------------------------------------------
def test_negatives_per_positive_zero_reproduces_the_old_behaviour(tmp_path, monkeypatch):
    src = _tree(str(tmp_path / "raw"), manifests=True)
    out = _run(tmp_path, monkeypatch=monkeypatch, cfg=_cfg(src, negatives_per_positive=0))

    assert _backgrounds(out) == {}
    assert set(_positives(out)) == {f"{p}_v1_{f:05d}" for p in PATIENTS for f in KEYFRAMES}
    assert set(_images(out)) == set(_positives(out))


def test_zero_negatives_does_not_trip_the_background_gate(tmp_path, monkeypatch, capsys):
    src = _tree(str(tmp_path / "raw"), manifests=True)
    _run(tmp_path, monkeypatch=monkeypatch, cfg=_cfg(src, negatives_per_positive=0))
    assert "LEAKAGE CHECK PASSED" in capsys.readouterr().out


def test_enabled_but_lesion_only_corpus_warns_instead_of_crashing(tmp_path, monkeypatch, capsys):
    # A tree with no non-lesion video at all (e.g. the pre-existing fixtures). There is nothing to
    # sample, so the run must say so loudly rather than assert -- but it must not silently pass.
    root = str(tmp_path / "raw")
    for p in PATIENTS:
        _write_video(root, p, "v1", gt_frames=KEYFRAMES)
    out = _run(tmp_path, monkeypatch=monkeypatch, root=root)

    printed = capsys.readouterr().out
    assert _backgrounds(out) == {}
    assert "WARNING" in printed and "non-lesion" in printed


def test_shipped_config_enables_negative_sampling():
    cfg = yaml.safe_load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "stenosis_yolo.yaml")))
    assert cfg["datasets"]["cadica"]["negatives_per_positive"] == 1.0


# --------------------------------------------------------------------------------------------
# pure helpers (no cv2 needed)
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("m,k,expected", [
    (9, 3, [0, 4, 8]), (5, 1, [2]), (4, 4, [0, 1, 2, 3]), (3, 9, [0, 1, 2]), (5, 0, []),
])
def test_evenly_spaced_spans_the_range(m, k, expected):
    assert c2y._evenly_spaced(list(range(m)), k) == expected


def test_select_negatives_interleaves_patients_before_exhausting_one():
    recs = [("p1", "v9", f"/x/p1_v9_{i:05d}.png", None) for i in range(10)] + \
           [("p2", "v9", f"/x/p2_v9_{i:05d}.png", None) for i in range(10)]
    got = c2y._select_negatives(recs, 4)
    assert len({r[0] for r in got}) == 2 and len(got) == 4


def test_select_negatives_cannot_exceed_what_exists():
    recs = [("p1", "v1", f"/x/p1_v1_{i:05d}.png", None) for i in range(3)]
    assert len(c2y._select_negatives(recs, 99)) == 3
    assert c2y._select_negatives(recs, 0) == []
