# tests/test_ingest_group_key.py
"""AVF patient grouping: the guard that keeps one patient out of both train and val.

Dialygo B5 splits BY PATIENT, never by image. group_key() is the single place that rule is
enforced: split_of() hashes group_key(stem), so if group_key does not collapse an AVF cine's
frames to one key, a patient's near-identical frames scatter across train AND val.

That is the documented failure behind this project's fake F1 -- PROJECT_TRACKER changelog
2026-07-12(a): F1 0.885 on a per-frame split, 0.214 after the honest patient-grouped re-split.
"""
import hashlib
import os

import pytest

from src.data_prep.io_utils import audit_split_leakage, group_key, split_of

PID = "avf_inu_3f9c21b04e"


def _patients(n=12):
    """Deterministic synthetic pseudo-patients in the locked stem grammar (no real IDs)."""
    return [f"avf_inu_{hashlib.md5(f'avf-test-patient-{i}'.encode()).hexdigest()[:10]}"
            for i in range(n)]


def _frames(pid, n_series=4, n_frames=6):
    return [f"{pid}_s{s:02d}_{f:05d}" for s in range(1, n_series + 1) for f in range(n_frames)]


def test_avf_frame_stem_collapses_to_the_patient_group():
    stem = "avf_inu_3f9c21b04e_s01_00012"
    assert group_key(stem) == PID, \
        f"AVF frame must collapse to its patient, got {group_key(stem)!r}"


def test_all_frames_of_one_patient_across_series_share_one_key():
    keys = {group_key(s) for s in _frames(PID, n_series=4, n_frames=25)}
    assert keys == {PID}, f"one patient must yield exactly one group, got {sorted(keys)}"


def test_two_avf_patients_get_distinct_keys():
    a, b = _patients(2)
    assert group_key(f"{a}_s01_00000") != group_key(f"{b}_s01_00000")
    assert group_key(f"{a}_s01_00000") == a and group_key(f"{b}_s01_00000") == b


def test_every_frame_of_a_patient_lands_in_the_same_split():
    # The B5 guarantee, over a corpus-shaped set of stems.
    stems = {pid: _frames(pid, n_series=4, n_frames=6) for pid in _patients(12)}
    total = sum(len(v) for v in stems.values())
    assert total >= 200, f"exercise at least 200 frames, got {total}"

    for pid, frames in stems.items():
        sides = {split_of(f) for f in frames}
        assert len(sides) == 1, f"patient {pid} leaked across {sorted(sides)}"


def test_split_of_agrees_for_a_bare_group_key_and_its_frames():
    # Converters call split_of(<patient>) directly in places and split_of(<frame stem>) in others;
    # the two must never disagree.
    for pid in _patients(12):
        assert split_of(pid) == split_of(f"{pid}_s02_00031"), f"{pid}: group/frame split disagree"


def test_the_grouped_split_is_not_degenerate():
    # Guards against a vacuous pass: if everything hashed to 'train', the test above proves nothing.
    assert {split_of(pid) for pid in _patients(12)} == {"train", "val"}


def test_near_miss_avf_stems_are_not_collapsed():
    # The regex is deliberately tight: over-collapsing unrelated names into one group would be the
    # mirror-image bug (a giant fake 'patient' swallowing the whole corpus).
    assert group_key("avf_inu_3f9c21b04e_s01") == "avf_inu_3f9c21b04e_s01"      # no frame index
    assert group_key("avf_inu_XYZ_s01_00012") == "avf_inu_XYZ_s01_00012"        # id is not hex10
    assert group_key("avf_inu_3f9c21b04_s01_00012") == "avf_inu_3f9c21b04_s01_00012"   # 9 hex
    assert group_key("notavf_inu_3f9c21b04e_s01_00012") == "notavf_inu_3f9c21b04e_s01_00012"


def test_existing_dataset_groupings_are_unchanged():
    # Regression: this module is imported by every converter, trainer and split auditor.
    assert group_key("14_002_5_0016") == "14_002"                       # Danilov
    assert group_key("14_002_8_0001") == "14_002"                       # Danilov, other sequence
    assert group_key("p12_v3_00045") == "p12"                           # CADICA
    assert group_key("p1_v1_0") == "p1"                                 # CADICA
    assert group_key("JFQ_j3383201_img-00000-0042") == "JFQ_j3383201"   # CathAction
    assert split_of("p12") == split_of("p12_v3_00045")                  # CADICA converter path


def test_unmatched_stems_still_return_themselves():
    assert group_key("800") == "800"              # ARCADE bare stem
    assert group_key("train_5") == "train_5"      # ARCADE-disambiguated stem
    assert group_key("") == ""


# --- audit_split_leakage: avf_stems silent-grouping-no-op guard (audit remediation P0.2) --------
# Group leakage (train & val group overlap) alone can't catch a *silent* regex no-op: a per-frame
# split has every group unique by construction, so it trivially "passes" while still leaking (the
# exact failure this task exists to close). avf_stems mirrors danilov_stems/cathaction_stems: pass
# the true set of AVF stems so the collapse is proven independently of the regex, not just trusted.

def _write_split(tmp, train_stems, val_stems):
    for split, stems in (("train", train_stems), ("val", val_stems)):
        d = os.path.join(tmp, "images", split)
        os.makedirs(d, exist_ok=True)
        for s in stems:
            open(os.path.join(d, s + ".png"), "w").close()
    return tmp


def test_avf_two_hundred_frames_one_patient_collapse_to_one_group():
    # Without the _AVF_RE branch, group_key falls through to `return name` and each of these 200
    # frames is its own group (200 groups) -- the exact corpus-shaped regression behind
    # F1 0.885 -> 0.214 (PROJECT_TRACKER 2026-07-12(a)). With the patch, one patient's frames,
    # however many series/frames they span, must collapse to exactly one group.
    frames = _frames(PID, n_series=8, n_frames=25)   # 8 * 25 = 200 frames, one patient
    assert len(frames) == 200
    assert len({group_key(f) for f in frames}) == 1


def test_audit_avf_default_none_is_backward_compat_noop(tmp_path):
    tmp = _write_split(str(tmp_path),
                       train_stems=[f"{PID}_s01_{i:05d}" for i in range(3)],
                       val_stems=[f"avf_inu_{'a' * 10}_s01_00000"])
    rep = audit_split_leakage(tmp)   # no avf_stems passed
    assert rep["avf"] is None


def test_audit_avf_grouped_frames_pass_and_report(tmp_path):
    # A whole patient's frames on one side -> group_key collapses each patient -> ungrouped == 0.
    a, b = _patients(2)
    train = [f"{a}_s{s:02d}_{f:05d}" for s in range(1, 3) for f in range(100)]
    val = [f"{b}_s{s:02d}_{f:05d}" for s in range(1, 3) for f in range(100)]
    tmp = _write_split(str(tmp_path), train_stems=train, val_stems=val)
    rep = audit_split_leakage(tmp, avf_stems=train + val)
    assert rep["avf"]["ungrouped"] == 0
    assert rep["avf"]["patient_groups"] == 2
    assert rep["avf"]["avf_frames"] == 400


def test_audit_avf_ungrouped_names_raise(tmp_path):
    # Malformed AVF-shaped names that _AVF_RE does NOT match (simulating a future regex no-op, e.g.
    # a pid that stops being 10 hex chars): group_key can't collapse them -> per-frame split -> must
    # raise, not bless it as honest, exactly as danilov_stems/cathaction_stems already do.
    bad = [f"avf_inu_notahexid_s01_{i:05d}" for i in range(20)]   # pid is not 10 hex chars
    assert all(group_key(s) == s for s in bad), "fixture must actually defeat _AVF_RE"
    tmp = _write_split(str(tmp_path), train_stems=bad[:14], val_stems=bad[14:])
    with pytest.raises(AssertionError, match="UNGROUPED AVF"):
        audit_split_leakage(tmp, avf_stems=bad)


# --- AngioCAD: the proxy corpus's own stem grammar ------------------------------------------------

def test_angiocad_frames_collapse_to_the_patient_not_the_series():
    """AngioCAD videos are one row per (patient, series) and a patient has SEVERAL series -- 277 of
    413 patients have videos on both coronary sides. Without a rule here group_key falls through to
    `return name` and groups per SERIES, scattering one patient's studies across train and val. That
    is the P0.2 / CADICA-2026-08-16 / AVF-2026-08-24 bug for a fourth time, and the proxy corpus is
    where it would land next."""
    stems = [f"angiocad_7_s{s:02d}_{f:05d}" for s in (1, 2, 9) for f in range(4)]
    assert {group_key(s) for s in stems} == {"angiocad_7"}


def test_angiocad_patients_stay_distinct():
    assert group_key("angiocad_7_s01_00000") != group_key("angiocad_70_s01_00000")
    assert group_key("angiocad_413_s07_00003") == "angiocad_413"


def test_angiocad_rule_does_not_over_collapse_foreign_names():
    """Anchored at both ends: a name that merely starts with 'angiocad' must fall through to itself
    rather than being swept into a fake patient group."""
    for odd in ("angiocad_notanumber_s01_00000", "angiocad_7_v01_00000", "angiocad_7_s01"):
        assert group_key(odd) == odd
