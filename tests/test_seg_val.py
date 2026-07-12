"""TDD for the held-out-val split + clDice gate helpers in src.train.train_seg.

Torch-free: these exercise ONLY the pure helpers (`split_stems`, `qualifies`). No torch / cv2
tensor work runs here — `split_stems` reuses io_utils.split_of's grouped hashing and `qualifies`
is arithmetic — so the whole file imports and runs on a torch-less box.
"""
from src.data_prep.io_utils import group_key
from src.train import train_seg as T


# ---- split_stems: patient-grouped, deterministic, non-overlapping ------------
def _danilov(patient, seq, n):
    return [f"{patient}_{seq}_{i:04d}" for i in range(n)]


def test_split_stems_empty_input_yields_two_empty_lists():
    assert T.split_stems([]) == ([], [])


def test_split_stems_is_a_pure_partition_no_overlap_full_cover():
    stems = _danilov("14_002", 5, 20) + _danilov("14_050", 2, 20) + ["800", "801", "900"]
    train, val = T.split_stems(stems)
    assert set(train) & set(val) == set()                 # non-overlapping
    assert set(train) | set(val) == set(stems)            # every stem placed exactly once


def test_split_stems_keeps_every_frame_of_a_patient_on_one_side():
    # The anti-leakage property: all frames of one patient/clip group land together.
    stems = _danilov("14_002", 5, 20) + _danilov("14_002", 8, 15) + _danilov("14_050", 2, 30)
    train, val = T.split_stems(stems)
    tset, vset = set(train), set(val)
    groups = {}
    for s in stems:
        groups.setdefault(group_key(s), set()).add("val" if s in vset else "train")
    for g, sides in groups.items():
        assert len(sides) == 1, f"group {g} straddles the split: {sides}"
    # 14_002_5_* and 14_002_8_* share group '14_002' -> must be on the SAME side.
    assert ("14_002_5_0000" in tset) == ("14_002_8_0000" in tset)


def test_split_stems_is_deterministic_across_calls():
    stems = _danilov("14_002", 5, 12) + _danilov("14_099", 1, 12) + ["42", "77"]
    assert T.split_stems(stems) == T.split_stems(stems)
    assert T.split_stems(stems) == T.split_stems(list(reversed(stems)))  # order-independent


def test_split_stems_val_frac_zero_puts_everything_in_train():
    stems = _danilov("14_002", 5, 10) + ["800"]
    train, val = T.split_stems(stems, val_frac=0.0)
    assert val == [] and set(train) == set(stems)


def test_split_stems_returns_sorted_lists():
    stems = ["900", "800", "801"]
    train, val = T.split_stems(stems)
    assert train == sorted(train) and val == sorted(val)


# ---- qualifies: Dice floor + clDice (absolute floor + teacher-relative) ------
DICE_ONLY = {"target": {"dice": 0.75}}


def test_qualifies_dice_only_pass():
    assert T.qualifies({"dice": 0.80, "cldice": 0.5}, DICE_ONLY) is True


def test_qualifies_dice_only_fail_below_floor():
    assert T.qualifies({"dice": 0.60, "cldice": 0.99}, DICE_ONLY) is False


def test_qualifies_dice_exact_floor_passes():
    assert T.qualifies({"dice": 0.75}, DICE_ONLY) is True


def test_qualifies_backward_compatible_no_cldice_key_needed():
    # teacher_scores=None and no absolute floor -> old Dice-only behaviour, cldice missing is fine.
    assert T.qualifies({"dice": 0.80}, DICE_ONLY) is True


# absolute clDice floor
def test_qualifies_absolute_cldice_floor_fails_when_below():
    cfg = {"target": {"dice": 0.75, "cldice": 0.60}}
    assert T.qualifies({"dice": 0.90, "cldice": 0.50}, cfg) is False


def test_qualifies_absolute_cldice_floor_passes_when_met():
    cfg = {"target": {"dice": 0.75, "cldice": 0.60}}
    assert T.qualifies({"dice": 0.90, "cldice": 0.65}, cfg) is True


# teacher-relative clDice tolerance
def test_qualifies_cldice_rel_teacher_pass_within_default_tolerance():
    # default tol 0.03; student 0.78 vs teacher 0.80 -> 0.78 >= 0.77 -> pass
    assert T.qualifies({"dice": 0.80, "cldice": 0.78}, DICE_ONLY,
                       teacher_scores={"cldice": 0.80}) is True


def test_qualifies_cldice_rel_teacher_fail_outside_tolerance():
    # student 0.70 vs teacher 0.80, tol 0.03 -> 0.70 < 0.77 -> fail even though Dice passes
    assert T.qualifies({"dice": 0.90, "cldice": 0.70}, DICE_ONLY,
                       teacher_scores={"cldice": 0.80}) is False


def test_qualifies_cldice_rel_teacher_exact_boundary_passes():
    # student == teacher - tol -> the >= boundary must PASS
    assert T.qualifies({"dice": 0.80, "cldice": 0.77}, DICE_ONLY,
                       teacher_scores={"cldice": 0.80}) is True


def test_qualifies_custom_cldice_rel_teacher_tolerance_is_honoured():
    cfg = {"target": {"dice": 0.75, "cldice_rel_teacher": 0.10}}
    # 0.72 vs teacher 0.80: fails at 0.03 default but passes at the configured 0.10
    assert T.qualifies({"dice": 0.80, "cldice": 0.72}, cfg,
                       teacher_scores={"cldice": 0.80}) is True
    assert T.qualifies({"dice": 0.80, "cldice": 0.72}, DICE_ONLY,
                       teacher_scores={"cldice": 0.80}) is False


def test_qualifies_teacher_gate_and_absolute_floor_both_enforced():
    # passes teacher-relative but fails absolute floor -> overall fail
    cfg = {"target": {"dice": 0.75, "cldice": 0.85, "cldice_rel_teacher": 0.03}}
    assert T.qualifies({"dice": 0.90, "cldice": 0.80}, cfg,
                       teacher_scores={"cldice": 0.80}) is False
