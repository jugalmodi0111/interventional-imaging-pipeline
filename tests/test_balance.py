"""Dataset-balanced oversampling for the merged stenosis YOLO train split (P2.2).

balance_plan is pure (no filesystem) so it must be unit-testable without torch/cv2 (repo
invariant: src/* imports torch/cv2-free). apply_balance does the actual image+label copying
under a 'bal_' prefix and must never touch val (duplicates are train->train only, so they
cannot leak).
"""
import os
from collections import Counter

from src.data_prep.balance import apply_balance, balance_plan
from src.eval.val_by_source import source_of


def _prefix_source(stem):
    """Simple, fixture-only source_fn: source is whatever precedes the first '_'."""
    return stem.split("_")[0]


# --- balance_plan: pure selection logic -----------------------------------------------------

def test_balance_plan_targets_max_source_by_default():
    # arcade=4, cadica=2, danilov=1 -> target defaults to the max (4) -> cadica +2, danilov +3,
    # arcade (already at max) +0.
    stems = ([f"arcade_{i}" for i in range(4)]
             + [f"cadica_{i}" for i in range(2)]
             + [f"danilov_{i}" for i in range(1)])

    plan = balance_plan(stems, _prefix_source)

    added = Counter(_prefix_source(s) for s in plan)
    assert added["cadica"] == 2
    assert added["danilov"] == 3
    assert added.get("arcade", 0) == 0
    assert len(plan) == 5


def test_balance_plan_duplicates_spread_across_sorted_source_stems():
    # cadica has m=2 originals and needs k=2 duplicates -> cycles stems[0%2], stems[1%2], i.e.
    # one duplicate per original stem (spread, not both piled on the same frame).
    stems = ([f"arcade_{i}" for i in range(4)]
             + [f"cadica_{i}" for i in range(2)]
             + [f"danilov_{i}" for i in range(1)])

    plan = balance_plan(stems, _prefix_source)

    cadica_added = [s for s in plan if _prefix_source(s) == "cadica"]
    assert cadica_added == ["cadica_0", "cadica_1"]

    # danilov has only m=1 original, so all k=3 duplicates necessarily cycle back onto it --
    # still deterministic (no RNG), just no other frame to spread onto.
    danilov_added = [s for s in plan if _prefix_source(s) == "danilov"]
    assert danilov_added == ["danilov_0", "danilov_0", "danilov_0"]


def test_balance_plan_cycles_through_sorted_stems_when_k_exceeds_m():
    # Deliberately unsorted input, single source, m=3, target=10 -> k=7 duplicates must cycle
    # stems[i % 3] against the SORTED stems, proving selection is order-independent and spread.
    stems = ["s2", "s0", "s1"]

    plan = balance_plan(stems, lambda _: "only", target=10)

    assert plan == ["s0", "s1", "s2", "s0", "s1", "s2", "s0"]


def test_balance_plan_never_downsamples_a_source_already_at_or_above_target():
    stems = [f"arcade_{i}" for i in range(4)]

    plan = balance_plan(stems, _prefix_source, target=2)

    assert plan == []


def test_balance_plan_explicit_target_above_max_oversamples_every_source():
    stems = [f"arcade_{i}" for i in range(2)] + [f"cadica_{i}" for i in range(2)]

    plan = balance_plan(stems, _prefix_source, target=5)

    added = Counter(_prefix_source(s) for s in plan)
    assert added["arcade"] == 3
    assert added["cadica"] == 3
    assert len(plan) == 6


def test_balance_plan_empty_input_returns_empty_plan():
    assert balance_plan([], _prefix_source) == []


# --- apply_balance: filesystem duplication under 'bal_' prefix -------------------------------

def _write_stem(images_dir, labels_dir, stem, content):
    open(os.path.join(images_dir, stem + ".png"), "w").write(content)
    open(os.path.join(labels_dir, stem + ".txt"), "w").write(content)


def test_apply_balance_equalizes_sources_and_leaves_originals_untouched(tmp_path):
    proc = str(tmp_path)
    images_dir = os.path.join(proc, "images", "train")
    labels_dir = os.path.join(proc, "labels", "train")
    os.makedirs(images_dir)
    os.makedirs(labels_dir)

    # 4 arcade frames (plain digits -> source_of == 'arcade'), 2 cadica frames (pXX_vYY_NNNNN).
    arcade_stems = ["1", "2", "3", "4"]
    cadica_stems = ["p1_v1_00000", "p1_v1_00001"]
    for stem in arcade_stems + cadica_stems:
        assert source_of(stem) in ("arcade", "cadica")  # sanity: real classifier agrees with plan
        _write_stem(images_dir, labels_dir, stem, "content-" + stem)

    result = apply_balance(proc)

    assert result["per_source_before"] == {"arcade": 4, "cadica": 2}
    assert result["per_source_after"] == {"arcade": 4, "cadica": 4}
    assert result["added"] == 2

    # cadica needed k=2 duplicates with m=2 originals -> one bal_0_ duplicate per original stem.
    expected_new = [f"bal_0_{s}" for s in cadica_stems]
    for new_stem in expected_new:
        img_path = os.path.join(images_dir, new_stem + ".png")
        lbl_path = os.path.join(labels_dir, new_stem + ".txt")
        assert os.path.isfile(img_path), f"missing duplicated image {img_path}"
        assert os.path.isfile(lbl_path), f"missing duplicated label {lbl_path}"

    # duplicate content matches its original (copy, not fabrication).
    for orig in cadica_stems:
        orig_content = open(os.path.join(images_dir, orig + ".png")).read()
        dup_content = open(os.path.join(images_dir, f"bal_0_{orig}.png")).read()
        assert dup_content == orig_content

    # no bal_ files were created for the source (arcade) already at target.
    all_images = os.listdir(images_dir)
    bal_files = [f for f in all_images if f.startswith("bal_")]
    assert len(bal_files) == 2
    assert all(source_of(f[len("bal_0_"):].rsplit(".", 1)[0]) == "cadica" for f in bal_files)

    # originals untouched: still present with original content, none renamed/removed.
    for stem in arcade_stems + cadica_stems:
        assert open(os.path.join(images_dir, stem + ".png")).read() == "content-" + stem
        assert open(os.path.join(labels_dir, stem + ".txt")).read() == "content-" + stem

    # val is never referenced or created by apply_balance.
    assert not os.path.exists(os.path.join(proc, "images", "val"))

    # total files on disk: 6 originals + 2 duplicates, images and labels in lockstep.
    assert len(os.listdir(images_dir)) == 8
    assert len(os.listdir(labels_dir)) == 8


def test_apply_balance_skips_stems_missing_a_label(tmp_path):
    proc = str(tmp_path)
    images_dir = os.path.join(proc, "images", "train")
    labels_dir = os.path.join(proc, "labels", "train")
    os.makedirs(images_dir)
    os.makedirs(labels_dir)

    # 4 arcade frames with labels; 1 cadica frame whose label is MISSING (only image exists).
    for stem in ["1", "2", "3", "4"]:
        _write_stem(images_dir, labels_dir, stem, "c-" + stem)
    open(os.path.join(images_dir, "p1_v1_00000.png"), "w").write("orphan")
    # no matching label written for p1_v1_00000

    result = apply_balance(proc)

    assert result["added"] == 0   # the only planned duplicate had no label -> skipped, not copied
    assert result["per_source_before"] == {"arcade": 4, "cadica": 1}
    assert result["per_source_after"] == {"arcade": 4, "cadica": 1}
    assert not any(f.startswith("bal_") for f in os.listdir(images_dir))


def test_apply_balance_explicit_target_and_custom_source_fn(tmp_path):
    proc = str(tmp_path)
    images_dir = os.path.join(proc, "images", "train")
    labels_dir = os.path.join(proc, "labels", "train")
    os.makedirs(images_dir)
    os.makedirs(labels_dir)

    for stem in ["a_0", "a_1", "b_0"]:
        _write_stem(images_dir, labels_dir, stem, stem)

    result = apply_balance(proc, source_fn=_prefix_source, target=3)

    assert result["per_source_before"] == {"a": 2, "b": 1}
    assert result["per_source_after"] == {"a": 3, "b": 3}
    assert result["added"] == 3
    assert sorted(f for f in os.listdir(images_dir) if f.startswith("bal_")) == [
        "bal_0_a_0.png", "bal_0_b_0.png", "bal_1_b_0.png",
    ]


# --- guard: importing the module must not require torch/cv2 ---------------------------------

def test_module_importable_without_torch_or_cv2():
    import src.data_prep.balance  # noqa: F401
