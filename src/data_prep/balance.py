"""Dataset-balanced oversampling for the merged stenosis YOLO train split (Stage 2 / P2.2).

ARCADE / CADICA / Danilov are pooled into one images/train + labels/train tree in unequal
amounts; if the deployment-target source is the minority it gets drowned out during training.
This module equalizes per-source frame counts by DUPLICATING minority-source frames (image +
label) under a ``bal_`` filename prefix. Duplicates are train->train only (val is never touched),
so they cannot leak across the split. The leakage auditor is taught to strip the ``bal_`` prefix
separately -- not this module's job.

Stdlib-only (shutil/os/glob): no cv2, no torch -- repo invariant is src/* must import without
either installed. balance_plan() is pure (no filesystem) so it is trivially unit-testable;
apply_balance() does the actual copying and imports source_of lazily to keep the pure helper
import-light.
"""
import glob
import os
import shutil


def balance_plan(stems, source_fn, target=None):
    """Return the flat list of stems to DUPLICATE so every source reaches ``target`` count.

    Groups ``stems`` by ``source_fn``. If ``target`` is None, uses the largest per-source count
    (every minority source is oversampled up to the majority). A source already at/above target
    contributes nothing (never downsamples). To add k copies to a source with m originals, cycles
    through that source's SORTED stems (index i -> stems[i % m]) so duplicates are spread across
    the source's frames rather than all being the same one. Fully deterministic, no RNG.
    """
    groups = {}
    for s in stems:
        groups.setdefault(source_fn(s), []).append(s)
    for src in groups:
        groups[src] = sorted(groups[src])

    if target is None:
        target = max((len(v) for v in groups.values()), default=0)

    plan = []
    for src in sorted(groups):
        m = len(groups[src])
        if m == 0 or m >= target:
            continue
        k = target - m
        plan.extend(groups[src][i % m] for i in range(k))
    return plan


def apply_balance(proc, split="train", source_fn=None, target=None):
    """Duplicate minority-source frames in ``<proc>/images/<split>`` (+ matching labels) so every
    source's frame count reaches ``target`` (default: the largest source's count).

    Each planned duplicate of stem ``orig`` is copied to a NEW stem ``bal_<n>_<orig>`` (n is a
    per-original counter, so repeated duplicates of the same original get distinct names and
    never overwrite each other or the original). Only copies a stem if BOTH its image and label
    exist. Returns {"added": int, "per_source_before": {...}, "per_source_after": {...}}.
    """
    if source_fn is None:
        from src.eval.val_by_source import source_of
        source_fn = source_of

    images_dir = os.path.join(proc, "images", split)
    labels_dir = os.path.join(proc, "labels", split)

    stems = sorted(os.path.splitext(os.path.basename(p))[0]
                    for p in glob.glob(os.path.join(images_dir, "*.png")))

    per_source_before = {}
    for s in stems:
        src = source_fn(s)
        per_source_before[src] = per_source_before.get(src, 0) + 1

    plan = balance_plan(stems, source_fn, target)

    per_source_after = dict(per_source_before)
    dup_counter = {}   # orig stem -> next duplicate index n
    added = 0
    for orig in plan:
        n = dup_counter.get(orig, 0)
        dup_counter[orig] = n + 1
        new_stem = f"bal_{n}_{orig}"

        src_img = os.path.join(images_dir, orig + ".png")
        src_lbl = os.path.join(labels_dir, orig + ".txt")
        if not (os.path.isfile(src_img) and os.path.isfile(src_lbl)):
            continue

        shutil.copyfile(src_img, os.path.join(images_dir, new_stem + ".png"))
        shutil.copyfile(src_lbl, os.path.join(labels_dir, new_stem + ".txt"))
        added += 1

        src = source_fn(orig)
        per_source_after[src] = per_source_after.get(src, 0) + 1

    return {"added": added, "per_source_before": per_source_before, "per_source_after": per_source_after}
