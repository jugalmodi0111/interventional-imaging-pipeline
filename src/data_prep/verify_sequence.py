"""Check whether a frame directory is ONE continuous sequence or many concatenated clips.

Ultralytics `model.track(source=<dir>)` streams frames in sorted filename order. If the dir
mixes clips, every clip boundary is a hard cut: ByteTrack kills all tracks and mints new IDs,
so `unique track ids` inflates and the ID-switch metric is meaningless. Run this before
trusting any tracking numbers from an image-dir source.

Heuristics (no dataset assumptions):
  1. Strip the trailing frame number from each stem -> distinct prefixes = candidate clips.
  2. Per prefix, check the frame numbers are contiguous (gaps = dropped/sampled frames,
     which also break temporal continuity).

Usage: python -m src.data_prep.verify_sequence data/raw/cathaction/img
"""
import os, re, sys
from collections import defaultdict

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def analyze(img_dir):
    stems = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in EXTS
    )
    if not stems:
        raise SystemExit(f"no images under {img_dir!r}")

    clips = defaultdict(list)  # prefix -> [frame numbers]
    unnumbered = 0
    for s in stems:
        m = re.match(r"^(.*?)(\d+)$", s)
        if m:
            clips[m.group(1)].append(int(m.group(2)))
        else:
            unnumbered += 1
            clips[s].append(-1)

    print(f"{len(stems)} frames | {len(clips)} distinct filename prefixes | {unnumbered} without trailing number")
    one_sequence = len(clips) == 1 and unnumbered == 0
    for prefix, nums in sorted(clips.items()):
        nums.sort()
        span = nums[-1] - nums[0] + 1
        gaps = span - len(nums)
        step = min((b - a for a, b in zip(nums, nums[1:])), default=1) or 1
        stride_ok = all((b - a) == step for a, b in zip(nums, nums[1:]))
        tag = "contiguous" if gaps == 0 else (f"uniform stride {step}" if stride_ok else f"{gaps} missing frames in span")
        if gaps and not stride_ok:
            one_sequence = False
        print(f"  {prefix or '<bare number>'!r}: {len(nums)} frames [{nums[0]}..{nums[-1]}] {tag}")

    print("VERDICT:", "one raw sequence — tracking IDs comparable across the run"
          if one_sequence else
          "NOT one continuous sequence — split by prefix and track each clip separately; "
          "unique-ID count over the concatenation is not a valid ID-switch signal")
    return one_sequence


if __name__ == "__main__":
    analyze(sys.argv[1] if len(sys.argv) > 1 else "data/raw/cathaction/img")
