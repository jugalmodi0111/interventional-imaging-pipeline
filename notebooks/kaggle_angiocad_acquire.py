"""Kaggle notebook — AngioCAD acquisition, probe, adapter validation, corpus build.

Paste each `# %% CELL n` block into its own Kaggle cell and run in order.

Kaggle settings REQUIRED before running:
  * Internet: ON            (Zenodo + GitHub + timm weights)
  * Accelerator: GPU T4     (only Cell 6 needs it; Cells 1-5 are CPU)
  * Persistence: Files only (so /kaggle/working survives between sessions)

What this run establishes, in order:
  1. Whether the "3.4 TB extracted" figure carried in docs/DATASETS.md is real. It is unsourced —
     it does NOT appear in the Zenodo record, and 16.4 GB of RAR'd PNG cannot expand 200x because
     PNG is already DEFLATE-compressed. Cell 3 reads the uncompressed sizes out of the RAR headers
     WITHOUT extracting and settles it with evidence. Everything downstream branches on the answer.
  2. Whether `angiocad_to_cls`'s assumed frame layout (<root>/<patient>/<series>/) matches the real
     archive. That path has never been checked against an actual tree — the adapter was written and
     tested against the 43 kB labels sheet alone. Cell 5 measures the hit rate and, if it misses,
     reports the real layout rather than guessing.
  3. Whether the 94 distinct series-spec formats parsed out of the sheet correspond to folders that
     actually exist. A spec that parses cleanly and names a nonexistent series is a silent corpus
     hole, not an error.

NOT in this run: the backbone bake-off. `train_classifier._tensorize` stacks every frame into one
tensor, which is fine for the synthetic suite and will OOM on a real corpus. Feature-caching the
frozen backbone (compute features once, train the head on those) is the fix, and it is local TDD
work rather than something to paste into a notebook. Cell 6 is a SMOKE TEST on a deliberately small
subset — first real-backbone run in this repo, not a result.
"""

# ==================================================================================================
# %% CELL 1 — preflight: env, disk, tooling, repo
# ==================================================================================================
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# The scaffold (train_classifier, frozen_backbone, cls_metrics) lives on the `model-one-scaffold`
# branch and is NOT on main. Cells 1-5 only need `angiocad_to_cls`, which IS on main. Set this to
# "model-one-scaffold" after you push that branch if you want Cell 6 to run.
BRANCH = "main"
REPO_URL = "https://github.com/jugalmodi0111/interventional-imaging-pipeline.git"

REPO = Path("/kaggle/working/repo")
SCRATCH = Path("/kaggle/temp/angiocad")          # big + ephemeral: RARs and the extracted tree
OUT = Path("/kaggle/working/angiocad_out")        # small + persisted: the corpus and reports
DL = SCRATCH / "download"
RAW = SCRATCH / "raw"
for d in (SCRATCH, OUT, DL, RAW):
    d.mkdir(parents=True, exist_ok=True)


def sh(cmd, check=True, quiet=False):
    """Run a shell command, streaming failures loudly rather than swallowing them."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if not quiet and r.stdout.strip():
        print(r.stdout.strip()[:4000])
    if r.returncode != 0:
        print(f"[rc={r.returncode}] {cmd}\n{r.stderr.strip()[:4000]}")
        if check:
            raise RuntimeError(f"command failed: {cmd}")
    return r


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


print("=" * 90)
print("PREFLIGHT")
print("=" * 90)

# Internet is the single most common Kaggle misconfiguration for this notebook; check it first and
# name the fix, because every later cell fails opaquely without it.
try:
    import urllib.request
    urllib.request.urlopen("https://zenodo.org/api/records/15826856", timeout=20).read(64)
    print("internet          : OK")
except Exception as e:
    raise SystemExit(f"internet is OFF or blocked ({e}).\n"
                     "Fix: notebook sidebar -> Settings -> Internet -> On, then rerun this cell.")

try:
    import torch
    print(f"torch             : {torch.__version__} | cuda={torch.cuda.is_available()} "
          f"| {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
except Exception as e:
    print(f"torch             : unavailable ({e}) — only Cell 6 needs it")

for p in ("/kaggle/working", "/kaggle/temp"):
    print(f"free {p:16s}: {free_gb(p):.1f} GB")

# 16.4 GB of archives plus the extracted tree must coexist: a multi-part RAR needs every part
# present during extraction, so the download cannot be deleted incrementally to make room.
if free_gb("/kaggle/temp") < 40:
    print(f"\n!! /kaggle/temp has {free_gb('/kaggle/temp'):.1f} GB free; 40+ GB is the comfortable "
          f"figure for 16.4 GB of archives alongside their extracted tree.")
    print("   Cell 3 measures the real requirement before Cell 4 commits to extracting anything.")

# --- RAR backend. Kaggle images vary; try the three that actually decode RAR5, in order of speed.
print("\nlocating a RAR backend ...")
sh("apt-get -qq update", check=False, quiet=True)
for pkg in ("unrar", "p7zip-rar", "libarchive-tools"):
    sh(f"apt-get -qq install -y {pkg}", check=False, quiet=True)

RAR_TOOL = None
for tool, probe in (("unrar", "unrar -inul"), ("7zz", "7zz"), ("7z", "7z"), ("bsdtar", "bsdtar --version")):
    if shutil.which(tool):
        RAR_TOOL = tool
        break
if RAR_TOOL is None:
    raise SystemExit("no RAR backend found (tried unrar, 7zz, 7z, bsdtar). "
                     "Confirm Internet is ON — apt-get needs it to install one.")
print(f"rar backend       : {RAR_TOOL}")

sh(f"{sys.executable} -m pip -q install openpyxl rarfile", check=False, quiet=True)

# --- repo
if REPO.exists():
    shutil.rmtree(REPO)
sh(f"git clone -q --depth 1 -b {BRANCH} {REPO_URL} {REPO}")
sys.path.insert(0, str(REPO))
head = sh(f"git -C {REPO} rev-parse --short HEAD", quiet=True).stdout.strip()
print(f"repo              : {BRANCH} @ {head}")

from src.data_prep import angiocad_to_cls as ac  # noqa: E402
print(f"adapter           : angiocad_to_cls OK ({len(ac.RCA_SEGMENTS)} RCA + "
      f"{len(ac.LCA_SEGMENTS)} LCA segments)")

HAS_SCAFFOLD = (REPO / "src" / "train" / "train_classifier.py").exists()
print(f"Model One scaffold: {'present — Cell 6 can run' if HAS_SCAFFOLD else 'ABSENT on this branch'}")
if not HAS_SCAFFOLD:
    print("                    (push `model-one-scaffold` and set BRANCH above to enable Cell 6)")
print("\npreflight OK")


# ==================================================================================================
# %% CELL 2 — download from Zenodo (16.4 GB, resumable)
# ==================================================================================================
import hashlib
import urllib.request

RECORD = "15826856"
print("=" * 90)
print(f"DOWNLOAD — Zenodo record {RECORD}")
print("=" * 90)

meta = json.loads(urllib.request.urlopen(
    f"https://zenodo.org/api/records/{RECORD}", timeout=60).read())
files = {f["key"]: f for f in meta["files"]}
print(f"license: {meta['metadata']['license']['id']} | {len(files)} files, "
      f"{sum(f['size'] for f in files.values()) / 1e9:.2f} GB total\n")


def md5sum(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def fetch(key):
    """Download one Zenodo file, skipping it when a byte-exact copy is already on disk.

    Kaggle sessions die; re-running this cell must not re-pull 16 GB. Size is the cheap gate and
    the MD5 from the record metadata is the real one — a truncated part only shows up as a
    confusing archive error three cells later otherwise.
    """
    f = files[key]
    dest = DL / key
    if dest.exists() and dest.stat().st_size == f["size"]:
        print(f"  {key:38s} present ({f['size'] / 1e9:.2f} GB) — skipped")
        return dest
    print(f"  {key:38s} downloading {f['size'] / 1e9:.2f} GB ...", end="", flush=True)
    t0 = time.time()
    urllib.request.urlretrieve(f["links"]["self"], dest)
    dt = time.time() - t0
    print(f" done in {dt / 60:.1f} min ({f['size'] / 1e6 / max(dt, 1):.0f} MB/s)")
    return dest


need = free_gb("/kaggle/temp")
want = sum(f["size"] for f in files.values()) / 1e9
if need < want + 2:
    raise SystemExit(f"/kaggle/temp has {need:.1f} GB free, need ~{want + 2:.1f} GB for the archives.")

paths = {k: fetch(k) for k in files}

print("\nverifying checksums (the truncated-part failure mode is worth 2 minutes here) ...")
for key, p in paths.items():
    want_md5 = files[key]["checksum"].split(":", 1)[1]
    got = md5sum(p)
    ok = got == want_md5
    print(f"  {key:38s} {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit(f"{key} checksum mismatch (got {got}, want {want_md5}). Delete it and rerun.")

LABELS_XLSX = paths["AngioCAD_Labels.xlsx"]
PART1 = paths["AngioCAD_Dataset.part1.rar"]
print(f"\nall files verified. free after download: {free_gb('/kaggle/temp'):.1f} GB")


# ==================================================================================================
# %% CELL 3 — PROBE the archive without extracting  (settles the 3.4 TB question)
# ==================================================================================================
import collections
import re

print("=" * 90)
print("PROBE — reading RAR headers, extracting nothing")
print("=" * 90)

entries = []          # (path, uncompressed_size)
try:
    import rarfile
    rarfile.UNRAR_TOOL = RAR_TOOL if RAR_TOOL in ("unrar", "bsdtar") else "unrar"
    with rarfile.RarFile(PART1) as rf:
        entries = [(i.filename.replace("\\", "/"), int(i.file_size or 0))
                   for i in rf.infolist() if not i.is_dir()]
    print(f"listed via rarfile ({len(entries)} entries)")
except Exception as e:
    print(f"rarfile listing unavailable ({e}); falling back to CLI listing")
    if RAR_TOOL == "unrar":
        raw = sh(f'unrar l -v "{PART1}"', quiet=True).stdout
        for line in raw.splitlines():
            m = re.match(r"\s+[\-drwx.]+\s+(\d+)\s+\S+\s+\S+\s+\S+\s+(.+)$", line)
            if m:
                entries.append((m.group(2).strip().replace("\\", "/"), int(m.group(1))))
    else:
        raw = sh(f'{RAR_TOOL} l "{PART1}"', quiet=True).stdout
        for line in raw.splitlines():
            m = re.match(r"^\d{4}-\d\d-\d\d\s+\S+\s+[\.DRHSA]+\s+(\d+)\s+\d*\s+(.+)$", line)
            if m:
                entries.append((m.group(2).strip().replace("\\", "/"), int(m.group(1))))
    print(f"listed via {RAR_TOOL} ({len(entries)} entries)")

if not entries:
    raise SystemExit("could not list the archive — inspect the raw CLI output above before extracting.")

total = sum(sz for _, sz in entries)
exts = collections.Counter(Path(p).suffix.lower() for p, _ in entries)
depth1 = collections.Counter(p.split("/")[0] for p, _ in entries)
depth2 = collections.Counter("/".join(p.split("/")[:2]) for p, _ in entries)

print(f"\nentries            : {len(entries):,}")
print(f"uncompressed total : {total / 1e9:.2f} GB   ({total / 1e12:.4f} TB)")
print(f"archive on disk    : {sum(files[k]['size'] for k in files if k.endswith('.rar')) / 1e9:.2f} GB")
print(f"expansion ratio    : {total / max(sum(files[k]['size'] for k in files if k.endswith('.rar')), 1):.2f}x")
print(f"extensions         : {dict(exts.most_common(8))}")
print(f"top-level dirs     : {len(depth1)}  e.g. {list(depth1)[:6]}")
print(f"depth-2 dirs       : {len(depth2)}  e.g. {list(depth2)[:6]}")
print("\nsample paths:")
for p, sz in entries[:12]:
    print(f"  {p}   ({sz / 1024:.0f} KB)")

print("\n" + "-" * 90)
print("VERDICT on the '3.4 TB extracted' figure carried in docs/DATASETS.md")
print("-" * 90)
if total < 100e9:
    print(f"REFUTED. Real uncompressed size is {total / 1e9:.2f} GB, not 3.4 TB — a "
          f"{3.4e12 / max(total, 1):.0f}x overstatement.")
    print("PNG is already DEFLATE-compressed, so RAR recovers only a few percent; a 200x expansion")
    print("was never physically possible. Wholesale extraction is viable and 'selective extraction")
    print("must be planned first' was a blocker built on a bad number. Update docs/DATASETS.md and")
    print("PROJECT_TRACKER after this run.")
else:
    print(f"NOT refuted: {total / 1e9:.1f} GB uncompressed. Cell 4 will extract selectively.")

FITS = total < (free_gb("/kaggle/temp") - 5) * 1e9
print(f"\nfree /kaggle/temp  : {free_gb('/kaggle/temp'):.1f} GB")
print(f"full extraction    : {'FITS — Cell 4 extracts everything' if FITS else 'DOES NOT FIT — Cell 4 goes selective'}")

PROBE = {"n_entries": len(entries), "uncompressed_bytes": total, "fits": bool(FITS),
         "extensions": dict(exts), "depth1_sample": list(depth1)[:40],
         "depth2_sample": list(depth2)[:40], "sample_paths": [p for p, _ in entries[:40]]}
(OUT / "probe.json").write_text(json.dumps(PROBE, indent=2))
print(f"\nwrote {OUT / 'probe.json'}")


# ==================================================================================================
# %% CELL 4 — extract
# ==================================================================================================
print("=" * 90)
print("EXTRACT")
print("=" * 90)

t0 = time.time()
if FITS:
    print(f"extracting all {PROBE['n_entries']:,} entries to {RAW} ...")
    if RAR_TOOL == "unrar":
        sh(f'unrar x -o+ -idq "{PART1}" "{RAW}/"')
    elif RAR_TOOL == "bsdtar":
        sh(f'bsdtar -xf "{PART1}" -C "{RAW}"')
    else:
        sh(f'{RAR_TOOL} x -y -o"{RAW}" "{PART1}"', quiet=True)
else:
    # Selective: take only the first N frames of each series directory. Enough to train and to
    # validate the layout; the full cine is not needed for a still-frame classifier (B3 serves one
    # frame at a time), and per-frame redundancy inside one clip is near-total anyway.
    KEEP_PER_SERIES = 8
    by_dir = collections.defaultdict(list)
    for p, _ in entries:
        by_dir[str(Path(p).parent)].append(p)
    wanted = [p for d in sorted(by_dir) for p in sorted(by_dir[d])[:KEEP_PER_SERIES]]
    print(f"selective: {len(wanted):,} of {len(entries):,} entries "
          f"({KEEP_PER_SERIES}/series across {len(by_dir):,} series)")
    listfile = SCRATCH / "wanted.txt"
    listfile.write_text("\n".join(wanted))
    if RAR_TOOL == "unrar":
        sh(f'unrar x -o+ -idq "{PART1}" @"{listfile}" "{RAW}/"')
    elif RAR_TOOL == "bsdtar":
        sh(f'bsdtar -xf "{PART1}" -C "{RAW}" -T "{listfile}"')
    else:
        sh(f'{RAR_TOOL} x -y -o"{RAW}" "{PART1}" -i@"{listfile}"', quiet=True)

n_png = sum(1 for _ in RAW.rglob("*.png"))
size = sum(f.stat().st_size for f in RAW.rglob("*") if f.is_file())
print(f"\nextracted in {(time.time() - t0) / 60:.1f} min: {n_png:,} PNG, {size / 1e9:.2f} GB on disk")
print(f"free /kaggle/temp: {free_gb('/kaggle/temp'):.1f} GB")

roots = [d for d in RAW.iterdir() if d.is_dir()]
print(f"\nextracted root(s): {[d.name for d in roots[:8]]}{' ...' if len(roots) > 8 else ''}")
# Descend through any single wrapper directory the archive adds around the real patient tree.
FRAMES_ROOT = RAW
while True:
    kids = [d for d in FRAMES_ROOT.iterdir() if d.is_dir()]
    if len(kids) == 1 and not any(FRAMES_ROOT.glob("*.png")):
        FRAMES_ROOT = kids[0]
        print(f"  descending through single wrapper dir -> {FRAMES_ROOT.relative_to(RAW)}")
    else:
        break
print(f"FRAMES_ROOT = {FRAMES_ROOT}")
for d in sorted([d for d in FRAMES_ROOT.iterdir() if d.is_dir()])[:5]:
    subs = sorted([s.name for s in d.iterdir() if s.is_dir()])[:6]
    n = sum(1 for _ in d.rglob("*.png"))
    print(f"  {d.name}/  subdirs={subs}  png={n}")


# ==================================================================================================
# %% CELL 5 — validate the adapter against the REAL tree
# ==================================================================================================
print("=" * 90)
print("ADAPTER VALIDATION — does angiocad_to_cls's assumed layout match reality?")
print("=" * 90)

recs50 = ac.build_records(LABELS_XLSX, threshold=50, frames_root=str(FRAMES_ROOT))
recs70 = ac.build_records(LABELS_XLSX, threshold=70, frames_root=str(FRAMES_ROOT))
pat = len({r["patient"] for r in recs50})
print(f"sheet says          : {len(recs50)} videos / {pat} patients")
print(f"  positive @50%     : {sum(r['positive'] for r in recs50)} "
      f"({sum(r['positive'] for r in recs50) / len(recs50):.1%})")
print(f"  positive @70%     : {sum(r['positive'] for r in recs70)} "
      f"({sum(r['positive'] for r in recs70) / len(recs70):.1%})")
print(f"  the 50-vs-70 gap  : {sum(r['positive'] for r in recs50) - sum(r['positive'] for r in recs70)}"
      f" videos change class — a CLINICAL choice for Dr. Reddy, not ours")

# The assumed layout is <frames_root>/<patient>/<series>/. It has never been checked against a real
# tree; the adapter was written against the 43 kB sheet alone. Measure the hit rate.
found = [r for r in recs50 if Path(r["frames"]).is_dir() and any(Path(r["frames"]).glob("*.png"))]
print(f"\nassumed layout      : <root>/<patient>/<series>/")
print(f"  resolved on disk  : {len(found)}/{len(recs50)} ({len(found) / len(recs50):.1%})")

if len(found) < 0.5 * len(recs50):
    print("\n  LOW HIT RATE — the assumed layout is wrong. Real structure, for the fix:")
    for d in sorted([d for d in FRAMES_ROOT.iterdir() if d.is_dir()])[:8]:
        kids = sorted([k.name for k in d.iterdir()])[:10]
        print(f"    {d.name}/ -> {kids}")
    sheet_pat = sorted({str(r['patient']) for r in recs50})[:10]
    disk_pat = sorted([d.name for d in FRAMES_ROOT.iterdir() if d.is_dir()])[:10]
    print(f"\n    patient ids in sheet: {sheet_pat}")
    print(f"    dir names on disk   : {disk_pat}")
    print("    -> reconcile these two before building a corpus; do NOT pattern-match blindly.")
else:
    found_ids = {id(r) for r in found}
    miss = [r for r in recs50 if id(r) not in found_ids]
    missing_pat = collections.Counter(str(r["patient"]) for r in miss)
    print(f"  unresolved        : {len(miss)} videos across {len(missing_pat)} patients")
    if missing_pat:
        print(f"  worst offenders   : {missing_pat.most_common(8)}")
        print("  (a series spec that parses cleanly but names no folder is a silent corpus hole)")

    # Orphans run the other way: folders on disk the sheet never names. Those are unlabelled videos.
    on_disk = {(d.name, s.name) for d in FRAMES_ROOT.iterdir() if d.is_dir()
               for s in d.iterdir() if s.is_dir()}
    claimed = {(str(r["patient"]), str(r["series"])) for r in recs50}
    orphans = on_disk - claimed
    print(f"  orphan folders    : {len(orphans)} on disk but unnamed by the sheet"
          f"{' e.g. ' + str(sorted(orphans)[:5]) if orphans else ''}")

frames_per = [len(list(Path(r["frames"]).glob("*.png"))) for r in found[:400]]
if frames_per:
    frames_per.sort()
    print(f"\nframes per video    : min {frames_per[0]}  p50 {frames_per[len(frames_per) // 2]}  "
          f"max {frames_per[-1]}  (n={len(frames_per)} sampled)")

report = {"videos_sheet": len(recs50), "patients": pat, "resolved_on_disk": len(found),
          "positive_50": sum(r["positive"] for r in recs50),
          "positive_70": sum(r["positive"] for r in recs70),
          "frames_per_video_p50": frames_per[len(frames_per) // 2] if frames_per else None,
          "frames_root": str(FRAMES_ROOT)}
(OUT / "adapter_validation.json").write_text(json.dumps(report, indent=2))
print(f"\nwrote {OUT / 'adapter_validation.json'}")


# ==================================================================================================
# %% CELL 6 — build the corpus, then a real-backbone SMOKE TEST
# ==================================================================================================
import numpy as np

print("=" * 90)
print("CORPUS BUILD -> ingest frame-store layout")
print("=" * 90)

# `train_classifier.load_examples` expects exactly what src/ingest/ produces:
#   frames/<stem>/f%05d.png  +  labels.jsonl of {"key": <stem>, "label": 0|1}
# Stems are `angiocad_<patient>_s<NN>`, which `io_utils._ANGIOCAD_RE` collapses to the PATIENT. That
# rule was added 2026-08-24 specifically for this corpus: without it group_key falls through to
# `return name`, groups per SERIES, and scatters a patient across train and val -- the P0.2 /
# CADICA / AVF bug a fourth time. Asserted below, not assumed.
FRAMES_PER_VIDEO = 4          # a still-frame classifier; frames within one clip are near-duplicates
THRESHOLD = 50                # PROVISIONAL. Dr. Reddy owns this; 70 is the other candidate.

corpus = OUT / "corpus"
cframes = corpus / "frames"
if corpus.exists():
    shutil.rmtree(corpus)
cframes.mkdir(parents=True)

import cv2  # noqa: E402

recs = ac.build_records(LABELS_XLSX, threshold=THRESHOLD, frames_root=str(FRAMES_ROOT))
rows, n_frames = [], 0
for r in recs:
    src = Path(r["frames"])
    if not src.is_dir():
        continue
    pngs = sorted(src.glob("*.png"))
    if not pngs:
        continue
    idx = np.linspace(0, len(pngs) - 1, min(FRAMES_PER_VIDEO, len(pngs))).round().astype(int)
    stem = f"angiocad_{r['patient']}_s{int(r['series']):02d}"
    dst = cframes / stem
    dst.mkdir(parents=True, exist_ok=True)
    for j, i in enumerate(sorted(set(idx))):
        img = cv2.imread(str(pngs[i]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        cv2.imwrite(str(dst / f"f{j:05d}.png"), cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA))
        n_frames += 1
    rows.append({"key": stem, "label": int(r["positive"]), "patient": str(r["patient"]),
                 "series": int(r["series"]), "side": r["side"]})

(corpus / "labels.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
pos = sum(r["label"] for r in rows)
print(f"corpus: {len(rows)} videos / {len({r['patient'] for r in rows})} patients / "
      f"{n_frames} frames @224 ({pos} pos, {len(rows) - pos} neg) "
      f"= {sum(f.stat().st_size for f in cframes.rglob('*.png')) / 1e9:.2f} GB")
print(f"       -> {corpus}  (save this notebook's output as a Kaggle Dataset for the bake-off run)")

# Hard gate: one group per patient, never per series. A corpus that fails this is unusable, and it
# fails SILENTLY -- a per-series split has every group unique by construction and sails through any
# group-overlap audit. Crash here rather than train on it.
if HAS_SCAFFOLD:
    from src.train.train_classifier import _patient_group
    groups = {_patient_group(r["key"]) for r in rows}
    patients = {f"angiocad_{r['patient']}" for r in rows}
    print(f"grouping check: {len(groups)} groups vs {len(patients)} patients "
          f"({len(rows)} videos)")
    assert groups == patients, (
        f"stems are NOT grouping per patient -- {len(groups)} groups for {len(patients)} patients. "
        f"io_utils._ANGIOCAD_RE is missing or does not match this stem grammar. Sample: "
        f"{sorted(groups)[:5]}")
    print("grouping check: PASS -- one group per patient")

# --- smoke test -----------------------------------------------------------------------------------
print("\n" + "=" * 90)
print("SMOKE TEST — first real timm backbone in this repo, on a deliberately small subset")
print("=" * 90)
if not HAS_SCAFFOLD:
    print("SKIPPED: `model-one-scaffold` is not on this branch. Push it, set BRANCH in Cell 1, rerun.")
else:
    from src.train.train_classifier import grouped_split, load_examples, train

    ex = load_examples(cframes, corpus / "labels.jsonl")
    tr, va = grouped_split(ex, val_frac=0.2, seed=0)
    print(f"full corpus: {len(ex)} frames | {len({e['group'] for e in ex})} patient groups")
    print(f"  train {len(tr)} frames / {len({e['group'] for e in tr})} patients")
    print(f"  val   {len(va)} frames / {len({e['group'] for e in va})} patients")
    print("  group overlap: none (grouped_split asserts it — a leak is a crash, not a warning)")

    # Small on purpose. `_tensorize` stacks EVERY frame into one tensor, so the full corpus would
    # OOM. This proves the real-backbone path runs on real data; it is not the bake-off.
    SUBSET = 60
    keep = {g for g in sorted({e["group"] for e in ex})[:SUBSET]}
    sub = corpus / "subset"
    if sub.exists():
        shutil.rmtree(sub)
    (sub / "frames").mkdir(parents=True)
    subrows = [r for r in rows if f"angiocad_{r['patient']}" in keep]
    for r in subrows:
        shutil.copytree(cframes / r["key"], sub / "frames" / r["key"])
    (sub / "labels.jsonl").write_text("\n".join(json.dumps(r) for r in subrows) + "\n")
    print(f"\nsubset: {len(subrows)} videos / {len(keep)} patients")

    for backbone in ("test-tiny", "dinov2_vitb14"):
        print(f"\n--- {backbone} " + "-" * (70 - len(backbone)))
        try:
            t0 = time.time()
            m = train(sub / "frames", sub / "labels.jsonl", OUT / f"smoke_{backbone}",
                      backbone=backbone, imgsz=32 if backbone == "test-tiny" else 224,
                      epochs=30, lr=1e-2, val_frac=0.3, seed=0)
            print(f"  {(time.time() - t0) / 60:.1f} min | auroc {m['auroc']:.3f} | ece {m['ece']:.3f} "
                  f"| sens {m['sensitivity']:.3f} | spec {m['specificity']:.3f}")
            print(f"  train {m['n_train_frames']}f/{m['n_train_groups']}p  "
                  f"val {m['n_val_frames']}f/{m['n_val_groups']}p  policy {m['threshold_policy']}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")

    print("\nRead these as PLUMBING evidence, not performance. n is tiny, the head trains full-batch")
    print("for 30 steps, and coronary angiography is a PROXY for AVF -- never a clinical claim.")

print(f"\nartifacts in {OUT}: {sorted(p.name for p in OUT.iterdir())}")
