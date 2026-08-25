"""Kaggle notebook 2 — Model One backbone bake-off on the AngioCAD proxy corpus.

Reuses the corpus built by `kaggle_angiocad_acquire.ipynb`. **Downloads nothing.**

Kaggle settings:
  * Add data -> Notebook Output -> `jugalmodi0111/angiocad`   (mounts the corpus read-only)
  * Internet: ON      (GitHub clone + timm pulls pretrained weights on first use)
  * Accelerator: GPU T4

Why this notebook exists rather than a `train_classifier` CLI call: `train()` does a FULL-BATCH
forward — `model(xt)` over every frame at once. That is fine for the seeded 32-dim `test-tiny`
backbone the unit tests use, and hopeless for a real one. Measured: the input tensor for 10,421
frames at 224 px is only 2.09 GB, but DINOv2 ViT-B activations for that batch are 257 tokens x 768
dim x 12 layers ~ 100 GB. So this notebook does the thing the frozen-backbone design implies
anyway: run the backbone ONCE in batches, cache [N, feat_dim] features, and train the linear head
on the cache. The backbone never updates, so recomputing it every epoch was always waste.

Honest framing, unchanged from notebook 1: coronary angiography is a PROXY for AVF. Nothing here is
a clinical claim about dialysis access, the corpus labels are per-video severity grades read off a
spreadsheet, and B7 floors are still unsigned.
"""

# ==================================================================================================
# %% CELL 1 — locate corpus, clone repo, verify patient grouping
# ==================================================================================================
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

BRANCH = "main"          # the Model One scaffold merged to main in PR #1
REPO_URL = "https://github.com/jugalmodi0111/interventional-imaging-pipeline.git"
REPO = Path("/kaggle/working/repo")
OUT = Path("/kaggle/working/bakeoff")
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("PREFLIGHT")
print("=" * 90)

# Auto-discover the corpus: the mount point depends on how the data source was added, and hardcoding
# one path is the single most likely way for this notebook to fail on someone else's setup.
cands = sorted(Path("/kaggle/input").rglob("corpus/labels.jsonl"))
if not cands:
    raise SystemExit(
        "corpus not found under /kaggle/input.\n"
        "Fix: sidebar -> Add Input -> Notebook Output -> jugalmodi0111/angiocad, then rerun.\n"
        f"Present inputs: {[p.name for p in Path('/kaggle/input').glob('*')] or '(none)'}")
CORPUS = cands[0].parent
FRAMES = CORPUS / "frames"
print(f"corpus         : {CORPUS}")

try:
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch          : {torch.__version__} | {torch.cuda.get_device_name(0) if dev == 'cuda' else 'CPU'}")
except Exception as e:
    raise SystemExit(f"torch unavailable: {e}")
if dev == "cpu":
    print("  !! no GPU — a real backbone over 10k frames on CPU is hours. Enable the T4 accelerator.")

subprocess.run(f"{sys.executable} -m pip -q install timm", shell=True, capture_output=True)
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.run(f"git clone -q --depth 1 -b {BRANCH} {REPO_URL} {REPO}", shell=True, check=True)
sys.path.insert(0, str(REPO))
head = subprocess.run(f"git -C {REPO} rev-parse --short HEAD", shell=True,
                      capture_output=True, text=True).stdout.strip()
print(f"repo           : {BRANCH} @ {head}")

from src.train.train_classifier import _patient_group, grouped_split, load_examples  # noqa: E402
import timm  # noqa: E402
print(f"timm           : {timm.__version__}")

examples = load_examples(FRAMES, CORPUS / "labels.jsonl")
groups = {e["group"] for e in examples}
rows = [json.loads(l) for l in (CORPUS / "labels.jsonl").read_text().splitlines() if l.strip()]
patients = {f"angiocad_{r['patient']}" for r in rows}
pos = sum(r["label"] for r in rows)
print(f"\ncorpus         : {len(examples)} frames / {len(rows)} videos / {len(groups)} groups")
print(f"labels         : {pos} positive, {len(rows) - pos} negative ({pos / len(rows):.1%})")

# The leak this repo has met four times. A per-series split has every group unique by construction,
# so it passes a group-overlap audit while scattering one patient across train and val. 277 of 413
# AngioCAD patients have videos on BOTH coronary sides, so it would bite immediately here.
assert groups == patients, (
    f"stems are NOT grouping per patient: {len(groups)} groups vs {len(patients)} patients. "
    f"io_utils._ANGIOCAD_RE missing on this branch. Sample: {sorted(groups)[:3]}")
print(f"grouping       : PASS — one group per patient ({len(groups)})")
print("\npreflight OK")


# ==================================================================================================
# %% CELL 2 — feature cache: run each frozen backbone ONCE, in batches
# ==================================================================================================
import numpy as np

# Verified against timm 1.0.28 (registry lookup, no downloads) on 2026-08-25. `dinov2_vitb14` --
# the name configs/avf_fistulography.yaml shipped -- is a torch.hub name and raises
# `RuntimeError: Unknown model` in timm; the config has been corrected. DINOv2 in timm is built at
# 518 px and asserts on a 224 input unless img_size is passed, which make_backbone now does.
BACKBONES = [
    "vit_base_patch14_dinov2.lvd142m",   # B4's declared default (name corrected)
    "vit_base_patch16_dinov3",           # B4's eventual target -- licence unverified for hosting
    "vit_base_patch16_224.dino",         # DINOv1, native 224: isolates DINOv2's gain from resolution
    "resnet50.a1_in1k",                  # supervised CNN control; if it wins, "foundation" is unearned
]
IMGSZ = 224
BATCH = 64

print("=" * 90)
print("FEATURE CACHE")
print("=" * 90)


def load_batch(exs, imgsz):
    import cv2
    x = np.stack([cv2.resize(cv2.imread(e["path"], cv2.IMREAD_GRAYSCALE), (imgsz, imgsz),
                             interpolation=cv2.INTER_AREA) for e in exs]).astype("float32") / 255.0
    return torch.from_numpy(x[:, None])


def feature_cache(name, exs, imgsz=IMGSZ, batch=BATCH):
    """[N, feat_dim] features from a FROZEN backbone. Computed once; the head trains on these.

    in_chans=1 makes timm sum the pretrained RGB stem weights into a single grayscale channel,
    which is the right adaptation for X-ray: replicating a gray frame to 3 channels wastes 3x the
    compute to feed the network three identical planes.
    """
    from src.models.frozen_backbone import make_backbone
    t0 = time.time()
    backbone, dim = make_backbone(name, imgsz=imgsz)
    backbone = backbone.to(dev).eval()
    feats = np.empty((len(exs), dim), dtype="float32")
    with torch.no_grad():
        for i in range(0, len(exs), batch):
            chunk = exs[i:i + batch]
            feats[i:i + len(chunk)] = backbone(load_batch(chunk, imgsz).to(dev)).float().cpu().numpy()
            if (i // batch) % 25 == 0:
                done = i + len(chunk)
                sp = done / max(time.time() - t0, 1e-9)
                print(f"\r  {name:28s} {done}/{len(exs)} frames  {sp:.0f} f/s  "
                      f"ETA {(len(exs) - done) / max(sp, 1e-9) / 60:.1f} min   ", end="", flush=True)
    print(f"\r  {name:28s} {len(exs)} frames -> [{len(exs)}, {dim}] in {(time.time() - t0) / 60:.1f} min      ")
    return feats, dim


y = np.array([e["label"] for e in examples], dtype="float32")
CACHE = {}
for name in BACKBONES:
    try:
        CACHE[name] = feature_cache(name, examples)
    except Exception as e:
        print(f"  {name:28s} FAILED: {type(e).__name__}: {e}")
print(f"\ncached: {list(CACHE)}")


# ==================================================================================================
# %% CELL 3 — train a head per backbone on patient-grouped splits, score, rank
# ==================================================================================================
from src.eval.calibration import apply_temperature, auroc, ece, temperature_scale
from src.eval.cls_metrics import bootstrap_ci, sensitivity, specificity, threshold_at_sensitivity

EPOCHS, LR, SEED, VAL_FRAC = 300, 1e-3, 0, 0.25
TARGET_SENSITIVITY = None      # B7 floor UNSIGNED -> threshold falls back to 0.5, marked unsigned

print("=" * 90)
print("BAKE-OFF")
print("=" * 90)

tr_idx, va_idx = [], []
tr_ex, va_ex = grouped_split(examples, val_frac=VAL_FRAC, seed=SEED)
tr_set = {e["path"] for e in tr_ex}
for i, e in enumerate(examples):
    (tr_idx if e["path"] in tr_set else va_idx).append(i)
tr_idx, va_idx = np.array(tr_idx), np.array(va_idx)
tr_g = {e["group"] for e in tr_ex}
va_g = {e["group"] for e in va_ex}
assert not (tr_g & va_g), "patient in both splits"
print(f"split: train {len(tr_idx)}f/{len(tr_g)}p  val {len(va_idx)}f/{len(va_g)}p  (no patient overlap)")
print(f"       train prevalence {y[tr_idx].mean():.1%} | val prevalence {y[va_idx].mean():.1%}\n")

results = []
for name, (feats, dim) in CACHE.items():
    torch.manual_seed(SEED)
    # Standardise on TRAIN statistics only — fitting the scaler on val leaks the val distribution
    # into the model, which is the same class of mistake as splitting by frame.
    mu, sd = feats[tr_idx].mean(0), feats[tr_idx].std(0) + 1e-6
    Xtr = torch.from_numpy((feats[tr_idx] - mu) / sd).to(dev)
    Xva = torch.from_numpy((feats[va_idx] - mu) / sd).to(dev)
    ytr = torch.from_numpy(y[tr_idx]).to(dev)

    head = torch.nn.Linear(dim, 1).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=1e-4)
    for _ in range(EPOCHS):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(head(Xtr).squeeze(-1), ytr).backward()
        opt.step()

    with torch.no_grad():
        logits = head(Xva).squeeze(-1).float().cpu().numpy()
    yv = y[va_idx].astype(int)
    T = float(temperature_scale(logits, yv))
    probs = apply_temperature(logits, T)
    thr = threshold_at_sensitivity(probs, yv, TARGET_SENSITIVITY)
    r = {"backbone": name, "feat_dim": dim, "auroc": float(auroc(probs, yv)),
         "ece": float(ece(probs, yv)), "temperature": T, "threshold": float(thr),
         "threshold_policy": "threshold-unsigned" if TARGET_SENSITIVITY is None else "at-target-sensitivity",
         "sensitivity": sensitivity(probs, yv, thr), "specificity": specificity(probs, yv, thr),
         "auroc_ci": bootstrap_ci(lambda p, l, t: float(auroc(p, l)), probs, yv, n_boot=300, seed=SEED),
         "n_train_frames": len(tr_idx), "n_val_frames": len(va_idx),
         "n_train_patients": len(tr_g), "n_val_patients": len(va_g)}
    results.append(r)
    print(f"{name:28s} AUROC {r['auroc']:.3f} {tuple(round(v, 3) for v in r['auroc_ci'])}  "
          f"ECE {r['ece']:.3f}  sens {r['sensitivity']:.3f}  spec {r['specificity']:.3f}")

results.sort(key=lambda r: -r["auroc"])
(OUT / "bakeoff.json").write_text(json.dumps(results, indent=2))

print("\n" + "=" * 90)
print("RANKING (val AUROC, patient-grouped, frame-level)")
print("=" * 90)
for i, r in enumerate(results, 1):
    print(f"{i}. {r['backbone']:28s} {r['auroc']:.3f}  [{r['auroc_ci'][0]:.3f}, {r['auroc_ci'][1]:.3f}]")
if len(results) > 1:
    a, b = results[0], results[1]
    overlap = a["auroc_ci"][0] <= b["auroc_ci"][1]
    print(f"\ntop-two CIs {'OVERLAP — this does NOT pick a winner' if overlap else 'are disjoint'}.")
    print("STAGE_ACCURACY_RESEARCH.md recorded 'REFUTED: DINOv2 wins', so the bake-off has to be")
    print("read on evidence and a difference inside the CI is not evidence.")

print("\nSCOPE: frame-level metrics on a coronary PROXY corpus with spreadsheet-derived per-video")
print("labels. Not per-study, not AVF, not a clinical claim. B7 floors remain unsigned, which is")
print(f"why threshold_policy is '{results[0]['threshold_policy'] if results else 'n/a'}'.")
print(f"\nwrote {OUT / 'bakeoff.json'}")
