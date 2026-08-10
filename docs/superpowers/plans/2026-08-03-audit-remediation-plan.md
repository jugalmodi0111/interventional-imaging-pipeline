# Dialygo Audit Remediation & Continuation Plan

**Date:** 2026-08-03 · **Status:** proposed, awaiting decisions · **Supersedes nothing** — this is a delta on top of [`2026-08-02-ingest-dicom-pipeline.md`](2026-08-02-ingest-dicom-pipeline.md) and [`2026-08-01-dialygo-realignment.md`](2026-08-01-dialygo-realignment.md).

Five parallel read-only audits ran against `main@2c9c854` and `feat/ingest-dicom-pipeline@874a713`. Every finding below was verified by executing code, not by reading prose. Commit freeze is in force; nothing here has been applied.

---

## 0. State of the world

| | |
|---|---|
| `main` | `2c9c854`, clean, **374 passing** |
| `feat/ingest-dicom-pipeline` (worktree, unmerged, unpushed) | `874a713`, clean, **470 passing** (+96 ingest) |
| Ingest plan | 16 tasks — **1–5 built**, 6–16 not started |
| Legal gates B5/B9 | **both unexecuted** (`configs/ingest_clearance.yaml` = `false`/`false`) |

**The one-line summary:** the ingest foundation is sound and its tests are genuinely adversarial, but the safety property the whole design rests on — "real patient data cannot be processed until the agreements execute" — **is not true today**, and the patient-grouping guard that prevents the project's signature failure mode **does not cover the AVF naming convention it is about to receive**.

---

## P0 — Safety. Must land before any real drive is ever plugged in.

These are ordered by how badly they fail. Every one was demonstrated live.

### P0.1 The B5/B9 clearance gate does not control data access

`--mode` defaults to `"synthetic"` on both CLIs (`scan.py:214`, `index_dicom.py:218`), and `require_clearance` returns at `clearance.py:59-60` **without reading the marker**. The command in the module's own docstring succeeds against a real drive:

```
$ python -m src.ingest.scan --src /Volumes/CATHLAB_HANDOVER --out .ingest
{"counts": {"dicom": 1}, "mode": "synthetic", ...}          # no refusal
$ python -m src.ingest.index_dicom --files .ingest/files.jsonl --out .ingest
→ dicom_index.jsonl now holds PatientName, PatientID, PatientBirthDate,
  AccessionNumber, ReferringPhysicianName for the whole cohort
```

Second bypass: `--clearance` accepts any path, so a two-line YAML written anywhere opens the gate. The plan's claim at `:1318-1320` — "the only way to point this at a real drive is to type `--mode real` *and* have a countersigned marker" — is refuted.

The fail-safe *parsing* is excellent and should not be touched: 17 malformed variants (quoted `"true"`, `1`, `null`, `[true]`, typo'd keys, non-dict root, YAML deserialization attack) all refuse correctly.

**Fix:**
1. Make `--mode` **required, no default**, on every ingest CLI.
2. Resolve the marker from a module constant; ignore `--clearance` unless a `--clearance-override-for-tests` flag is passed.
3. Corroborate the `synthetic` claim instead of trusting it: refuse any `--src` root not under a configured synthetic-data directory, so declaring "synthetic" while pointing at `/Volumes/...` fails.
4. Push `require_clearance` down to the six ungated functions that read patient bytes: `index_dicom.read_header`, `scan.is_dicom`, `scan.classify`, `scan._row`, `manifest.head_key`, `manifest.sha256_file`. The gate currently lives only at two orchestrating entry points, so every new caller silently inherits no gate.

### P0.2 AVF frames will split per-frame, and the leakage auditor will certify it

`group_key` (`io_utils.py:23-34`) has no AVF branch. Measured, not reasoned:

```
one patient, 200 frames  →  200 groups  →  165 train / 35 val
audit_split_leakage(...)  →  PASSES, reports val_frac_by_group = 0.175
```

`audit_split_leakage`'s only leak test is "does a group appear on both sides", which is **vacuous for a per-frame convention** — every group is unique by construction. So the first real cohort prints "LEAKAGE CHECK PASSED — split is patient-grouped and honest" while training on memorised neighbouring frames. This is the exact mechanism behind F1 0.885 → 0.214.

**Fix:** ship plan Task 12 (`_AVF_RE`). Verified: applying the patch in-process leaves the **full 470-test suite green**, and Danilov/CADICA/CathAction/ARCADE grouping is byte-for-byte unchanged. Additionally add an `avf_stems=` guard to `audit_split_leakage` mirroring `danilov_stems`/`cathaction_stems`, so a future regex no-op is detected rather than blessed.

### P0.3 The re-identification key is not gitignored under its own planned name

`*.crosswalk.csv` matches `site_a.crosswalk.csv` but **not** `crosswalk.csv` or `deid_crosswalk.csv` — proven by staging in a throwaway repo. The plan's canonical layout names the file `_keys/crosswalk.csv` (`:69`).

**Fix:** add `crosswalk.csv`, `*crosswalk*.csv`, `_keys/`, and `data/interim/` to `.gitignore`.

### P0.4 None of the ingest ignore patterns exist on `main`

`.ingest/`, `*.crosswalk.csv`, `*.salt`, `salt.bin` live only on the feature branch, while `scan.py:213` defaults its output to `<repo>/.ingest/`. Anyone running ingest tooling from a `main` checkout, or a merge that drops `.gitignore:30-34`, stages PHI.

**Fix:** land the `.gitignore` block on `main` **first**, independently of the ingest merge.

### P0.5 `index_dicom --out` writes raw PHI to an unvalidated path

`index_dicom.py:164` does a bare `mkdir` on caller-supplied `out_dir`; `INDEX_TAGS` deliberately retains every identifier. The docstring at `:27-28` asserts the index "lives on the cleared drive … never in the repo" — enforced by nothing. `--out .` puts it in the repo tree.

**Fix:** refuse any `out_dir` inside the repo root when `mode == "real"`; `chmod 0600` the index (it currently lands `0644` while the harmless summary is `0600` by accident of `mkstemp`).

### P0.6 Corrupt DICOM is dropped with no record and no count

`build_index` `continue`s on `read_header → None` for three different situations — non-DICOM, missing `SOPInstanceUID`, genuinely corrupt — identically and invisibly. `counts` exposes `n_files_seen` and `n_dicom` but never the number of rows typed `dicom`, so the drop count is not even derivable.

Scenario: 12 of patient INU-00902's 40 instances truncated by a failed CD burn → index looks clean. If all of one patient's files are damaged, that patient vanishes from the cohort with zero signal. Task 6's PHI audit reads this index, so the corruption never reaches human review.

**Fix:** `index_errors.jsonl` rows `{path, reason, exception}` for every `None`, plus `n_dicom_rows_seen` / `n_unparsed` in `counts`.

### P0.7 An unreadable directory is silently omitted

`os.walk` with default `onerror=None` discards scandir errors. A mode-`000` directory (routine on vendor NTFS/exFAT dumps) containing DICOM → `n_files: 1`, exit 0, no error row. Contradicts `scan.py:15-18`.

**Fix:** pass an `onerror=` callback emitting `{"path": dir, "kind": "unreadable_dir", "error": …}`.

### P0.8 Smaller, same batch

- `scan_tree` does not validate roots — a typo'd drive path yields a clean, confident, empty inventory (`{"n_files": 0}`, exit 0), indistinguishable from "drive is empty". Raise instead.
- Durability inversion: `append_jsonl` never fsyncs while `write_json_atomic` does, so after power loss the checkpoint vouches for rows that were lost — resume then skips the directory forever. Fsync `files.jsonl` before each checkpoint.
- `read_jsonl` raises `UnicodeDecodeError` on a torn multi-byte line — near-certain on an Indian institutional drive — and neither `scan.py` call site catches it (`json.JSONDecodeError` does not cover it). Open with `errors="replace"`, widen the guard to `ValueError`.
- `INDEX_TAGS` omits PS3.15 identifiers `OperatorsName`, `StationName`, `DeviceSerialNumber`, `PatientAddress`, `ImageComments`, `PatientComments`, `AdditionalPatientHistory` — so the Task 6 PHI audit reports clean on fields it never examined.
- SOP dedupe discards the losing copy's `size`/`head_key`, destroying the only signal that one duplicate is truncated. A 40 MB re-burn beats a 118 MB good copy on lexicographic path order.

---

## P1 — Rulings that unblock ingest Tasks 6–16

Every one verified by executing the plan's code on the installed stack (pydicom 3.0.2, cv2 5.0.0, numpy 2.4.3).

| Task | Verdict | Ruling |
|---|---|---|
| 6 | needs-ruling | Line 2922: fixture emits `Siemens CathLab Model-1`, not `Artis Zee`. → `assert "\| Siemens / Siemens CathLab Model-1 \| 4 \|" in text` |
| 7 | **ready** | — |
| 8 | needs-ruling | Lines 4132/4385 `"Artis Zee"` → `"Siemens CathLab Model-1"`; line 4136 `1200.0` → `1000.0`. Lines 4388/4390 are vacuous (assert absence of strings the fixture never writes) → assert `b"SYNTHETIC^INU-00417"`, `b"Synthetic Regional Dialysis Centre"`, `b"ACC-2024-0517-33"`, `b"MRN-88213"` |
| 9 | **blocked** | With `NumberOfFrames=1`, pydicom returns a **2-D** `pixel_array`, so `ds.pixel_array[0]` is a 1-D row. Line 4627 raises `IndexError`; line 4644 fails; line 4499 passes vacuously. → `n_frames=2` at plan lines 4499, 4620, 4636. Also fix the `_u8` docstring (banner is 4000→250, not 4095) and the negative-origin clip in `mask_regions` |
| 10 | needs-ruling | **Ruling #3 was incomplete** — line 5262 has the same `require_clearance("read")` defect. Delete `_sha256_file` (4985-4990), import `manifest.sha256_file`. `--salt` default → `_keys/salt.bin`. `n_frames=2` at 4856/4858 |
| 11 | needs-ruling | Line 5521 `require_clearance("read")`; import `sha256_file` from manifest at 5492/5527 |
| 12 | **ready** | Leak confirmed live; full 470-suite green with the patch; no over/under-collapse. Optional: reject site slugs outside `[a-z0-9]+` in `stem_prefix` |
| 13 | **ready** | — |
| 14 | **ready** | — |
| 15 | **ready** | — |
| 16 | **blocked** | `MODE=cleared` is not a valid mode (**new defect** — `VALID_MODES = ("synthetic","real")`); Makefile flags for scan/index/deid/extract don't match the built CLIs; `Consumes:` wrongly claims all CLIs take `--mode`; arithmetic says +42, actual +45; DoD omits `pixel_deid` |

**Standing ruling #3 must be widened to Tasks 9, 10 *and* 11.** The three `require_clearance("read")` sites are plan lines 4742, 5262, 5521.

---

## P2 — New tasks. The plan cannot produce its own promised output without these.

Eight gaps found between the canonical layout (plan `:63-76`) and what Tasks 6–16 actually build. Three are blocking:

**Task 17 — series index allocator.** `extract_series`/`extract_video` take `series_idx` as a caller argument and the CLI defaults it to `1`, so **every series of one patient collides on `avf_<site>_<pid>_s01`** — frames and sidecars silently overwrite each other. This is silent data loss. Needs a per-patient enumerator derived from the index's study→series hierarchy.

**Task 18 — Phase-3 batch driver.** Nothing walks `dicom_index.jsonl` → `deid_dataset` → writes the clean DICOM tree. `deid.main()` only provisions a salt, so `make ingest-deid` has no batch CLI to call. Must also call `residual_phi` as the pre-write gate its docstring claims to be, wire `pixel_deid` into `extract_series` (currently screening happens only in `extract.main()`, so any batch caller writes unscreened frames), and gate both `extract_series`/`extract_video` with `require_clearance`.

**Task 19 — crosswalk writer.** `write_crosswalk` is built in Task 7 and **called nowhere**. Without it the cohort is un-re-identifiable, defeating the entire per-patient HMAC design.

Non-blocking gaps to schedule: `qa_review.jsonl` is never written (no aggregated human-review queue); `_manifest/{scan,index,deid,extract}.jsonl` never materialise.

---

## P3 — Serve subsystem: delete the video path

Three verified criticals, all in `analyze_video`, all currently **masked** by `floor_ok: false` — they go live the day Phase A clears the F1 floor.

1. **False "normal" on confident evidence.** `orchestrator.py:253-260` propagates the pre-vote triage only when pre-vote *deferred*. Two 0.9 boxes on separate frames that don't overlap → both tracks dropped by `min_hits=2` → `deferred=False, reason='clean', confidence=0.0`. `analyze_frame` on identical evidence says `confident, 0.9`. This is the false normal B3 forbids. Existing test `:249-278` runs this path and asserts only `boxes == []`, blessing the verdict.
2. **Multi-lesion deletion.** `_flatten_voted` de-dupes by float confidence equality (`:93`) while `_densify` stamps every frame of a track with the *same* conf. Two separate lesions both at 0.8 → 2 survivors → 1 returned. Juxta-anastomotic *plus* cephalic-arch is the exact case AVF fistulography exists to find.
3. **Any positive finding returns HTTP 500.** `infer.py:97-99` emits `numpy.float32` coordinates; `report.to_dict()` never casts; `app.py:143` sits outside the try. Verified: `TypeError: Object of type float32 is not JSON serializable` → 500.

Plus: `router.classify` is called with no try/except (`:151`, `:285`), and the router has no weights (`timm` uninstalled, `runs/router/student.pt` absent) — so the B3 validity gate does not exist in practice, and every real `/analyze` call already fails to `analysis-error`.

**Recommendation: delete the video path for Model One.** B3's input is a single still frame. `analyze_video`, `_iter_frames`, `_flatten_voted`, the box adapters, `temporal_vote.py`, `track.py`, `realtime.py`, `stenosis_infer.py` are all dead weight — and criticals 1 and 2 live entirely inside them. Deleting removes both outright rather than fixing them.

Fix regardless of that decision: the `float32` cast in `report.to_dict()`, and a `RouterUnavailable` fail-safe mirroring `ModelUnavailable`.

**Reuse verdict for the AVF pivot:**

| | Files |
|---|---|
| **As-is** | `report.py`, `registry.py` (its fail-safe `floor_ok` parse is exactly the gate B3 wants), `eval/calibration.py`, `eval/audit.py`, `preprocess.clahe_unsharp` |
| **Adapt** | `router.py` (right shape, needs AVF label set + real weights), `stenosis_triage.calibrate_confidences` (ports directly to a binary logit), `diagnosis.det_to_findings` → `cls_to_finding`, `orchestrator.analyze_frame` half, `app.py` (drop localhost/CoreML posture for B8) |
| **Discard** | the entire video path — see above |
| **Build new** | `src/train/train_cls.py` (frozen DINOv2/v3 + linear head, patient-grouped split, temperature fit), `src/serve/infer_cls.py` (hosted torch — `infer.py` is CoreML-only), `src/eval/cls_metrics.py` (sensitivity/specificity/AUROC/ECE), the validity-gate model + its training set, hosted deployment target, clinician UI |

**Confirmed: the repo has no classification training path.** Zero hits for `dinov2|dinov3|rad_dino` anywhere in `src/`. `train_audio.py:4` is the only classification-adjacent file and it raises `NotImplementedError`. `timm` is still absent from `requirements.txt` despite T1.9 being the task that flagged it.

---

## P4 — Documentation reconciliation

`docs/PROJECT_TRACKER.md`, the declared single source of truth, still describes a coronary edge-deployment project: Stage 4 AVF "not started, data-gated", AVF imaging specified as "lightweight U-Net from coronary weights" (`:142-148`), the golden invariant assuming a procedure cart (`:13`), and a test count of 150 against an actual 374 (`:32`). A new reader would be sent down the wrong critical path. Meanwhile the realignment plan that corrects it is still stamped "proposed — not yet accepted" (`:4`).

14 further contradictions were catalogued. The ones that change decisions: Danilov patient count 100 vs 64; DSA floor 0.80 vs 0.85; `Model_Pipeline_Playbook.md:68` still mandates Dice ≥ 0.75 for AVF imaging while `avf_fistulography.yaml:12` says "NOT Dice"; `src/serve/app.py:1-9` documents an air-gapped localhost service while B8 mandates hosted/central.

`docs/HOSTING_QUESTIONNAIRE.md` is **1 byte** (a single space), untouched since 2026-07-02, while B8 mandates hosted serving and the only serving code in the repo is an edge/CoreML wrapper. It needs: hosting jurisdiction and physical location (India/CDSCO-relevant); whether inference runs inside the Institute's network and if not, exactly what leaves it (B5 forbids processing outside the agreement's environment — this determines whether hosted serving is even *legal*); who holds weights at rest and how non-distribution is technically enforced; PHI-in-transit posture; retention for uploads and for `runs/audit.jsonl` (currently an ungoverned local append); auth model; failure mode when unreachable; and the DINOv3 licence terms that force hosted serving in the first place.

---

## Decisions required

1. **Commit freeze** — lift, or keep holding? Nothing can land until it lifts.
2. **`874a713`** — keep, or `git reset --soft HEAD~1`?
3. **P0 sequencing** — fix all of P0 before resuming Task 6, or resume Tasks 6–8 (which touch no real data) in parallel with P0? The gates only matter when a drive is plugged in, and no drive is connected.
4. **Video path** — delete for Model One, or keep and fix the two criticals?
5. **Track 0** — has either agreement been executed? The realignment plan asked this on 2026-08-01 (`:197-199`) and it has never been answered. Everything in Track 3 and every real metric depends on it.
