# INU HDD Ingest Runbook — sterilise & modularise the fistulography drive

**Script:** `.claude/worktrees/ingest-dicom/scripts/ingest_hdd.py`
**Status:** built and verified on synthetic data (2026-08-09). Nothing has touched the HDD yet. Nothing is committed (commit freeze).

> **Why you don't see the code in VS Code's source control:** all new code lives in a git
> worktree at `.claude/worktrees/ingest-dicom/` on branch `feat/ingest-dicom-pipeline`, and
> `.claude/worktrees/` is gitignored in the main checkout — so the main window only shows this
> doc and the remediation plan. To browse the code: `code .claude/worktrees/ingest-dicom`.

---

## 1. What the script does

One command runs five resumable phases against the drive. Every byte of patient data stays on
the drive — nothing patient-identifying is written inside the repository.

| Phase | What happens | Output |
|---|---|---|
| 1 scan | Read-only inventory of every file (magic-byte typing, no pixel decode) | `intv-img-work/inu/files.jsonl` |
| 2 index | Header-only DICOM index, SOP de-duplication, corrupt-file accounting | `dicom_index.jsonl`, `index_errors.jsonl` |
| 3 audit | PHI audit report — **the run STOPS here for you to read it** | `phi_audit.md` |
| 4 deid | PS3.15 tag scrub, HMAC-SHA256 pseudonyms, per-patient date shift, UID remap; residual-PHI gate quarantines anything that fails | `intv-img-clean/inu/dicom/…`, `_keys/crosswalk.csv` |
| 5 extract | Burned-in text detection + masking, VOI-LUT windowing, PNG frame store with collision-free stems | `frames/avf_inu_<pid>_s<NN>/f00000.png…`, `sidecar/*.json` |

## 2. Before you run — checklist

- [ ] Drive mounted at `/Volumes/INU` (the One Touch copy at `/Volumes/One Touch/Dataset-INU` stays untouched as backup).
- [ ] Free space on the drive: dataset is ~246 GB; de-identified DICOM + PNG output needs roughly the same again. Drive has ~754 GB free — OK.
- [ ] **Legal gate:** edit `configs/ingest_clearance.yaml` (inside the worktree) and set **both** flags to unquoted `true` — `data_agreement_executed` (B5) and `ip_agreement_executed` (B9). Flipping these asserts the signed agreements exist. If they are not signed, stop here; the script will (correctly) refuse.
- [ ] Terminal at the worktree root: `cd .claude/worktrees/ingest-dicom` (from the repo root).
- [ ] Python 3.12 with pydicom 3.0.2, opencv, numpy (your pyenv 3.12.9 default already has all three).

## 3. Commands to run

**Recommended first: a 20-instance pilot** (verifies everything on real data in minutes):

```bash
cd .claude/worktrees/ingest-dicom
python scripts/ingest_hdd.py --src "/Volumes/INU" --site inu --mode real --limit 20
```

**Run 1 — scan, index, PHI audit, then stop:**

```bash
python scripts/ingest_hdd.py --src "/Volumes/INU" --site inu --mode real
```

Read `/Volumes/INU/intv-img-work/inu/phi_audit.md` (identifier density, burned-in flags, drop log).

**Run 2 — the same command plus the acknowledgement flag; runs de-identification + extraction:**

```bash
python scripts/ingest_hdd.py --src "/Volumes/INU" --site inu --mode real --ack-phi-audit
```

Safe to interrupt at any point (Ctrl-C). Re-running the same command resumes where it stopped
and never duplicates work.

## 4. Expected console output

Run 1 (numbers will differ; these markers are what to look for):

```
[gate ] mode=real clearance OK · 40 scan roots (3 top-level entries skipped)
[scan ] phase 1: read-only inventory ...
[scan ] {"dicom": 51234, "video": 12, "image": 830, ...}
[index] phase 2: header-only DICOM index ...
[index] {"n_files_seen": 52076, "n_dicom": 51234, "n_unique_sop": 50419,
         "n_patients": 412, "n_studies": 430, "n_series": 2210}
[index] drop accounting: {"n_dicom_typed": 51234, "n_indexed": 50419,
                          "n_unparsed": 15, "n_sop_duplicates": 800}
[audit] phase 3: PHI audit report ...
[audit] written: /Volumes/INU/intv-img-work/inu/phi_audit.md

STOP — human review checkpoint.
Read /Volumes/INU/intv-img-work/inu/phi_audit.md (identifier density, burned-in flags, drop log).
Then re-run the same command with --ack-phi-audit to run de-identification.
```

Run 2 continues:

```
[deid ] phase 4: PS3.15 tag scrub + HMAC pseudonyms ...
[deid ] {"n_deid_new": 50419, "n_deid_total": 50419, "n_deid_errors": 3,
         "n_quarantined": 0, "n_crosswalk": 412}
[extr ] phase 5: pixel screen + PNG frame store ...
[extr ] {"n_extract_new": 50416, "n_extract_total": 50416,
         "n_extract_errors": 0, "n_flagged_for_review": 210}

DONE. Outputs:
  clean DICOM   /Volumes/INU/intv-img-clean/inu/dicom
  frames        /Volumes/INU/intv-img-clean/inu/frames
  sidecars      /Volumes/INU/intv-img-clean/inu/sidecar
  crosswalk     /Volumes/INU/intv-img-clean/inu/_keys/crosswalk.csv  <- guard this file; it re-identifies
  review queue  /Volumes/INU/intv-img-work/inu/qa_review.jsonl (210 flagged)
  quarantine    /Volumes/INU/intv-img-work/inu/deid_quarantine.jsonl (0 held back)
  full report   /Volumes/INU/intv-img-work/inu/ingest_report.json
```

Timing on a USB HDD: scan + index are file-count-bound (expect tens of minutes to ~2 h);
phase 4 rewrites every DICOM byte (~246 GB read + write — expect several hours); phase 5
depends on total frame count. Leave it running; interrupt + resume is safe.

## 5. What lands where

```
/Volumes/INU/intv-img-work/inu/          working state + logs (contains PHI — stays on drive)
  files.jsonl            one row per file on the drive
  dicom_index.jsonl      header index — PatientID, UIDs, dates (raw PHI)
  index_errors.jsonl     every corrupt/duplicate DICOM, with reason — check its size
  phi_audit.md           the human-review report (run 1 stops here)
  deid_quarantine.jsonl  files that FAILED the residual-PHI gate — must be 0 or investigated
  deid_errors.jsonl      unreadable/unwritable files (logged, batch continues)
  qa_review.jsonl        frames flagged for human review (burned-in text detected/declared)
  ingest_report.json     final counts for everything above

/Volumes/INU/intv-img-clean/inu/         the de-identified, modular dataset
  dicom/<pseudo_patient>/<pseudo_study>/<pseudo_series>/<pseudo_sop>.dcm
  frames/avf_inu_<pid10hex>_s<NN>/f00000.png …      one dir per cine run
  sidecar/avf_inu_<pid10hex>_s<NN>.json             per-run metadata (windowing, fps, provenance)
  _keys/salt.bin           HMAC salt — NEVER leaves the drive, never into git
  _keys/crosswalk.csv      real ID ↔ pseudonym map — the re-identification key. Guard it.
```

## 6. After the run — verify

1. `deid_quarantine.jsonl` — should be empty. Any row = a file whose scrub left residual PHI; it was **not** written to the clean tree. Investigate before sharing anything.
2. `index_errors.jsonl` — skim. A handful of `unparseable_or_missing_sop` rows is normal for CD burns; hundreds clustered in one folder means a damaged patient — check that folder.
3. `qa_review.jsonl` — open a few of the flagged frame PNGs; confirm the masked band covers all burned-in text.
4. Spot-check one clean DICOM: `PatientName` empty, `PatientID` like `inu_37d2eca1c6`, `PatientIdentityRemoved = YES`.
5. `ingest_report.json` — totals line up with the audit counts.

## 7. If it refuses

| Message | Meaning | Fix |
|---|---|---|
| `Dialygo B5/B9 REFUSAL …` | Clearance flags not both `true` | Sign-off first, then edit `configs/ingest_clearance.yaml` |
| `--mode synthetic with --src on /Volumes/` | Declared synthetic against a real drive | Use `--mode real` |
| `--work/--clean-root … is inside the repository` | Real-mode PHI output pointed into the repo | Leave the defaults (they live on the drive) |
| `--src … is not a mounted directory` | Drive not mounted / path typo | Check `ls /Volumes/` |

## 8. Known limits

- Exported videos and the `Anonymised Images/` folder are inventoried but **not processed** — no DICOM header to pseudonymise from. They need manual patient mapping later; counts appear in `ingest_report.json`.
- The drive is exFAT/NTFS: `chmod 0600` on `_keys/` is best-effort there. Physical custody of the drive is the real control.
- **Before any train/val split on these frames:** land the Task 12 `group_key` AVF patch (`src/data_prep/io_utils.py`), or the split will leak neighbouring frames of the same patient across train/val and the leakage auditor will certify it clean — the F1 0.885→0.214 failure mode.
