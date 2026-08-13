"""Ingest health check — what you run when something looks wrong.

Never raises: a health check that crashes tells you nothing, and the moment you
most need output is the moment a check is most likely to throw. Read-only, so it
deliberately does NOT call require_clearance — diagnostics must run in any legal
state.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from src.ingest.link import verify_link

MANIFEST_ARTIFACTS = ("files.jsonl", "scan_state.json", "dicom_index.jsonl")

PHI_SUFFIXES = (".dcm", ".crosswalk.csv", ".salt")
PHI_NAMES = ("salt.bin",)
SKIP_DIRS = (".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".venv")


def _result(name, ok, detail) -> dict:
    return {"name": name, "ok": bool(ok), "detail": str(detail)}


def check_mounted(paths) -> dict:
    """Are the configured drive roots present? The drive is usually unplugged."""
    roots = [str(p) for p in (paths or [])]
    if not roots:
        return _result(
            "mounted", True,
            "no drive_roots configured — expected while the B5 data-use "
            "agreement is unexecuted (synthetic data only)")
    missing = sorted(p for p in roots if not Path(p).is_dir())
    detail = f"{len(roots) - len(missing)}/{len(roots)} drive root(s) present"
    if missing:
        detail += (f"; missing: {', '.join(missing)} "
                   f"— external drive not mounted?")
    return _result("mounted", not missing, detail)


def check_links(data_raw) -> dict:
    """Do the data/raw/ symlinks resolve, or are they dangling?"""
    root = Path(data_raw)
    if not root.is_dir():
        return _result("links", True, f"{root} does not exist yet — nothing linked")
    links = sorted(p for p in root.iterdir() if p.is_symlink())
    if not links:
        return _result("links", True, f"no symlinks under {root}")
    dangling = []
    for p in links:
        info = verify_link(p)
        if not info["resolves"]:
            dangling.append(f"{p.name} -> {info['target'] or '?'}")
    detail = f"{len(links) - len(dangling)}/{len(links)} symlink(s) resolve"
    if dangling:
        detail += (f"; dangling: {', '.join(dangling)} "
                   f"— external drive not mounted?")
    return _result("links", not dangling, detail)


def check_manifest(work_dir) -> dict:
    """Do the working-directory artifacts exist and parse?"""
    work = Path(work_dir)
    if not work.is_dir():
        return _result("manifest", True,
                       f"{work} does not exist yet — no ingest run has happened")
    problems = []
    for name in MANIFEST_ARTIFACTS:
        p = work / name
        if not p.is_file():
            problems.append(f"{name} missing")
            continue
        try:
            text = p.read_text(encoding="utf-8")
            if name.endswith(".jsonl"):
                for i, line in enumerate(text.splitlines(), start=1):
                    if line.strip():
                        json.loads(line)
            elif text.strip():
                json.loads(text)
        except (OSError, UnicodeDecodeError) as exc:
            problems.append(f"{name} unreadable ({type(exc).__name__})")
        except ValueError as exc:
            problems.append(f"{name} does not parse ({exc})")
    detail = (f"{len(MANIFEST_ARTIFACTS) - len(problems)}/"
              f"{len(MANIFEST_ARTIFACTS)} artifact(s) ok in {work}")
    if problems:
        detail += "; " + "; ".join(problems)
    return _result("manifest", not problems, detail)


def _is_phi_file(name: str) -> bool:
    lowered = name.lower()
    return lowered in PHI_NAMES or lowered.endswith(PHI_SUFFIXES)


def _git_check_ignore(root: Path, rel: str) -> bool:
    try:
        proc = subprocess.run(["git", "check-ignore", "-q", rel], cwd=str(root),
                              capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def check_no_phi_in_repo(repo_root) -> dict:
    """The safety check: no real PHI-bearing file may live under the repo.

    Symlinks are fine — pointing at patient data is the design (B5); holding a
    copy is the bug. `data/raw/` must be gitignored, and git is asked directly
    rather than by re-implementing its pattern matching.
    """
    root = Path(repo_root)
    offenders = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_DIRS
                       and not os.path.islink(os.path.join(dirpath, d))]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.is_symlink():
                continue
            if _is_phi_file(fn):
                offenders.append(str(p))
    # `data/raw` alone returns 1 when the dir does not exist on disk, because the
    # rule is the directory pattern `data/raw/`. Try both spellings.
    ignored = (_git_check_ignore(root, "data/raw")
               or _git_check_ignore(root, "data/raw/"))
    parts = []
    if offenders:
        parts.append(f"{len(offenders)} PHI-bearing real file(s): "
                     + ", ".join(sorted(offenders)))
    if not ignored:
        parts.append("`git check-ignore -q data/raw` did not pass — data/raw is "
                     "NOT ignored (or this is not a git work tree)")
    if not parts:
        parts.append("no real .dcm/.crosswalk.csv/.salt/salt.bin under the repo; "
                     "data/raw is gitignored")
    return _result("no_phi_in_repo", not offenders and ignored, "; ".join(parts))


def run_doctor(cfg) -> dict:
    """Run every check. Aggregates; never raises."""
    cfg = dict(cfg or {})
    repo_root = str(cfg.get("repo_root") or ".")
    site = str(cfg.get("site") or "inu")
    data_raw = str(cfg.get("data_raw") or Path(repo_root) / "data" / "raw")
    work_dir = str(cfg.get("work_dir") or Path(repo_root) / ".ingest" / site)
    plan = (
        ("mounted", lambda: check_mounted(cfg.get("drive_roots") or [])),
        ("links", lambda: check_links(data_raw)),
        ("manifest", lambda: check_manifest(work_dir)),
        ("no_phi_in_repo", lambda: check_no_phi_in_repo(repo_root)),
    )
    checks = []
    for name, fn in plan:
        try:
            res = fn() or {}
            checks.append(_result(res.get("name", name), res.get("ok", False),
                                  res.get("detail", "")))
        except Exception as exc:  # a crashing check must still report
            checks.append(_result(name, False,
                                  f"check raised {type(exc).__name__}: {exc}"))
    return {"ok": all(c["ok"] for c in checks), "checks": checks}


def main(argv=None) -> int:
    import argparse

    import yaml

    ap = argparse.ArgumentParser(
        prog="python -m src.ingest.doctor",
        description="Ingest health check (read-only; runs in any legal state).")
    ap.add_argument("--config", default="configs/ingest_sites.yaml")
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args(argv)

    cfg = {}
    cfg_path = Path(args.config)
    if cfg_path.is_file():
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            print(f"[doctor] config {cfg_path} unreadable ({exc}) — "
                  f"falling back to defaults")
            cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}
    cfg.setdefault("repo_root", args.repo_root)

    report = run_doctor(cfg)
    for c in report["checks"]:
        flag = "ok  " if c["ok"] else "FAIL"
        print(f"[{flag}] {c['name']}: {c['detail']}")
    print(f"[doctor] ok={report['ok']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
