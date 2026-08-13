"""Phase 5 — point the repo at the clean frame store. Never copy into it.

Dialygo B5: patient data stays inside the environment the institutional
data-use agreement governs. The repo holds a symlink under the already-gitignored
data/raw/, so no pixel can reach version control. A real directory at the link
destination means somebody copied data into the repo — that raises, it is never
overwritten.
"""
from __future__ import annotations

import os
from pathlib import Path

from src.ingest.clearance import require_clearance

DEFAULT_LINK_NAME = "avf_fistulography"


class LinkError(RuntimeError):
    """Raised when the link destination holds something that is not a symlink."""


def link_site(clean_frames_dir, repo_data_raw, name) -> str:
    """Create/replace `<repo_data_raw>/<name>` -> `<clean_frames_dir>`.

    Idempotent for an existing symlink. Refuses to touch a real file/dir.
    """
    target = Path(clean_frames_dir).expanduser()
    try:
        target = target.resolve()
    except OSError:
        target = target.absolute()
    root = Path(repo_data_raw).expanduser()
    dest = root / str(name)

    root.mkdir(parents=True, exist_ok=True)

    if os.path.lexists(dest):
        if not dest.is_symlink():
            raise LinkError(
                f"refusing to replace non-symlink path {dest}: real data in the "
                f"repo is exactly what this phase prevents (B5). Move or delete "
                f"it by hand, then re-run."
            )
        dest.unlink()

    dest.symlink_to(target, target_is_directory=True)
    return str(dest)


def unlink_site(path) -> None:
    """Remove the symlink. Never removes the target, never removes a real dir."""
    p = Path(path)
    if not os.path.lexists(p):
        return
    if not p.is_symlink():
        raise LinkError(
            f"refusing to remove non-symlink path {p}: this module never deletes "
            f"real data."
        )
    p.unlink()


def verify_link(path) -> dict:
    """Report link health. Never raises — a dangling link is a normal state."""
    p = Path(path)
    exists = bool(os.path.lexists(p))
    is_symlink = bool(p.is_symlink())
    target = ""
    if is_symlink:
        try:
            target = str(os.readlink(p))
        except OSError:
            target = ""
    resolves = False
    if exists:
        try:
            resolves = bool(p.exists())
        except OSError:
            resolves = False
    return {"exists": exists, "is_symlink": is_symlink,
            "resolves": resolves, "target": target}


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m src.ingest.link",
        description="Symlink the clean frame store into the repo (Dialygo B5).")
    ap.add_argument("--clean-frames", help="<clean_root>/<site>/frames")
    ap.add_argument("--data-raw", default="data/raw")
    ap.add_argument("--name", default=DEFAULT_LINK_NAME)
    ap.add_argument("--unlink", action="store_true",
                    help="remove the symlink (never the target)")
    ap.add_argument("--verify", action="store_true",
                    help="report link health and exit without changing anything")
    ap.add_argument("--mode", default="synthetic",
                    help="synthetic until the B5 data-use agreement executes")
    args = ap.parse_args(argv)

    dest = Path(args.data_raw) / args.name

    if args.verify:
        info = verify_link(dest)
        print(f"[link] {dest} exists={info['exists']} "
              f"is_symlink={info['is_symlink']} resolves={info['resolves']} "
              f"target={info['target'] or '-'}")
        return 0 if info["resolves"] else 1

    require_clearance(args.mode)

    if args.unlink:
        unlink_site(dest)
        print(f"[link] removed {dest} (target untouched)")
        return 0

    if not args.clean_frames:
        ap.error("--clean-frames is required unless --unlink/--verify is given")

    out = link_site(args.clean_frames, args.data_raw, args.name)
    info = verify_link(out)
    print(f"[link] {out} -> {info['target'] or '-'} resolves={info['resolves']}")
    if not info["resolves"]:
        print("[link] WARNING: link is dangling — the external drive is not "
              "mounted. Re-run `make ingest-doctor` once it is plugged in.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
