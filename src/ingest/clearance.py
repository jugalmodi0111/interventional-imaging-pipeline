"""Dialygo B5/B9 clearance gate: refuse to process real patient data until both agreements execute.

B5 (institutional data agreement) and B9 (IP/engagement agreement) are legal preconditions, and a
prose reminder in a requirements doc does not stop `--src /Volumes/CATHLAB_HANDOVER` from being
typed. This module turns both into a precondition every ingest entry point calls before it opens a
path.

Fail-safe parsing, identical in spirit to src/serve/registry.py's floor_ok: PyYAML's SafeLoader
only resolves the unquoted tokens true/false/yes/no/on/off to real booleans, so a quoted "true",
the number 1, a typo, null, a list or a mapping all load as something that is NOT the `True`
singleton. Only `value is True` opens the gate. A missing file, unreadable YAML, a non-mapping top
level, a missing key, a malformed value, or an unrecognized mode string all degrade to refusal --
never to a confident-looking success.

CLI:  python -m src.ingest.clearance --mode real
"""
import os
from pathlib import Path

import yaml

DATA_FLAG = "data_agreement_executed"     # B5: institutional data-sharing agreement
IP_FLAG = "ip_agreement_executed"         # B9: IP / engagement agreement
DEFAULT_CLEARANCE_PATH = "configs/ingest_clearance.yaml"
VALID_MODES = ("synthetic", "real")


class ClearanceError(RuntimeError):
    """Raised when real-data processing is attempted without executed B5 + B9 agreements."""


def read_clearance(path):
    """Load the clearance marker as a dict. Any failure yields {} (which is never cleared).

    Returns {} for: a missing/unreadable file, invalid YAML, or a top level that is not a mapping.
    Never raises -- the refusal is expressed by require_clearance, not by a parse error.
    """
    try:
        with open(path) as f:
            c = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return {}
    return c if isinstance(c, dict) else {}


def is_cleared(c):
    """True only if BOTH flags are the genuine YAML boolean `true`. Everything else is False."""
    if not isinstance(c, dict):
        return False
    return c.get(DATA_FLAG) is True and c.get(IP_FLAG) is True


def require_clearance(mode, clearance_path=DEFAULT_CLEARANCE_PATH):
    """Gate an ingest run. Returns None if permitted; raises ClearanceError otherwise.

    mode="synthetic" -> always permitted, marker never read (the whole test suite runs here).
    mode="real"      -> permitted only when read_clearance(clearance_path) is_cleared.
    anything else    -> refused, because an unrecognized mode is a typo, not a permission.
    """
    if mode == "synthetic":
        return None
    if mode not in VALID_MODES:
        raise ClearanceError(
            f"Dialygo B5/B9 REFUSAL: unrecognized ingest mode {mode!r}; expected one of "
            f"{list(VALID_MODES)}. Refusing rather than guessing which one you meant.")

    resolved = os.path.abspath(clearance_path)
    c = read_clearance(resolved)
    if is_cleared(c):
        return None
    raise ClearanceError(
        "Dialygo B5/B9 REFUSAL: real patient data may not be processed until BOTH agreements "
        "have executed.\n"
        f"  B5 institutional data agreement -> {DATA_FLAG} = {c.get(DATA_FLAG)!r}\n"
        f"  B9 IP / engagement agreement    -> {IP_FLAG} = {c.get(IP_FLAG)!r}\n"
        f"  clearance marker read from      -> {resolved}\n"
        "Only an unquoted YAML `true` counts; a quoted \"true\", a 1, a typo, null or a list all "
        "read as NOT executed. Until legal sign-off lands, run with mode='synthetic' against the "
        "synthetic fixture (tests/fixtures/synthetic_dicom.py).")


def refuse_synthetic_against_mounted_drive(mode, paths):
    """Corroborate a `mode="synthetic"` claim against the paths it is actually about to touch.

    `mode` is a string the caller typed on the command line, not a fact `require_clearance` can
    verify on its own -- and the audit proved that a two-line synthetic claim happily walked a real
    cathlab export. Mirrors `scripts/ingest_hdd.py`'s `check_paths`: any path that resolves under
    "/Volumes/" (a mounted drive on macOS, where every real handover in this project is mounted)
    while `mode == "synthetic"` is refused with a ClearanceError. A no-op for every other mode --
    `mode == "real"` already requires a signed B5/B9 marker via `require_clearance`, which is a
    strictly stronger check than a path prefix.

    `paths` may contain `None`/empty entries (a malformed row, an unset field); those are ignored
    rather than raised on, since they carry no location to corroborate.
    """
    if mode != "synthetic":
        return
    hits = sorted({
        str(p) for p in paths
        if p is not None and str(p).strip() != ""
        and str(Path(str(p)).resolve()).startswith("/Volumes/")
    })
    if hits:
        raise ClearanceError(
            "Dialygo B5/B9 REFUSAL: mode='synthetic' was declared, but the following path(s) "
            "resolve under /Volumes/ (a mounted drive):\n  " + "\n  ".join(hits) + "\n"
            "Declaring synthetic while pointing at a mounted drive is exactly the bypass the "
            "audit flagged (P0.1 fix #3) -- a two-line YAML would then be all that stood between "
            "this run and real patient data. Use mode='real' with an executed B5/B9 clearance "
            "marker instead."
        )


def main():
    """CLI: report the clearance state and exit non-zero if the requested mode is refused."""
    import argparse

    ap = argparse.ArgumentParser(description="Check the Dialygo B5/B9 ingest clearance gate.")
    ap.add_argument("--mode", default="real", choices=list(VALID_MODES))
    ap.add_argument("--clearance", default=DEFAULT_CLEARANCE_PATH)
    a = ap.parse_args()

    resolved = os.path.abspath(a.clearance)
    c = read_clearance(resolved)
    print(f"clearance marker: {resolved}")
    print(f"  {DATA_FLAG} = {c.get(DATA_FLAG)!r}   (B5)")
    print(f"  {IP_FLAG} = {c.get(IP_FLAG)!r}   (B9)")
    try:
        require_clearance(a.mode, resolved)
    except ClearanceError as e:
        print(str(e))
        return 1
    print(f"OK: mode={a.mode!r} is permitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
