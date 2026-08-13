"""Ingest bookkeeping: content hashing, append-only JSONL, atomic resume state, provenance.

head_key hashes SIZE + the first 64 KB rather than the whole file. Vendor handovers duplicate the
same series across BACKUP/, to_send/ and CD images constantly, and duplicate detection has to run
over the entire drive before anything else can be planned -- full-SHA-256-ing a terabyte to answer
that is ~90 minutes of USB reads for a question the header answers. A DICOM's first 64 KB contains
the file-meta group and the whole identifying header (UIDs, patient, study, series, acquisition),
so equal size + equal head hash means "same instance" for inventory purposes. The tradeoff is
deliberate and tested: same-size files with equal first 64 KB collide. Use sha256_file to PROVE a
candidate group is byte-identical before anything is merged or deleted.

Write durability is split on purpose:
  * files.jsonl is append-only and line-oriented. A drive disconnecting mid-append leaves a torn
    final line; read_jsonl drops it and that one file is re-scanned. Cheap, self-healing.
  * the state checkpoint is a single object rewritten in full after every directory. A half-written
    checkpoint that still parses is WORSE than no checkpoint: resume trusts it and silently skips
    directories that were never scanned, producing a manifest that looks complete. So state goes
    through write_json_atomic (temp file in the same dir -> fsync -> os.replace), and a reader sees
    either the whole old state or the whole new one.
  * append_jsonl itself never fsyncs (that would mean one fsync per file, which is too slow for a
    200k-file drive) -- but the caller MUST fsync files.jsonl via fsync_file before trusting a
    checkpoint that claims those rows are safe. Otherwise the state checkpoint can vouch for rows
    that a power loss only ever put in the OS page cache, and resume then skips that directory
    forever believing it complete.

Module imports are stdlib-only: no torch, no cv2, no pydicom.

CLI:  python -m src.ingest.manifest --jsonl .ingest/files.jsonl --state .ingest/scan_state.json
"""
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone

SCHEMA_VERSION = 1
UNKNOWN_COMMIT = "<unknown>"


def sha256_file(path, chunk=1 << 20):
    """Full SHA-256 of a file, streamed in `chunk`-byte blocks. The expensive, exact answer."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def head_key(path, n=65536):
    """Cheap duplicate key: '<size>:<sha256 of the first n bytes>'.

    Reads at most n bytes. Files shorter than n are hashed whole. Deliberately collides for
    same-size files whose first n bytes match -- see the module docstring.
    """
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(n)
    return f"{size}:{hashlib.sha256(head).hexdigest()}"


def append_jsonl(path, row):
    """Append one JSON object as a line, creating parent directories as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path):
    """Read a JSONL file into a list of dicts. Missing file -> [].

    Blank lines are skipped. Lines that do not parse (a torn tail from an interrupted append) or
    that parse to something other than an object are skipped rather than raising: one dropped row
    means one re-scanned file, whereas raising loses the entire manifest.

    Opened with errors="replace": a torn append can land mid multi-byte UTF-8 character (routine
    on an institutional drive full of non-ASCII patient/site names), which raises
    UnicodeDecodeError under strict decoding. That is exactly the same "one dropped row" failure
    as a torn JSON tail and is handled the same way -- degrade, don't raise.
    """
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def fsync_file(path):
    """Force `path`'s already-written content to durable storage without rewriting it.

    Durability was inverted: write_json_atomic fsyncs the resume checkpoint, but append_jsonl
    never fsyncs, so after a power loss the checkpoint can vouch for files.jsonl rows that were
    only ever in the OS page cache and never reached disk -- resume then trusts the checkpoint
    and skips that directory forever. Opening in append mode ('a') and calling os.fsync on the
    resulting descriptor flushes ALL of that inode's outstanding dirty pages, including ones
    written by earlier, already-closed append_jsonl calls, without touching a single byte of
    content. Call this once before each checkpoint, not once per append -- fsync is expensive.
    """
    with open(path, "a") as f:
        f.flush()
        os.fsync(f.fileno())


def write_json_atomic(path, obj):
    """Write JSON so a reader never observes a partial file: temp -> fsync -> os.replace."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def load_state(path):
    """Load a resume checkpoint. Missing, unreadable, corrupt, or non-object JSON -> {}.

    {} means "nothing is known to be done", i.e. re-scan everything. Re-scanning is idempotent and
    cheap; trusting a damaged checkpoint silently skips unscanned directories.
    """
    try:
        with open(path) as f:
            st = json.load(f)
    except (OSError, ValueError):
        return {}
    return st if isinstance(st, dict) else {}


def save_state(path, state):
    """Atomically persist a resume checkpoint. Stores `state` verbatim (load_state round-trips)."""
    write_json_atomic(path, state)


def _git_commit():
    """Short HEAD hash of the checkout this module lives in, or '<unknown>' outside one."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN_COMMIT
    out = (r.stdout or "").strip()
    return out if r.returncode == 0 and out else UNKNOWN_COMMIT


def provenance(tool, **extra):
    """Stamp an artifact with what produced it. `extra` keys are merged in (and may override)."""
    p = {
        "tool": tool,
        "schema_version": SCHEMA_VERSION,
        "git_commit": _git_commit(),
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": platform.python_version(),
    }
    p.update(extra)
    return p


def main():
    """CLI: print provenance and summarize an existing JSONL manifest and/or state checkpoint."""
    import argparse

    ap = argparse.ArgumentParser(description="Inspect ingest manifest / state artifacts.")
    ap.add_argument("--jsonl", default=None, help="path to a files.jsonl to count")
    ap.add_argument("--state", default=None, help="path to a scan_state.json to summarize")
    a = ap.parse_args()

    print(json.dumps(provenance("ingest.manifest"), sort_keys=True))
    if a.jsonl:
        print(f"{a.jsonl}: {len(read_jsonl(a.jsonl))} rows")
    if a.state:
        st = load_state(a.state)
        print(f"{a.state}: {len(st)} keys, done_dirs={len(st.get('done_dirs', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
