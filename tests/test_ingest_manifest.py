"""Hashing, append-only JSONL, atomic state, provenance -- the ingest bookkeeping primitives."""
import hashlib
import json
import os
import re

from src.ingest import manifest


def test_schema_version_is_one():
    assert manifest.SCHEMA_VERSION == 1


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "blob.bin"
    data = os.urandom(4096)
    p.write_bytes(data)
    assert manifest.sha256_file(str(p)) == hashlib.sha256(data).hexdigest()


def test_sha256_file_chunk_size_does_not_change_digest(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"x" * 100_000)
    assert manifest.sha256_file(str(p), chunk=7) == manifest.sha256_file(str(p))


def test_head_key_format(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"abc")
    key = manifest.head_key(str(p))
    assert re.fullmatch(r"3:[0-9a-f]{64}", key)
    assert key.split(":")[1] == hashlib.sha256(b"abc").hexdigest()


def test_head_key_reads_only_first_n_bytes(tmp_path):
    """DELIBERATE collision: same size + same first 64K -> same key, tails ignored.

    This is the whole point (vendor CDs duplicate series constantly and full-hashing a terabyte to
    find them is wasteful). sha256_file is the tie-breaker when a group must be PROVEN identical.
    """
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    head = b"H" * 65536
    a.write_bytes(head + b"tail-one")
    b.write_bytes(head + b"tail-two")
    assert manifest.head_key(str(a)) == manifest.head_key(str(b))
    assert manifest.sha256_file(str(a)) != manifest.sha256_file(str(b))


def test_head_key_differs_on_size(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"H" * 65536 + b"xx")
    b.write_bytes(b"H" * 65536 + b"xxx")
    assert manifest.head_key(str(a)) != manifest.head_key(str(b))


def test_head_key_short_file_hashes_whole_file(tmp_path):
    p = tmp_path / "s.bin"
    p.write_bytes(b"tiny")
    assert manifest.head_key(str(p), n=65536) == f"4:{hashlib.sha256(b'tiny').hexdigest()}"


def test_append_jsonl_creates_parent_dirs_and_appends(tmp_path):
    p = tmp_path / "deep" / "nested" / "files.jsonl"
    manifest.append_jsonl(str(p), {"path": "/a", "kind": "dicom"})
    manifest.append_jsonl(str(p), {"path": "/b", "kind": "video"})
    rows = manifest.read_jsonl(str(p))
    assert [r["path"] for r in rows] == ["/a", "/b"]
    assert rows[1]["kind"] == "video"


def test_read_jsonl_missing_returns_empty(tmp_path):
    assert manifest.read_jsonl(str(tmp_path / "nothing.jsonl")) == []


def test_read_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_text('{"a": 1}\n\n   \n{"a": 2}\n')
    assert manifest.read_jsonl(str(p)) == [{"a": 1}, {"a": 2}]


def test_read_jsonl_skips_torn_tail_line(tmp_path):
    """A drive yanked mid-append leaves a half-written line. Drop it, keep the rest."""
    p = tmp_path / "f.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}\n{"a": 3, "pa')
    assert manifest.read_jsonl(str(p)) == [{"a": 1}, {"a": 2}]


def test_read_jsonl_survives_torn_multibyte_utf8_tail(tmp_path):
    """A torn append that lands mid multi-byte UTF-8 character must not raise.

    Institutional drives carry non-ASCII patient/site names constantly; a drive yanked
    mid-append can leave a lone lead byte of a multi-byte sequence with no continuation byte,
    which raises UnicodeDecodeError under strict decoding -- an exception neither scan.py call
    site was built to catch (only json.JSONDecodeError was). Open with errors="replace" so a
    torn tail degrades the same way a torn JSON tail already does: dropped, not fatal.
    """
    p = tmp_path / "f.jsonl"
    good = json.dumps({"a": 1}).encode("utf-8") + b"\n"
    torn = b'{"name": "R\xc3'          # first byte of a 2-byte UTF-8 sequence, no continuation
    p.write_bytes(good + torn)
    assert manifest.read_jsonl(str(p)) == [{"a": 1}]


def test_fsync_file_flushes_without_truncating(tmp_path, monkeypatch):
    """fsync_file must force durability without clobbering what was already written."""
    p = tmp_path / "f.jsonl"
    p.write_text('{"a": 1}\n')

    calls = []
    monkeypatch.setattr(os, "fsync", lambda fd: calls.append(fd))
    manifest.fsync_file(str(p))

    assert calls                                  # os.fsync was actually invoked
    assert p.read_text() == '{"a": 1}\n'          # append-mode open must not truncate content


def test_write_json_atomic_leaves_no_temp_files(tmp_path):
    p = tmp_path / "out" / "state.json"
    manifest.write_json_atomic(str(p), {"done_dirs": ["/a"]})
    assert os.listdir(tmp_path / "out") == ["state.json"]
    assert json.loads(p.read_text()) == {"done_dirs": ["/a"]}


def test_write_json_atomic_overwrites(tmp_path):
    p = tmp_path / "state.json"
    manifest.write_json_atomic(str(p), {"n": 1})
    manifest.write_json_atomic(str(p), {"n": 2})
    assert json.loads(p.read_text()) == {"n": 2}
    assert os.listdir(tmp_path) == ["state.json"]


def test_save_load_state_roundtrip(tmp_path):
    p = str(tmp_path / "scan_state.json")
    state = {"schema_version": 1, "site": "site_a", "done_dirs": ["/x", "/y"]}
    manifest.save_state(p, state)
    assert manifest.load_state(p) == state


def test_load_state_missing_returns_empty(tmp_path):
    assert manifest.load_state(str(tmp_path / "absent.json")) == {}


def test_load_state_corrupt_returns_empty(tmp_path):
    p = tmp_path / "scan_state.json"
    p.write_text('{"done_dirs": ["/x", ')
    assert manifest.load_state(str(p)) == {}


def test_load_state_non_mapping_returns_empty(tmp_path):
    p = tmp_path / "scan_state.json"
    p.write_text("[1, 2, 3]")
    assert manifest.load_state(str(p)) == {}


def test_provenance_has_required_keys():
    p = manifest.provenance("ingest.scan")
    assert p["tool"] == "ingest.scan"
    assert p["schema_version"] == manifest.SCHEMA_VERSION
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", p["utc"])
    assert p["python"].startswith("3.")
    assert isinstance(p["git_commit"], str) and p["git_commit"]


def test_provenance_extra_kwargs_included():
    p = manifest.provenance("ingest.scan", site="site_a", roots=["/Volumes/X"])
    assert p["site"] == "site_a"
    assert p["roots"] == ["/Volumes/X"]


def test_provenance_git_commit_is_short_hash_or_unknown():
    assert re.fullmatch(r"[0-9a-f]{7,40}|<unknown>", manifest.provenance("t")["git_commit"])


def test_provenance_git_commit_falls_back_when_git_missing(monkeypatch):
    """Outside a checkout (or with no git binary) provenance still returns -- it is metadata."""
    def boom(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(manifest.subprocess, "run", boom)
    assert manifest.provenance("ingest.scan")["git_commit"] == "<unknown>"


def test_main_prints_row_and_state_counts(tmp_path, monkeypatch, capsys):
    jl = tmp_path / "files.jsonl"
    manifest.append_jsonl(str(jl), {"path": "/a", "kind": "dicom"})
    manifest.append_jsonl(str(jl), {"path": "/b", "kind": "image"})
    st = tmp_path / "scan_state.json"
    manifest.save_state(str(st), {"done_dirs": ["/a", "/b", "/c"]})
    monkeypatch.setattr("sys.argv", ["manifest", "--jsonl", str(jl), "--state", str(st)])
    assert manifest.main() == 0
    out = capsys.readouterr().out
    assert "2 rows" in out
    assert "done_dirs=3" in out
