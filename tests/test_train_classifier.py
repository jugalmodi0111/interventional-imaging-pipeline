"""Trainer scaffolding: dataset loading + the patient-grouped split (B5: split by patient, never
by frame). Synthetic frame store on tmp_path -- no real data, no network, CPU only."""
import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.train.train_classifier import grouped_split, load_examples


def _store(tmp_path, stems_frames_labels):
    frames = tmp_path / "frames"
    rows = []
    for stem, n, label in stems_frames_labels:
        d = frames / stem
        d.mkdir(parents=True)
        for i in range(n):
            cv2.imwrite(str(d / f"f{i:05d}.png"), np.full((32, 32), 60 + 60 * label, np.uint8))
        rows.append({"key": stem, "label": label})
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return frames, labels


def test_load_examples_one_row_per_frame_with_patient_group(tmp_path):
    frames, labels = _store(tmp_path, [("avf_inu_aaaaaaaaaa_s01", 3, 1),
                                       ("avf_inu_bbbbbbbbbb_s01", 2, 0)])
    ex = load_examples(frames, labels)
    assert len(ex) == 5
    assert {e["group"] for e in ex} == {"avf_inu_aaaaaaaaaa", "avf_inu_bbbbbbbbbb"}
    assert all(e["label"] in (0, 1) and e["path"].endswith(".png") for e in ex)


def test_two_series_from_one_patient_collapse_to_one_group(tmp_path):
    """The regression this module exists to prevent. The labels JSONL keys are SERIES stems
    (avf_..._s01, avf_..._s02); grouping on the series would put one patient's two studies in
    different splits -- audit P0.2 / A1a, the third time this repo has met that bug."""
    frames, labels = _store(tmp_path, [("avf_inu_aaaaaaaaaa_s01", 2, 1),
                                       ("avf_inu_aaaaaaaaaa_s02", 2, 1)])
    ex = load_examples(frames, labels)
    assert {e["group"] for e in ex} == {"avf_inu_aaaaaaaaaa"}
    train, val = grouped_split(ex, val_frac=0.5, seed=0)
    assert not ({e["group"] for e in train} & {e["group"] for e in val})
    assert (len(train) == 4) != (len(val) == 4)      # all four frames land on ONE side


def test_unrecognized_stem_grammar_falls_back_to_the_series_key(tmp_path):
    """A key group_key has no rule for must degrade to one group per SERIES -- never to one group
    per FRAME, which is the shape that silently passes a group-overlap audit while leaking."""
    frames, labels = _store(tmp_path, [("weird_stem_x", 3, 0)])
    ex = load_examples(frames, labels)
    assert {e["group"] for e in ex} == {"weird_stem_x"}


def test_load_examples_skips_labels_with_no_frames_on_disk(tmp_path):
    frames, labels = _store(tmp_path, [("avf_inu_aaaaaaaaaa_s01", 2, 1)])
    labels.write_text(labels.read_text() + json.dumps({"key": "avf_inu_gone_s01", "label": 0}) + "\n")
    assert len(load_examples(frames, labels)) == 2


def test_grouped_split_never_splits_a_patient(tmp_path):
    trips = [(f"avf_inu_{i:010x}_s01", 4, i % 2) for i in range(10)]
    frames, labels = _store(tmp_path, trips)
    train, val = grouped_split(load_examples(frames, labels), val_frac=0.3, seed=1)
    tg, vg = {e["group"] for e in train}, {e["group"] for e in val}
    assert tg and vg and not (tg & vg)


def test_grouped_split_is_deterministic(tmp_path):
    trips = [(f"avf_inu_{i:010x}_s01", 2, i % 2) for i in range(6)]
    frames, labels = _store(tmp_path, trips)
    ex = load_examples(frames, labels)
    a = grouped_split(ex, val_frac=0.5, seed=42)
    b = grouped_split(ex, val_frac=0.5, seed=42)
    assert [e["path"] for e in a[1]] == [e["path"] for e in b[1]]


# --- Task 4: training loop, calibration, artifacts, CLI ------------------------------------


def test_train_end_to_end_writes_artifacts_and_learns(tmp_path):
    torch = pytest.importorskip("torch")
    from src.train.train_classifier import train
    trips = ([(f"avf_inu_{i:010x}_s01", 6, 1) for i in range(4)]
             + [(f"avf_inu_{i + 8:010x}_s01", 6, 0) for i in range(4)])
    frames, labels = _store(tmp_path, trips)      # label 1 -> bright frames, 0 -> dark (separable)
    m = train(frames, labels, tmp_path / "run", backbone="test-tiny", imgsz=32,
              epochs=8, val_frac=0.45, seed=3)
    assert (tmp_path / "run" / "head.pt").exists()
    assert (tmp_path / "run" / "metrics.json").exists()
    assert m["auroc"] >= 0.9                       # brightness is trivially separable
    assert m["threshold_policy"] == "threshold-unsigned"     # no target given (floor unsigned)
    ckpt = torch.load(tmp_path / "run" / "head.pt", weights_only=False)
    assert ckpt["backbone"] == "test-tiny" and 0 < ckpt["temperature"]
    assert ckpt["defer_band"] == [0.3, 0.6]


def test_train_with_target_sensitivity_selects_threshold_from_val(tmp_path):
    pytest.importorskip("torch")
    from src.train.train_classifier import train
    trips = ([(f"avf_inu_{i:010x}_s01", 6, 1) for i in range(4)]
             + [(f"avf_inu_{i + 8:010x}_s01", 6, 0) for i in range(4)])
    frames, labels = _store(tmp_path, trips)
    m = train(frames, labels, tmp_path / "run", backbone="test-tiny", imgsz=32,
              epochs=8, val_frac=0.45, seed=3, target_sensitivity=1.0)
    assert m["threshold_policy"] == "at-target-sensitivity"
    assert m["sensitivity"] == 1.0                 # by construction of the threshold rule


def test_train_records_group_counts_not_just_frame_counts(tmp_path):
    """metrics.json must carry PATIENT counts: 18 train frames from 3 patients is a different
    claim from 18 frames from 18 patients, and only the group count says which (B5/B6)."""
    pytest.importorskip("torch")
    from src.train.train_classifier import train
    trips = ([(f"avf_inu_{i:010x}_s01", 6, 1) for i in range(4)]
             + [(f"avf_inu_{i + 8:010x}_s01", 6, 0) for i in range(4)])
    frames, labels = _store(tmp_path, trips)
    m = train(frames, labels, tmp_path / "run", backbone="test-tiny", imgsz=32,
              epochs=2, val_frac=0.45, seed=3)
    assert m["n_train_groups"] + m["n_val_groups"] == 8
    assert m["n_train_frames"] == 6 * m["n_train_groups"]
    assert m["n_val_frames"] == 6 * m["n_val_groups"]


def test_cli_smoke(tmp_path):
    pytest.importorskip("torch")
    from src.train.train_classifier import main
    trips = [("avf_inu_aaaaaaaaaa_s01", 4, 1), ("avf_inu_bbbbbbbbbb_s01", 4, 0),
             ("avf_inu_cccccccccc_s01", 4, 1), ("avf_inu_dddddddddd_s01", 4, 0)]
    frames, labels = _store(tmp_path, trips)
    rc = main(["--frames", str(frames), "--labels", str(labels), "--out", str(tmp_path / "o"),
               "--backbone", "test-tiny", "--imgsz", "32", "--epochs", "2", "--val-frac", "0.5"])
    assert rc == 0 and (tmp_path / "o" / "metrics.json").exists()


# --- Task 7: the whole Model One path composes ------------------------------------------------


def test_trained_head_serves_through_the_orchestrator(tmp_path, monkeypatch):
    """train -> head.pt -> ClsModel -> analyze_frame: the full Model One path, synthetic only.

    ModalityDecision comes from src.serve.validity, not the deleted src.serve.router (the video
    path and the router went on 2026-08-13/16; the gate kept the same .classify protocol).
    """
    pytest.importorskip("torch")
    from src.serve import orchestrator as orch_mod
    from src.serve.orchestrator import DiagnosticOrchestrator, _model_factory
    from src.serve.registry import TaskEntry
    from src.serve.validity import ModalityDecision
    from src.train.train_classifier import train

    monkeypatch.setattr(orch_mod, "record", lambda *a, **k: None)
    trips = ([(f"avf_inu_{i:010x}_s01", 6, 1) for i in range(4)]
             + [(f"avf_inu_{i + 8:010x}_s01", 6, 0) for i in range(4)])
    frames, labels = _store(tmp_path, trips)
    train(frames, labels, tmp_path / "run", backbone="test-tiny", imgsz=32, epochs=8,
          val_frac=0.45, seed=3)

    entry = TaskEntry("avf_fistulography", "cls", str(tmp_path / "run" / "head.pt"),
                      "AVF fistulography", "avf_ja_stenosis", "Possible JA stenosis",
                      floor_ok=True)

    class R:
        def classify(self, frame):
            return ModalityDecision("avf_fistulography", None, True, 0.9, False, "confident")

    orch = DiagnosticOrchestrator(R(), {"avf_fistulography": entry}, _model_factory)
    report = orch.analyze_frame(np.full((64, 64), 120, dtype=np.uint8))   # bright = positive class
    assert report.modality == "avf_fistulography"
    assert report.findings and report.findings[0].label == "avf_ja_stenosis"
    # No assertion on deferred: with a defer band this small model may legitimately abstain.
    # The claim under test is the PLUMBING: trained artifact -> real factory -> typed finding.


def test_shipped_registry_declares_the_cls_path_with_floors_unsigned():
    """configs/orchestrator.yaml must carry the avf entry, and it must ship floor_ok:false --
    B7 floors are null in configs/avf_fistulography.yaml, so a confident AVF call is not yet
    something this repo is allowed to emit."""
    import yaml
    from src.serve.registry import load_registry
    with open("configs/orchestrator.yaml") as f:
        raw = yaml.safe_load(f)["modalities"]["avf_fistulography"]
    assert raw["task"] == "cls" and raw["floor_ok"] is False
    entry = load_registry("configs/orchestrator.yaml")["avf_fistulography"]
    assert entry.task == "cls" and entry.floor_ok is False
