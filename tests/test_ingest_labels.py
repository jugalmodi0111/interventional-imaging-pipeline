"""Label adapters + the index<->label join (Dialygo B7: labels pass through verbatim)."""
import json
from pathlib import Path

import pytest

from src.ingest.labels import (
    is_narrative_column,
    load_coco_labels,
    load_csv_labels,
    load_mask_dir_labels,
    normalize_label,
    split_stem,
)

CSV_LINES = [
    "StudyInstanceUID,Segment,Label,Impression",
    '1.2.840.1,Juxta-Anastomotic,  Significant Stenosis (>50%) ,'
    '"68F on HD via left radiocephalic AVF; poor thrill, referred by Dr K"',
    '1.2.840.2,juxta_anastomotic,Normal,"no significant lesion"',
]


def _write_csv(tmp_path: Path) -> Path:
    p = tmp_path / "batch1_labels.csv"
    p.write_text("\n".join(CSV_LINES) + "\n", encoding="utf-8")
    return p


def test_load_csv_labels_reads_header_and_normalizes(tmp_path):
    rows = load_csv_labels(_write_csv(tmp_path))
    assert len(rows) == 2
    assert rows[0]["key"] == "1.2.840.1"
    assert rows[0]["segment"] == "juxta-anastomotic"
    assert rows[0]["label"] == "significant stenosis (>50%)"
    assert rows[0]["source"].endswith("batch1_labels.csv")
    assert rows[1]["key"] == "1.2.840.2"
    assert rows[1]["label"] == "normal"


def test_load_csv_labels_quarantines_narrative_columns(tmp_path):
    rows = load_csv_labels(_write_csv(tmp_path))
    assert rows[0]["quarantined"] == ["Impression"]
    assert "impression" not in rows[0]
    # the densest PHI carrier must not survive anywhere in the emitted rows
    blob = json.dumps(rows)
    assert "68F" not in blob
    assert "Dr K" not in blob


def test_is_narrative_column_flags_report_prose():
    for name in ("Impression", "report", "Findings_Text", "clinical notes",
                 "History", "Indication", "Conclusion", "Remarks"):
        assert is_narrative_column(name) is True
    for name in ("StudyInstanceUID", "Segment", "Label", "PatientID", "key"):
        assert is_narrative_column(name) is False


def test_normalize_label_is_verbatim_passthrough():
    # B7: strip + lowercase ONLY. No threshold, no vocabulary mapping.
    assert normalize_label("  Moderate ") == "moderate"
    assert normalize_label("50-70%") == "50-70%"
    assert normalize_label("Significant Stenosis (>50%)") == "significant stenosis (>50%)"
    assert normalize_label(None) == ""
    assert normalize_label("") == ""


def test_split_stem_splits_frame_suffix():
    assert split_stem("avf_inu_3f9c21b04e_s01_00012") == ("avf_inu_3f9c21b04e_s01", "00012")
    assert split_stem("avf_inu_3f9c21b04e_s01") == ("avf_inu_3f9c21b04e_s01", "")


def test_load_coco_labels_reads_bbox_and_category(tmp_path):
    doc = {
        "images": [{"id": 1, "file_name": "avf_inu_3f9c21b04e_s01_00012.png"}],
        "categories": [{"id": 7, "name": "Significant Stenosis"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 7,
                         "bbox": [10, 20, 30, 40]}],
    }
    p = tmp_path / "export.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    rows = load_coco_labels(p)
    assert len(rows) == 1
    assert rows[0]["key"] == "avf_inu_3f9c21b04e_s01"
    assert rows[0]["frame"] == "00012"
    assert rows[0]["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert rows[0]["label"] == "significant stenosis"
    assert rows[0]["source"].endswith("export.json")


def test_load_mask_dir_labels_lists_masks(tmp_path):
    d = tmp_path / "masks"
    d.mkdir()
    (d / "avf_inu_3f9c21b04e_s01_00012.png").write_bytes(b"")
    (d / "avf_inu_3f9c21b04e_s01_00013.png").write_bytes(b"")
    rows = load_mask_dir_labels(d)
    assert [r["frame"] for r in rows] == ["00012", "00013"]
    assert {r["key"] for r in rows} == {"avf_inu_3f9c21b04e_s01"}
    assert rows[0]["mask_path"].endswith("avf_inu_3f9c21b04e_s01_00012.png")


def test_missing_inputs_degrade_to_empty_list(tmp_path):
    # fail-safe: an absent/garbled export yields nothing, so every index row
    # surfaces as unmatched instead of a quiet partial success.
    assert load_csv_labels(tmp_path / "nope.csv") == []
    assert load_coco_labels(tmp_path / "nope.json") == []
    assert load_mask_dir_labels(tmp_path / "nope") == []
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_coco_labels(bad) == []


from src.ingest import labels as labels_mod
from src.ingest.labels import join_labels, main, write_labels_jsonl
from src.ingest.manifest import append_jsonl, read_jsonl


def _index_row(uid, series="1.9.1"):
    return {
        "path": f"/vol/drive/{uid}.dcm", "PatientID": "P1",
        "StudyInstanceUID": uid, "SeriesInstanceUID": series,
        "SOPInstanceUID": series + ".1", "Modality": "XA",
        "NumberOfFrames": 30, "Manufacturer": "Siemens", "StudyDate": "20260714",
    }


def test_join_labels_matches_on_key():
    index_rows = [_index_row("1.2.840.1"), _index_row("1.2.840.2", "1.9.2")]
    label_rows = [
        {"key": "1.2.840.1", "segment": "juxta_anastomotic", "label": "significant",
         "source": "b1.csv", "quarantined": []},
        {"key": "1.2.840.2", "segment": "juxta_anastomotic", "label": "normal",
         "source": "b1.csv", "quarantined": []},
    ]
    matched, unmatched_labels, unmatched_index = join_labels(
        index_rows, label_rows, key="StudyInstanceUID")
    assert len(matched) == 2
    assert unmatched_labels == []
    assert unmatched_index == []
    assert matched[0]["key"] == "1.2.840.1"
    assert matched[0]["index_row"]["SeriesInstanceUID"] == "1.9.1"
    assert matched[0]["label_row"]["label"] == "significant"


def test_join_labels_reports_label_row_that_matches_nothing():
    index_rows = [_index_row("1.2.840.1")]
    label_rows = [{"key": "1.2.840.999", "segment": "", "label": "significant",
                   "source": "b1.csv", "quarantined": []}]
    matched, unmatched_labels, unmatched_index = join_labels(
        index_rows, label_rows, key="StudyInstanceUID")
    assert matched == []
    assert len(unmatched_labels) == 1
    assert unmatched_labels[0]["key"] == "1.2.840.999"


def test_join_labels_reports_index_row_no_label_covers():
    index_rows = [_index_row("1.2.840.1"), _index_row("1.2.840.2", "1.9.2")]
    label_rows = [{"key": "1.2.840.1", "segment": "", "label": "normal",
                   "source": "b1.csv", "quarantined": []}]
    matched, unmatched_labels, unmatched_index = join_labels(
        index_rows, label_rows, key="StudyInstanceUID")
    assert len(matched) == 1
    assert unmatched_labels == []
    assert [r["StudyInstanceUID"] for r in unmatched_index] == ["1.2.840.2"]


def test_join_labels_returns_both_unmatched_lists_populated():
    # The failure this module exists to catch: the spreadsheet and the drive
    # disagree in BOTH directions and the join still reports, never drops.
    index_rows = [_index_row("1.2.840.1"), _index_row("1.2.840.2", "1.9.2")]
    label_rows = [
        {"key": "1.2.840.1", "segment": "", "label": "normal",
         "source": "b1.csv", "quarantined": []},
        {"key": "1.2.840.777", "segment": "", "label": "significant",
         "source": "b1.csv", "quarantined": []},
    ]
    matched, unmatched_labels, unmatched_index = join_labels(
        index_rows, label_rows, key="StudyInstanceUID")
    assert len(matched) == 1
    assert len(unmatched_labels) == 1 and len(unmatched_index) == 1
    assert unmatched_labels[0]["key"] == "1.2.840.777"
    assert unmatched_index[0]["StudyInstanceUID"] == "1.2.840.2"


def test_join_labels_rejects_blank_key_or_blank_label():
    index_rows = [_index_row("1.2.840.1")]
    label_rows = [
        {"key": "", "segment": "", "label": "normal", "source": "b1.csv",
         "quarantined": []},
        {"key": "1.2.840.1", "segment": "", "label": "  ", "source": "b1.csv",
         "quarantined": []},
    ]
    matched, unmatched_labels, unmatched_index = join_labels(
        index_rows, label_rows, key="StudyInstanceUID")
    assert matched == []
    assert len(unmatched_labels) == 2
    assert len(unmatched_index) == 1


def test_write_labels_jsonl_roundtrip(tmp_path):
    matched = [{"key": "1.2.840.1", "index_row": _index_row("1.2.840.1"),
                "label_row": {"key": "1.2.840.1", "label": "normal"}}]
    out = write_labels_jsonl(tmp_path / "sub" / "labels.jsonl", matched)
    assert Path(out).is_file()
    back = read_jsonl(out)
    assert len(back) == 1
    assert back[0]["key"] == "1.2.840.1"
    assert back[0]["label_row"]["label"] == "normal"
    assert "provenance" in back[0]
    # empty match set still produces the artifact, so doctor can see it
    out2 = write_labels_jsonl(tmp_path / "empty.jsonl", [])
    assert Path(out2).is_file()
    assert read_jsonl(out2) == []


def test_main_exits_nonzero_when_labels_unmatched(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(labels_mod, "require_clearance", lambda *a, **k: None)
    index_path = tmp_path / "dicom_index.jsonl"
    append_jsonl(index_path, _index_row("1.2.840.1"))
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "StudyInstanceUID,Segment,Label\n1.2.840.999,juxta,significant\n",
        encoding="utf-8")
    rc = main(["--index", str(index_path), "--labels", str(csv_path),
               "--kind", "csv", "--key", "StudyInstanceUID",
               "--out", str(tmp_path / "labels.jsonl"), "--mode", "synthetic"])
    assert rc != 0
    assert "BLOCKING" in capsys.readouterr().out
