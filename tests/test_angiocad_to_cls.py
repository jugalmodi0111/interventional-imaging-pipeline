"""TDD for src.data_prep.angiocad_to_cls — AngioCAD severity labels -> classification records.

AngioCAD (Zenodo 10.5281/zenodo.15826856, CC-BY-4.0) is 413 patients, ONE ROW EACH, carrying a
7-grade severity per artery segment. It has NO bounding boxes (verified 2026-08-23 by reading
AngioCAD_Labels.xlsx — see docs/DATASETS.md), so it cannot train the YOLO detector; it is a
patient/segment-level CLASSIFICATION dataset.

THE CORRECTNESS CONSTRAINT these tests exist to pin:
    A video shows ONE coronary side. 83 of 413 patients (20%) have disease on one side and a normal
    other side. Stamping the patient-level label onto every one of their videos would label a
    healthy artery as diseased -- the same class of error as audit A1a (using lesion-video frames as
    negatives). Per-video labels MUST be restricted to the segments that video actually shows, using
    the sheet's own "Right/Left Coronary Series" columns.
"""
import os

import pytest

from src.data_prep.angiocad_to_cls import (LCA_SEGMENTS, RCA_SEGMENTS, build_records,
                                           is_significant, parse_series_spec, severity_band,
                                           side_of_series, video_label)


# --- series spec parsing: the sheet uses "1-2", "7", "3-9" -------------------------------------
@pytest.mark.parametrize("spec,expected", [
    ("1-2", (1, 2)), ("7", (7,)), ("3-9", (3, 4, 5, 6, 7, 8, 9)), ("1", (1,)),
    (" 2-3 ", (2, 3)), (1, (1,)),                      # openpyxl hands back ints for bare numbers
])
def test_parse_series_spec(spec, expected):
    assert tuple(parse_series_spec(spec)) == expected


@pytest.mark.parametrize("bad", [None, "", "  ", "n/a"])
def test_parse_series_spec_tolerates_blanks(bad):
    assert list(parse_series_spec(bad)) == []


# --- severity grades ----------------------------------------------------------------------------
@pytest.mark.parametrize("grade,band", [
    ("NL", (0, 0)), ("1-25", (1, 25)), ("26-50", (26, 50)), ("51-75", (51, 75)),
    ("76-90", (76, 90)), ("91-99", (91, 99)), ("100", (100, 100)),
])
def test_severity_band(grade, band):
    assert severity_band(grade) == band


def test_severity_band_unknown_is_none_not_a_guess():
    # An unrecognised grade must NOT silently become 0 (a false negative on a diseased artery).
    assert severity_band("???") is None
    assert severity_band(None) is None


def test_is_significant_uses_the_whole_band_not_its_midpoint():
    # Positive only if the ENTIRE band clears the threshold, so a grade that merely *might* reach it
    # is not counted as disease. '26-50' tops out AT 50 and is therefore NOT >=51.
    assert is_significant("51-75", 50) is True
    assert is_significant("26-50", 50) is False
    assert is_significant("100", 50) is True
    assert is_significant("NL", 50) is False


def test_is_significant_threshold_is_configurable():
    # 70% is the other common clinical cut; the choice is Dr. Reddy's, not ours.
    assert is_significant("51-75", 70) is False      # band starts below 70
    assert is_significant("76-90", 70) is True


def test_unknown_grade_never_reads_as_negative():
    with pytest.raises(ValueError, match="unknown severity"):
        is_significant("???", 50)


# --- the side constraint ------------------------------------------------------------------------
def _row(rca_series="1-2", lca_series="3-4", **grades):
    """A sheet row as {column: value}; unnamed segments default to NL."""
    r = {"ID": 1, "Right Coronary Series": rca_series, "Left Coronary Series": lca_series}
    for seg in RCA_SEGMENTS + LCA_SEGMENTS:
        r[seg] = "NL"
    r.update(grades)
    return r


def test_segment_lists_are_disjoint_and_cover_the_sheet():
    assert not set(RCA_SEGMENTS) & set(LCA_SEGMENTS)
    assert len(RCA_SEGMENTS) + len(LCA_SEGMENTS) == 15


def test_side_of_series_maps_from_the_sheets_own_columns():
    r = _row(rca_series="1-2", lca_series="3-9")
    assert side_of_series(r, 1) == "rca" and side_of_series(r, 2) == "rca"
    assert side_of_series(r, 3) == "lca" and side_of_series(r, 9) == "lca"
    assert side_of_series(r, 99) is None          # not listed -> unknown, never guessed


def test_an_rca_video_is_NEGATIVE_when_only_the_LCA_is_diseased():
    # THE constraint. 68 real patients look exactly like this. Stamping the patient-level label onto
    # the RCA video would teach the model that a healthy right coronary is diseased.
    r = _row(**{"Prox LAD": "100"})
    assert video_label(r, series=1, threshold=50)["positive"] is False   # RCA view -> negative
    assert video_label(r, series=3, threshold=50)["positive"] is True    # LCA view -> positive


def test_an_lca_video_is_NEGATIVE_when_only_the_RCA_is_diseased():
    r = _row(**{"Mid RCA": "91-99"})
    assert video_label(r, series=3, threshold=50)["positive"] is False
    assert video_label(r, series=1, threshold=50)["positive"] is True


def test_video_label_reports_only_the_segments_that_video_shows():
    r = _row(**{"Prox LAD": "100", "Mid RCA": "76-90"})
    lca = video_label(r, series=3, threshold=50)
    assert set(lca["segments"]) == set(LCA_SEGMENTS)
    assert "Mid RCA" not in lca["segments"]                 # not visible -> not reported
    assert lca["segments"]["Prox LAD"] is True


def test_video_label_of_an_unknown_series_is_refused_not_guessed():
    r = _row()
    assert video_label(r, series=42, threshold=50) is None


def test_patient_level_is_derived_as_any_segment(monkeypatch):
    r = _row(**{"Dist LCX": "51-75"})
    v = video_label(r, series=3, threshold=50)
    assert v["positive"] is (True in v["segments"].values())


# --- manifest building --------------------------------------------------------------------------
def _sheet(tmp_path, rows):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Labels"
    hdr = ["ID", "Right Coronary Series", "Left Coronary Series"] + RCA_SEGMENTS + LCA_SEGMENTS
    ws.append(hdr)
    for r in rows:
        ws.append([r.get(h, "NL") for h in hdr])
    p = os.path.join(str(tmp_path), "labels.xlsx"); wb.save(p)
    return p


def test_build_records_emits_one_record_per_video_series(tmp_path):
    p = _sheet(tmp_path, [_row(rca_series="1-2", lca_series="3", **{"Prox LAD": "100"})])
    recs = build_records(p, threshold=50)
    assert len(recs) == 3                                    # series 1, 2 (rca) + 3 (lca)
    by_series = {r["series"]: r for r in recs}
    assert by_series[1]["positive"] is False and by_series[1]["side"] == "rca"
    assert by_series[3]["positive"] is True and by_series[3]["side"] == "lca"


def test_build_records_carries_the_patient_id_for_grouping(tmp_path):
    p = _sheet(tmp_path, [_row(rca_series="1", lca_series="2")])
    recs = build_records(p, threshold=50)
    assert all(r["patient"] == 1 for r in recs)
    assert all("group_key" in r for r in recs)               # split must group BY PATIENT


def test_build_records_group_key_collapses_a_patients_videos(tmp_path):
    p = _sheet(tmp_path, [_row(rca_series="1-2", lca_series="3-4")])
    recs = build_records(p, threshold=50)
    assert len({r["group_key"] for r in recs}) == 1, "all of one patient's videos share a group"


def test_build_records_threshold_changes_the_labels(tmp_path):
    p = _sheet(tmp_path, [_row(rca_series="1", lca_series="2", **{"Prox LAD": "51-75"})])
    assert build_records(p, threshold=50)[1]["positive"] is True
    assert build_records(p, threshold=70)[1]["positive"] is False


# --------------------------------------------------------------------------------------------
# Comma-separated and reversed series specs. Found 2026-08-26 by running the adapter against the
# real AngioCAD_Labels.xlsx: 20 distinct comma-containing formats across 23 patients (e.g. patient
# 252 RCA "3,4,11,12,13") parsed to [] and vanished, taking 82 videos with them -- the corpus is
# 2,726 videos, not the 2,644 every doc reports. The bug is invisible because a malformed spec
# returned the SAME [] as a genuinely blank cell, which the docstring documents as "missing data,
# not a corrupt sheet". Present, valid, multi-run specs were being conflated with absent ones.
# --------------------------------------------------------------------------------------------


def test_comma_separated_series_are_all_parsed():
    assert parse_series_spec("3,4,11,12,13") == [3, 4, 11, 12, 13]


def test_comma_separated_ranges_expand_and_concatenate():
    assert parse_series_spec("1-3,6-8") == [1, 2, 3, 6, 7, 8]


def test_mixed_singles_and_ranges_parse():
    assert parse_series_spec("2, 5-7, 9") == [2, 5, 6, 7, 9]


def test_duplicates_across_parts_are_collapsed_and_order_is_stable():
    assert parse_series_spec("3-5,4,5") == [3, 4, 5]


def test_a_reversed_range_is_read_as_the_span_it_names():
    """Real data: patient 332 RCA "7-6", patient 353 LCA "8-3". range(7, 6+1) is empty, so both
    silently vanished. A reversed pair still unambiguously names its endpoints."""
    assert parse_series_spec("7-6") == [6, 7]
    assert parse_series_spec("8-3") == [3, 4, 5, 6, 7, 8]


def test_a_genuinely_unparseable_spec_RAISES_instead_of_masquerading_as_blank():
    """The core defect: silence. Blank means "no series recorded"; garbage means "this sheet is
    not what we think it is". Collapsing the second into the first is how 82 videos disappeared
    without a single warning."""
    with pytest.raises(ValueError):
        parse_series_spec("3;4")
    with pytest.raises(ValueError):
        parse_series_spec("seven")


def test_blank_and_none_still_mean_missing_data_not_corruption():
    assert parse_series_spec(None) == []
    assert parse_series_spec("") == []
    assert parse_series_spec("   ") == []
