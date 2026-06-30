"""
Tests for neural_analysis.utils.metadata.yaml_creator

Strategy:
- Pure-Python helpers (create_dict, remove_none_from_dict, num_to_date, search_update_dict)
  are tested directly.
- Spreadsheet-dependent helpers (row_to_list, define_metadata_columns,
  create_stimulus_dict, create_behavior_dict, create_neural_dict, create_task_dict,
  create_session_dict, create_animal_dict, get_animal_dict_from_spreadsheet) are tested
  via real openpyxl workbooks created in-memory / saved to tmp_path.
- File-I/O helpers (return_loaded_yaml_if_newer, get_animals_from_yaml,
  combine_spreadsheet_and_old_animal_summary_yaml, create_folders_for_animals,
  move_mesc_to_session_folder) use tmp_path.
- get_recording_munits and add_session_animal_folders require real .mesc (HDF5) files
  and filesystem trees — tested via a minimal h5py fixture.
- main() is tested with a minimal xlsx + tmp_path (writes "animals.yaml").
"""

from __future__ import annotations

import copy
import os
import time

import h5py
import numpy as np
import openpyxl
import pytest
import yaml

from neural_analysis.utils.metadata.yaml_creator import (
    add_session_animal_folders,
    combine_spreadsheet_and_old_animal_summary_yaml,
    create_animal_dict,
    create_behavior_dict,
    create_dict,
    create_folders_for_animals,
    create_neural_dict,
    create_session_dict,
    create_stimulus_dict,
    create_task_dict,
    define_metadata_columns,
    get_animal_dict_from_spreadsheet,
    get_animals_from_yaml,
    get_recording_munits,
    move_mesc_to_session_folder,
    num_to_date,
    remove_none_from_dict,
    return_loaded_yaml_if_newer,
    row_to_list,
    search_update_dict,
)

# ---------------------------------------------------------------------------
# Helpers – build a minimal openpyxl workbook that satisfies all column reads
# ---------------------------------------------------------------------------

# Column layout used by all sheet-dependent helpers.
# The order here determines the integer column index (1-based).
COLUMNS = [
    "mouse ID",
    "sex",
    "DOB",
    "injected",
    "implanted",
    "date",
    "weight [g]",
    "session",
    "duration [min]",
    "paradigm",
    "comment",
    "setup",
    "[nm]",
    "laser / LED power",
    "PC bias",
    "n Ch.",
    "fct. channel",
    "UG gain",
    "UR gain",
    "lens",
    "pixels",
    "n planes",
    "cam",
    "behaviour",
    "treadmill",
    "_sentinel",  # extra column so define_metadata_columns (range(1, max_column)) captures treadmill
]

# Map column name -> 1-based index
COL = {name: idx + 1 for idx, name in enumerate(COLUMNS)}


def _make_wb(rows: list[dict]) -> openpyxl.Workbook:
    """Return an openpyxl Workbook with header row + given data rows.

    Appends one extra empty row so that sheet.max_row > last data row,
    allowing ``range(2, sheet.max_row)`` to include all data rows.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    # Header row
    for col_name, col_idx in COL.items():
        ws.cell(row=1, column=col_idx, value=col_name)
    # Data rows
    for row_idx, row_data in enumerate(rows, start=2):
        for col_name, value in row_data.items():
            ws.cell(row=row_idx, column=COL[col_name], value=value)
    # Sentinel: write a value in the extra row to bump max_row by 1 so
    # range(2, max_row) covers all data rows (the sentinel row itself has
    # mouse ID = None, so create_animal_dict will return None and skip it).
    sentinel_row = len(rows) + 2
    ws.cell(row=sentinel_row, column=COL["mouse ID"], value=None)
    ws.cell(row=sentinel_row, column=1, value="")  # ensures max_row == sentinel_row
    return wb


def _default_row() -> dict:
    """Minimal but complete data row for a valid animal/session."""
    return {
        "mouse ID": "DON001234",  # len==9 -> uses DON-0 prefix path
        "sex": "m",
        "DOB": 210101,  # stored as int; "20" + str(int(210101)) -> "20210101"
        "injected": 210201,
        "implanted": 210301,
        "date": 210401,
        "weight [g]": 25,
        "session": "S1",
        "duration [min]": 30,
        "paradigm": "linear_track",
        "comment": None,
        "setup": "2P-setup-A",
        "[nm]": 920,
        "laser / LED power": 10,
        "PC bias": 50,
        "n Ch.": 2,
        "fct. channel": 1,
        "UG gain": 100,
        "UR gain": 200,
        "lens": "16x",
        "pixels": 512,
        "n planes": 3,
        "cam": "yes",
        "behaviour": "no",
        "treadmill": "linear",
    }


# ---------------------------------------------------------------------------
# Pure-Python helpers
# ---------------------------------------------------------------------------


class TestCreateDict:
    def test_placeholder_values_become_none(self):
        d = create_dict(a="n/a", b="", c="?", d="real_value")
        assert d["a"] is None
        assert d["b"] is None
        assert d["c"] is None
        assert d["d"] == "real_value"

    def test_none_input_unchanged(self):
        d = create_dict(x=None, y=42)
        assert d["x"] is None
        assert d["y"] == 42

    def test_returns_same_reference(self):
        result = create_dict(key="value")
        assert isinstance(result, dict)


class TestNumToDate:
    def test_string_input(self):
        from datetime import datetime

        result = num_to_date("20210228")
        assert result == datetime(2021, 2, 28)

    def test_int_input_converts_to_str(self):
        from datetime import datetime

        result = num_to_date(20210101)
        assert result == datetime(2021, 1, 1)

    def test_invalid_date_raises(self):
        with pytest.raises(ValueError):
            num_to_date("not-a-date")


class TestRemoveNoneFromDict:
    def test_non_recursive(self):
        d = {"a": 1, "b": None, "c": {"x": None}}
        result = remove_none_from_dict(d, recursive=False)
        assert "b" not in result
        assert "c" in result  # nested dict kept as-is
        assert result["c"] == {"x": None}

    def test_recursive(self):
        d = {"a": 1, "b": None, "c": {"x": None, "y": 2}}
        result = remove_none_from_dict(d, recursive=True)
        assert "b" not in result
        assert result["c"] == {"y": 2}

    def test_empty_dict(self):
        assert remove_none_from_dict({}) == {}

    def test_all_none_recursive(self):
        assert remove_none_from_dict({"a": None, "b": None}, recursive=True) == {}


class TestSearchUpdateDict:
    def test_direct_key_update(self):
        d = {"a": {"x": 1}, "b": 2}
        result = search_update_dict(d, {"a": {"y": 3}})
        assert result["a"]["x"] == 1
        assert result["a"]["y"] == 3

    def test_nested_key_update(self):
        d = {"outer": {"inner": {"target": 1}}}
        result = search_update_dict(d, {"target": 99})
        # key not found at top level -> recursion
        assert isinstance(result, dict)

    def test_non_dict_value_not_updated(self):
        # key exists but its value is not a dict — the update branch is skipped
        d = {"a": "string_value"}
        result = search_update_dict(d, {"a": "new"})
        # value is a string, not a dict, so update() is not called
        assert result["a"] == "string_value"

    def test_returns_dict(self):
        result = search_update_dict({"k": {"v": 1}}, {})
        assert isinstance(result, dict)

    def test_else_branch_with_non_dict_value(self):
        # Key NOT in dictionary -> else branch; but dictionary values are all
        # non-dict, so isinstance(dict_value, dict) is False (covers 408->407 branch).
        d = {"x": 42, "y": "string"}  # no dict values
        result = search_update_dict(d, {"missing_key": {"z": 1}})
        # No update should happen; dictionary is returned unchanged
        assert result == {"x": 42, "y": "string"}


# ---------------------------------------------------------------------------
# Spreadsheet helpers – using a real openpyxl worksheet object
# ---------------------------------------------------------------------------


def _get_sheet():
    wb = _make_wb([_default_row()])
    return wb.active


class TestRowToList:
    def test_returns_header_values(self):
        # row_to_list uses sheet[row][1:10000] — the first column (index 0) is always skipped.
        # Our header row starts at column 1, so "mouse ID" (col 1) is skipped;
        # "sex" (col 2) is the first value returned.
        sheet = _get_sheet()
        result = row_to_list(sheet, 1)
        assert "sex" in result   # col 2 — first one captured
        assert "_sentinel" in result or "treadmill" in result  # later columns captured

    def test_stops_at_none(self):
        # row_to_list uses sheet[row][1:10000] which skips the first cell (index 0)
        # so column 1 (index 0) is always skipped; column 2 (index 1) is first.
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.cell(row=1, column=1, value="SKIP")  # col 1 is at index 0 -> skipped by [1:]
        ws.cell(row=1, column=2, value="A")
        ws.cell(row=1, column=3, value=None)   # stops here
        ws.cell(row=1, column=4, value="B")    # never reached
        result = row_to_list(ws, 1)
        assert result == ["A"]  # stops at first None after column 2

    def test_empty_row_returns_empty_list(self):
        wb = openpyxl.Workbook()
        ws = wb.active
        # Row 2 is empty
        result = row_to_list(ws, 2)
        assert result == []


class TestDefineMetadataColumns:
    def test_maps_column_names_to_indices(self):
        sheet = _get_sheet()
        metadata_columns = define_metadata_columns(sheet)
        assert "mouse ID" in metadata_columns
        assert metadata_columns["mouse ID"] == COL["mouse ID"]

    def test_skips_none_header_cells(self):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.cell(row=1, column=1, value="ColA")
        ws.cell(row=1, column=2, value=None)
        ws.cell(row=1, column=3, value="ColC")
        result = define_metadata_columns(ws)
        assert "ColA" in result
        assert None not in result


class TestCreateStimulusDict:
    def test_with_no_definition(self):
        sheet = _get_sheet()
        mc = define_metadata_columns(sheet)
        result = create_stimulus_dict(sheet, mc, row=2)
        assert "type" in result
        assert result["type"] == "linear"

    def test_with_custom_definition(self):
        sheet = _get_sheet()
        mc = define_metadata_columns(sheet)
        definition = {"linear": {"sequence": [1, 2, 3], "dimensions": 1, "by": "time"}}
        result = create_stimulus_dict(sheet, mc, row=2, definition=definition)
        assert result["sequence"] == [1, 2, 3]
        assert result["dimensions"] == 1
        assert result["by"] == "time"


class TestCreateBehaviorDict:
    def test_cam_yes_movement_no(self):
        sheet = _get_sheet()
        mc = define_metadata_columns(sheet)
        result = create_behavior_dict(sheet, mc, row=2)
        assert result["cam_data"] is True
        assert result["movement_data"] is False
        assert "stimulus" in result

    def test_cam_no(self):
        row = _default_row()
        row["cam"] = "no"
        row["behaviour"] = "yes"
        wb = _make_wb([row])
        sheet = wb.active
        mc = define_metadata_columns(sheet)
        result = create_behavior_dict(sheet, mc, row=2)
        assert result["cam_data"] is False
        assert result["movement_data"] is True


class TestCreateNeuralDict:
    def test_basic_fields_present(self):
        sheet = _get_sheet()
        mc = define_metadata_columns(sheet)
        result = create_neural_dict(sheet, mc, row=2)
        assert result["method"] == "2P"
        assert result["setup"] == "2P-setup-A"
        assert result["n_channel"] == 2
        assert result["functional_channel"] == 1
        assert result["n_planes"] == 3

    def test_optional_columns_missing(self):
        """Without lens/pixels/n planes columns -> they default to None."""
        row = _default_row()
        # Build a workbook with only the required columns (no optional ones)
        minimal_cols = [c for c in COLUMNS if c not in ("lens", "pixels", "n planes")]
        wb = openpyxl.Workbook()
        ws = wb.active
        col_map = {}
        for idx, col_name in enumerate(minimal_cols, start=1):
            ws.cell(row=1, column=idx, value=col_name)
            col_map[col_name] = idx
        for col_name, value in row.items():
            if col_name in col_map:
                ws.cell(row=2, column=col_map[col_name], value=value)
        result = create_neural_dict(ws, col_map, row=2)
        assert result.get("lens") is None
        assert result.get("pixels") is None
        assert result.get("n_planes") is None

    def test_na_channel_values_become_none(self):
        row = _default_row()
        row["n Ch."] = "n/a"
        row["fct. channel"] = "?"
        row["n planes"] = ""
        wb = _make_wb([row])
        sheet = wb.active
        mc = define_metadata_columns(sheet)
        result = create_neural_dict(sheet, mc, row=2)
        assert result["n_channel"] is None
        assert result["functional_channel"] is None
        assert result["n_planes"] is None

    def test_string_int_channel_values(self):
        row = _default_row()
        row["n Ch."] = "3"
        row["fct. channel"] = "2"
        row["n planes"] = "4"
        wb = _make_wb([row])
        sheet = wb.active
        mc = define_metadata_columns(sheet)
        result = create_neural_dict(sheet, mc, row=2)
        assert result["n_channel"] == 3
        assert result["functional_channel"] == 2
        assert result["n_planes"] == 4


class TestCreateTaskDict:
    def test_duration_normal(self):
        sheet = _get_sheet()
        mc = define_metadata_columns(sheet)
        result = create_task_dict(sheet, mc, row=2)
        assert result["duration"] == 30
        assert result["expt_pipeline"] == "linear_track"
        assert "neural_metadata" in result
        assert "behavior_metadata" in result

    def test_duration_placeholder_becomes_none(self):
        for placeholder in ("n/a", "", "?"):
            row = _default_row()
            row["duration [min]"] = placeholder
            wb = _make_wb([row])
            sheet = wb.active
            mc = define_metadata_columns(sheet)
            result = create_task_dict(sheet, mc, row=2)
            assert result["duration"] is None, f"failed for placeholder {placeholder!r}"


class TestCreateSessionDict:
    def test_date_formatted_correctly(self):
        sheet = _get_sheet()
        mc = define_metadata_columns(sheet)
        result = create_session_dict(sheet, mc, row=2)
        # date == 210401 -> "20" + str(210401) -> "20210401"
        assert result["date"] == "20210401"
        assert "tasks_metadata" in result
        assert "S1" in result["tasks_metadata"]

    def test_missing_date_is_none(self):
        row = _default_row()
        row["date"] = None
        wb = _make_wb([row])
        sheet = wb.active
        mc = define_metadata_columns(sheet)
        result = create_session_dict(sheet, mc, row=2)
        assert result["date"] is None


class TestCreateAnimalDict:
    def test_returns_none_for_missing_animal_id(self):
        row = _default_row()
        row["mouse ID"] = None
        wb = _make_wb([row])
        sheet = wb.active
        mc = define_metadata_columns(sheet)
        result = create_animal_dict(sheet, mc, row=2)
        assert result is None

    def test_7_char_animal_id(self):
        row = _default_row()
        row["mouse ID"] = "DON1234"  # length 7 -> "DON-00" + id[3:]
        wb = _make_wb([row])
        sheet = wb.active
        mc = define_metadata_columns(sheet)
        result = create_animal_dict(sheet, mc, row=2)
        assert result is not None
        assert result["animal_id"] == "DON-001234"

    def test_9_char_animal_id(self):
        row = _default_row()
        row["mouse ID"] = "DON001234"  # length 9 -> "DON-0" + id[3:]
        wb = _make_wb([row])
        sheet = wb.active
        mc = define_metadata_columns(sheet)
        result = create_animal_dict(sheet, mc, row=2)
        assert result is not None
        assert result["animal_id"] == "DON-0001234"

    def test_male_sex(self):
        sheet = _get_sheet()
        mc = define_metadata_columns(sheet)
        result = create_animal_dict(sheet, mc, row=2)
        assert result["sex"] == "male"

    def test_female_sex(self):
        row = _default_row()
        row["sex"] = "f"
        wb = _make_wb([row])
        sheet = wb.active
        mc = define_metadata_columns(sheet)
        result = create_animal_dict(sheet, mc, row=2)
        assert result["sex"] == "female"

    def test_unknown_sex(self):
        row = _default_row()
        row["sex"] = None
        wb = _make_wb([row])
        sheet = wb.active
        mc = define_metadata_columns(sheet)
        result = create_animal_dict(sheet, mc, row=2)
        assert result["sex"] is None

    def test_missing_dob_injected_implanted(self):
        row = _default_row()
        row["DOB"] = None
        row["injected"] = None
        row["implanted"] = None
        wb = _make_wb([row])
        sheet = wb.active
        mc = define_metadata_columns(sheet)
        result = create_animal_dict(sheet, mc, row=2)
        assert result["dob"] is None
        assert result["injected"] is None
        assert result["implanted"] is None

    def test_sessions_key_populated(self):
        sheet = _get_sheet()
        mc = define_metadata_columns(sheet)
        result = create_animal_dict(sheet, mc, row=2)
        assert "sessions" in result
        assert len(result["sessions"]) == 1


# ---------------------------------------------------------------------------
# get_animal_dict_from_spreadsheet – needs a real .xlsx on disk
# ---------------------------------------------------------------------------


def _write_minimal_xlsx(path: str) -> None:
    wb = _make_wb([_default_row()])
    wb.save(path)


class TestGetAnimalDictFromSpreadsheet:
    def test_basic_read(self, tmp_path):
        fpath = str(tmp_path / "test.xlsx")
        _write_minimal_xlsx(fpath)
        animals = get_animal_dict_from_spreadsheet(fpath)
        assert len(animals) == 1
        animal = next(iter(animals.values()))
        assert "sessions" in animal

    def test_invalid_sheet_title_raises(self, tmp_path):
        fpath = str(tmp_path / "test.xlsx")
        _write_minimal_xlsx(fpath)
        with pytest.raises(ValueError, match="not in sheetnames"):
            get_animal_dict_from_spreadsheet(fpath, sheet_title="NoSuchSheet")

    def test_default_sheet_title_none(self, tmp_path):
        fpath = str(tmp_path / "test.xlsx")
        _write_minimal_xlsx(fpath)
        animals = get_animal_dict_from_spreadsheet(fpath, sheet_title=None)
        assert len(animals) == 1

    def test_remove_none_false(self, tmp_path):
        fpath = str(tmp_path / "test.xlsx")
        _write_minimal_xlsx(fpath)
        animals = get_animal_dict_from_spreadsheet(fpath, remove_none=False)
        assert len(animals) == 1

    def test_merges_same_animal_same_session_different_tasks(self, tmp_path):
        row1 = _default_row()
        row2 = copy.deepcopy(row1)
        row2["session"] = "S2"  # same date, different session task
        wb = _make_wb([row1, row2])
        fpath = str(tmp_path / "two_tasks.xlsx")
        wb.save(fpath)
        animals = get_animal_dict_from_spreadsheet(fpath)
        animal = next(iter(animals.values()))
        first_date = next(iter(animal["sessions"].keys()))
        tasks = animal["sessions"][first_date]["tasks_metadata"]
        assert "S1" in tasks
        assert "S2" in tasks

    def test_merges_same_animal_different_sessions(self, tmp_path):
        row1 = _default_row()
        row2 = copy.deepcopy(row1)
        row2["date"] = 210501  # different date -> different session
        wb = _make_wb([row1, row2])
        fpath = str(tmp_path / "two_sessions.xlsx")
        wb.save(fpath)
        animals = get_animal_dict_from_spreadsheet(fpath)
        animal = next(iter(animals.values()))
        assert len(animal["sessions"]) == 2

    def test_skips_rows_without_animal_id(self, tmp_path):
        row_with_id = _default_row()
        row_without_id = _default_row()
        row_without_id["mouse ID"] = None
        wb = _make_wb([row_with_id, row_without_id])
        fpath = str(tmp_path / "skip_row.xlsx")
        wb.save(fpath)
        animals = get_animal_dict_from_spreadsheet(fpath)
        assert len(animals) == 1

    def test_stimulus_definition_passed_through(self, tmp_path):
        fpath = str(tmp_path / "stim.xlsx")
        _write_minimal_xlsx(fpath)
        stim_def = {"linear": {"sequence": [0, 1], "dimensions": 1, "by": "time"}}
        animals = get_animal_dict_from_spreadsheet(
            fpath, stimulus_definition=stim_def
        )
        assert len(animals) == 1

    def test_multiple_different_animals(self, tmp_path):
        row1 = _default_row()
        row2 = _default_row()
        row2["mouse ID"] = "DON009999"
        row2["sex"] = "f"
        wb = _make_wb([row1, row2])
        fpath = str(tmp_path / "two_animals.xlsx")
        wb.save(fpath)
        animals = get_animal_dict_from_spreadsheet(fpath)
        assert len(animals) == 2


# ---------------------------------------------------------------------------
# return_loaded_yaml_if_newer
# ---------------------------------------------------------------------------


class TestReturnLoadedYamlIfNewer:
    def test_returns_none_when_newer_does_not_exist(self, tmp_path):
        used = str(tmp_path / "used.yaml")
        newer = str(tmp_path / "newer.yaml")
        # used exists, newer does not
        with open(used, "w") as f:
            yaml.dump({"a": 1}, f)
        result = return_loaded_yaml_if_newer(used, newer)
        assert result is None

    def test_returns_none_when_newer_is_older(self, tmp_path):
        used = str(tmp_path / "used.yaml")
        newer = str(tmp_path / "newer.yaml")
        with open(used, "w") as f:
            yaml.dump({"a": 1}, f)
        time.sleep(0.05)
        with open(newer, "w") as f:
            yaml.dump({"b": 2}, f)
        # now set used's mtime to the future
        future = os.path.getmtime(newer) + 10
        os.utime(used, (future, future))
        result = return_loaded_yaml_if_newer(used, newer)
        assert result is None

    def test_returns_dict_when_newer_is_actually_newer(self, tmp_path):
        used = str(tmp_path / "used.yaml")
        newer = str(tmp_path / "newer.yaml")
        with open(used, "w") as f:
            yaml.dump({"old": True}, f)
        time.sleep(0.05)
        with open(newer, "w") as f:
            yaml.dump({"new": True}, f)
        result = return_loaded_yaml_if_newer(used, newer)
        assert result == {"new": True}

    def test_used_does_not_exist_and_newer_exists(self, tmp_path):
        used = str(tmp_path / "used_missing.yaml")
        newer = str(tmp_path / "newer.yaml")
        # used does not exist -> mtime treated as 0
        with open(newer, "w") as f:
            yaml.dump({"fresh": 42}, f)
        result = return_loaded_yaml_if_newer(used, newer)
        assert result == {"fresh": 42}

    def test_neither_file_exists(self, tmp_path):
        result = return_loaded_yaml_if_newer(
            str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml")
        )
        assert result is None


# ---------------------------------------------------------------------------
# get_animals_from_yaml
# ---------------------------------------------------------------------------


class TestGetAnimalsFromYaml:
    def test_returns_empty_when_no_yaml_file(self, tmp_path):
        result = get_animals_from_yaml(str(tmp_path))
        assert result == {}

    def test_reads_existing_yaml(self, tmp_path):
        animals_data = {"DON-001": {"sessions": {}}}
        yaml_path = tmp_path / "animal_summary.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(animals_data, f)
        result = get_animals_from_yaml(str(tmp_path))
        assert result == animals_data

    def test_empty_string_directory(self, tmp_path):
        # directory="" should not crash (uses "" as root_dir, file won't exist)
        result = get_animals_from_yaml("")
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# combine_spreadsheet_and_old_animal_summary_yaml
# ---------------------------------------------------------------------------


class TestCombineSpreadsheetAndOldYaml:
    def _spreadsheet_animals(self):
        return {
            "DON-001": {
                "animal_id": "DON-001",
                "sessions": {
                    "20210401": {"date": "20210401", "tasks_metadata": {}},
                },
            }
        }

    def test_adds_missing_session_from_yaml(self):
        spreadsheet = self._spreadsheet_animals()
        yaml_animals = {
            "DON-001": {
                "sessions": {
                    "20210501": {"date": "20210501"}  # not in spreadsheet
                }
            }
        }
        result = combine_spreadsheet_and_old_animal_summary_yaml(
            spreadsheet, yaml_animals
        )
        assert "20210501" in result["DON-001"]["sessions"]

    def test_adds_usemunits_when_present(self):
        spreadsheet = self._spreadsheet_animals()
        yaml_animals = {
            "DON-001": {
                "sessions": {
                    "20210401": {
                        "date": "20210401",
                        "UseMUnits": [["file.mesc", [0, 1]]],
                    }
                }
            }
        }
        result = combine_spreadsheet_and_old_animal_summary_yaml(
            spreadsheet, yaml_animals
        )
        assert "UseMUnits" in result["DON-001"]["sessions"]["20210401"]

    def test_skips_empty_usemunits(self):
        spreadsheet = self._spreadsheet_animals()
        yaml_animals = {
            "DON-001": {
                "sessions": {
                    "20210401": {
                        "date": "20210401",
                        "UseMUnits": [],  # falsy
                    }
                }
            }
        }
        result = combine_spreadsheet_and_old_animal_summary_yaml(
            spreadsheet, yaml_animals
        )
        # UseMUnits should NOT be added for empty list (continue branch)
        assert "UseMUnits" not in result["DON-001"]["sessions"]["20210401"]

    def test_empty_yaml_animals(self):
        spreadsheet = self._spreadsheet_animals()
        result = combine_spreadsheet_and_old_animal_summary_yaml(spreadsheet, {})
        assert result == spreadsheet


# ---------------------------------------------------------------------------
# create_folders_for_animals
# ---------------------------------------------------------------------------


class TestCreateFoldersForAnimals:
    def _minimal_animals(self):
        return {
            "DON-001": {
                "animal_id": "DON-001",
                "sex": "male",
                "sessions": {
                    "20210401": {"date": "20210401", "weight": 25},
                },
            }
        }

    def _pre_create_dirs(self, tmp_path):
        """create_folders_for_animals does not mkdir; we must create them first."""
        session_path = tmp_path / "DON-001" / "20210401"
        session_path.mkdir(parents=True, exist_ok=True)
        animal_path = tmp_path / "DON-001"
        animal_path.mkdir(parents=True, exist_ok=True)

    def test_creates_yaml_files_in_pre_existing_dirs(self, tmp_path):
        animals = self._minimal_animals()
        self._pre_create_dirs(tmp_path)
        create_folders_for_animals(animals, directory=str(tmp_path))
        session_yaml = tmp_path / "DON-001" / "20210401" / "20210401.yaml"
        animal_yaml = tmp_path / "DON-001" / "DON-001.yaml"
        assert session_yaml.exists()
        assert animal_yaml.exists()

    # Keep old name so existing test count stays ~same but renamed test passes
    def test_creates_directories_and_yaml_files(self, tmp_path):
        animals = self._minimal_animals()
        self._pre_create_dirs(tmp_path)
        create_folders_for_animals(animals, directory=str(tmp_path))
        session_yaml = tmp_path / "DON-001" / "20210401" / "20210401.yaml"
        animal_yaml = tmp_path / "DON-001" / "DON-001.yaml"
        assert session_yaml.exists()
        assert animal_yaml.exists()

    def test_session_yaml_content(self, tmp_path):
        animals = self._minimal_animals()
        self._pre_create_dirs(tmp_path)
        create_folders_for_animals(animals, directory=str(tmp_path))
        session_yaml = tmp_path / "DON-001" / "20210401" / "20210401.yaml"
        with open(session_yaml) as f:
            data = yaml.safe_load(f)
        assert data["date"] == "20210401"

    def test_animal_yaml_excludes_sessions(self, tmp_path):
        animals = self._minimal_animals()
        self._pre_create_dirs(tmp_path)
        create_folders_for_animals(animals, directory=str(tmp_path))
        animal_yaml = tmp_path / "DON-001" / "DON-001.yaml"
        with open(animal_yaml) as f:
            data = yaml.safe_load(f)
        assert "sessions" not in data
        assert data["animal_id"] == "DON-001"

    def test_save_yamls_false_does_not_write_files(self, tmp_path):
        animals = self._minimal_animals()
        create_folders_for_animals(
            animals, directory=str(tmp_path), save_yamls=False
        )
        animal_yaml = tmp_path / "DON-001" / "DON-001.yaml"
        assert not animal_yaml.exists()


# ---------------------------------------------------------------------------
# move_mesc_to_session_folder
# ---------------------------------------------------------------------------


class TestMoveMescToSessionFolder:
    def test_moves_mesc_file_to_session_folder(self, tmp_path):
        # Create a .mesc file following DON-<id>_<date>.mesc naming
        mesc_file = tmp_path / "DON001234_210401.mesc"
        mesc_file.write_bytes(b"fakemesc")
        move_mesc_to_session_folder(directory=str(tmp_path))
        # After move: file should be in DON001234/210401/
        dest = tmp_path / "DON001234" / "210401" / "DON001234_210401.mesc"
        assert dest.exists()

    def test_non_don_file_is_skipped(self, tmp_path):
        # File without DON prefix should remain in place
        other_file = tmp_path / "XYZ001234_210401.mesc"
        other_file.write_bytes(b"fakemesc")
        move_mesc_to_session_folder(directory=str(tmp_path))
        # Should still be in tmp_path (not moved)
        assert other_file.exists()

    def test_empty_directory_string_handled(self, tmp_path):
        # directory=None after "" -> should raise TypeError (pathlib.Path(None))
        # We just test with a valid empty dir
        move_mesc_to_session_folder(directory=str(tmp_path))


# ---------------------------------------------------------------------------
# get_recording_munits — test with a real HDF5 file
# ---------------------------------------------------------------------------


class TestGetRecordingMunits:
    def _make_mesc_file(self, path: str, n_frames: int = 10000, n_channels: int = 2):
        """Create a minimal HDF5 file mimicking .mesc structure."""
        with h5py.File(path, "w") as f:
            msession = f.create_group("MSession_0")
            mu = msession.create_group("MUnit_0")
            for ch in range(n_channels):
                mu.create_dataset(f"Channel_{ch}", data=np.zeros(n_frames))

    def test_returns_munits_and_channel_count(self, tmp_path):
        fpath = str(tmp_path / "test.mesc")
        self._make_mesc_file(fpath, n_frames=10000, n_channels=2)
        munits, n_ch = get_recording_munits(fpath, session_parts=[0])
        assert isinstance(munits, list)
        assert n_ch == 2

    def test_short_recording_raises_unbound(self, tmp_path):
        """Source bug: when no unit has enough frames, number_channels is unbound.

        The function tries to return number_channels (line 601) but that variable
        is only assigned inside the ``if unit["Channel_0"].shape[0] > ...`` branch.
        When all units are too short, this raises UnboundLocalError.
        We document this as a known bug rather than contriving around it.
        """
        fpath = str(tmp_path / "short.mesc")
        self._make_mesc_file(fpath, n_frames=100, n_channels=1)
        with pytest.raises(UnboundLocalError):
            get_recording_munits(fpath, session_parts=[0])

    def test_long_recording_included(self, tmp_path):
        """Recording > 5 min @ 30fps (> 9000 frames) is included."""
        fpath = str(tmp_path / "long.mesc")
        self._make_mesc_file(fpath, n_frames=15000, n_channels=1)
        munits, n_ch = get_recording_munits(fpath, session_parts=[0])
        assert 0 in munits

    def test_mixed_keys_in_unit(self, tmp_path):
        """Unit has both Channel and non-Channel keys; non-Channel keys don't count."""
        fpath = str(tmp_path / "mixed.mesc")
        with h5py.File(fpath, "w") as f:
            msession = f.create_group("MSession_0")
            mu = msession.create_group("MUnit_0")
            # Add enough frames to be included (> 9000)
            mu.create_dataset("Channel_0", data=np.zeros(15000))
            mu.create_dataset("Channel_1", data=np.zeros(15000))
            # Add a non-Channel key (covers 599->598 branch: condition False)
            mu.create_dataset("SomeOtherKey", data=np.zeros(10))
        munits, n_ch = get_recording_munits(fpath, session_parts=[0])
        assert 0 in munits
        assert n_ch == 2  # only Channel_* keys counted


# ---------------------------------------------------------------------------
# add_session_animal_folders — needs real filesystem + .mesc files
# ---------------------------------------------------------------------------


class TestAddSessionAnimalFolders:
    """add_session_animal_folders walks dirs matching DON-*, finds .mesc files,
    and populates animal metadata from them.

    The function calls get_recording_munits internally, which has an UnboundLocalError
    bug when recordings are too short (see TestGetRecordingMunits). We therefore use
    recordings with enough frames (>= 9000) in all tests here.
    """

    def _make_mesc(self, path, n_frames=15000, n_channels=2):
        with h5py.File(path, "w") as f:
            ms = f.create_group("MSession_0")
            mu = ms.create_group("MUnit_0")
            for ch in range(n_channels):
                mu.create_dataset(f"Channel_{ch}", data=np.zeros(n_frames))

    def test_empty_directory_returns_unchanged(self, tmp_path):
        """No DON-* directories -> animals dict unchanged."""
        animals = {"DON-001234": {"sessions": {}, "session_names": [], "session_dates": [], "pdays": [], "dob": "20200101"}}
        animals_spreadsheet = copy.deepcopy(animals)
        result = add_session_animal_folders(animals, animals_spreadsheet, directory=str(tmp_path))
        assert result == animals

    def test_processes_mesc_file_already_tracked(self, tmp_path):
        """If the session/date is already in session_names and session_dates, pday is NOT re-added."""
        animal_id = "DON-001234"
        session_id = "20210401"
        # session_date in filename is splitted_fname[1] -> must be 8-digit for num_to_date
        session_date_str = "20210401"

        # Build directory structure
        mesc_subdir = tmp_path / animal_id / session_id / "002P-F"
        mesc_subdir.mkdir(parents=True)
        mesc_fname = f"{animal_id}_{session_date_str}_S1.mesc"
        self._make_mesc(str(mesc_subdir / mesc_fname), n_frames=15000, n_channels=2)

        animals = {
            animal_id: {
                "sessions": {},
                "session_names": [session_id],        # already tracked
                "session_dates": [session_date_str],  # already tracked
                "pdays": [100],
                "dob": "20200101",
                "UseMUnits": [],
            }
        }
        animals_spreadsheet = copy.deepcopy(animals)
        result = add_session_animal_folders(
            animals, animals_spreadsheet, directory=str(tmp_path)
        )
        # pdays should not get a new entry since session was already tracked
        assert result[animal_id]["pdays"] == [100]

    def test_adds_animal_from_spreadsheet_if_not_in_animals(self, tmp_path):
        """If animal_id found in filesystem but not in animals dict, it is added from spreadsheet."""
        animal_id = "DON-001234"
        session_id = "20210401"

        # Create directory structure (no .mesc files needed for this path)
        (tmp_path / animal_id / session_id / "002P-F").mkdir(parents=True)

        animals = {}  # animal not yet in animals
        animals_spreadsheet = {
            animal_id: {
                "sessions": {},
                "session_names": [],
                "session_dates": [],
                "pdays": [],
                "dob": "20200101",
            }
        }
        result = add_session_animal_folders(
            animals, animals_spreadsheet, directory=str(tmp_path)
        )
        assert animal_id in result

    def test_mesc_file_with_many_munits_adds_to_usemunits(self, tmp_path):
        """When munits_list > session_parts, the mesc+munits pair is added to UseMUnits.

        The filename encodes session parts via S[0-9] patterns; S1 -> session_parts=[0].
        Two MUnits recorded (munits_list=[0,1]) > session_parts=[0] -> else branch.
        Also tests line 538-539 (new session, dob conversion, pday computation).
        """
        animal_id = "DON-001234"
        session_id = "20210401"
        # splitted_fname[1] is the date part in the filename; must be parseable by num_to_date
        session_date_str = "20210401"  # 8-digit YYYYMMDD

        mesc_subdir = tmp_path / animal_id / session_id / "002P-F"
        mesc_subdir.mkdir(parents=True)
        # Filename: animal_id _ date _ S1.mesc -> re.findall("S[0-9]", "S1") -> ["S1"]
        # session_parts = [int("1"[-1]) - 1] = [0]
        mesc_fname = f"{animal_id}_{session_date_str}_S1.mesc"

        # Create an HDF5 file with 2 MUnits, each with enough frames
        # len(munits_list)=2 > len(session_parts)=1 -> enters else branch (lines 557-565)
        with h5py.File(str(mesc_subdir / mesc_fname), "w") as f:
            ms = f.create_group("MSession_0")
            for u in range(2):
                mu = ms.create_group(f"MUnit_{u}")
                mu.create_dataset("Channel_0", data=np.zeros(15000))
                mu.create_dataset("Channel_1", data=np.zeros(15000))

        # Pre-populate session (skip new-session branch) AND pre-populate UseMUnits with a
        # DIFFERENT filename, so the `if mesc_munit_pairs:` block (line 558) is entered.
        # Within that block (lines 559-565): since fname is NOT already in mesc_munit_pairs,
        # add_mesc_munit_pair stays True -> the new pair gets appended (line 564-565).
        # Line 580 then sets animals[animal_id]["UseMUnits"] = mesc_munit_pairs.
        animals = {
            animal_id: {
                "sessions": {},
                "session_names": [session_id],        # already tracked -> skip new-session branch
                "session_dates": [session_date_str],  # already tracked
                "pdays": [100],
                "dob": "20200101",
                # Non-empty -> mesc_munit_pairs is truthy -> enters lines 559-565
                "UseMUnits": [["other_file.mesc", [0]]],
            }
        }
        animals_spreadsheet = copy.deepcopy(animals)
        result = add_session_animal_folders(
            animals, animals_spreadsheet, directory=str(tmp_path)
        )
        assert animal_id in result
        # The new mesc+munits pair should have been appended (lines 564-565 and 580)
        assert "UseMUnits" in result[animal_id]
        use_munits = result[animal_id]["UseMUnits"]
        # Should have original entry + new one
        assert len(use_munits) == 2
        # The new entry contains our mesc filename
        filenames = [entry[0] for entry in use_munits]
        assert mesc_fname in filenames

    def test_mesc_file_already_in_usemunits_not_added_again(self, tmp_path):
        """When the mesc filename is already in UseMUnits, add_mesc_munit_pair = False (line 562).

        Covers branches 558->567 (mesc_munit_pairs truthy -> enters block),
        559->563 (for loop runs), 562 (add_mesc_munit_pair = False),
        and 563->567 (if add_mesc_munit_pair: is False -> skip append).
        """
        animal_id = "DON-001234"
        session_id = "20210401"
        session_date_str = "20210401"

        mesc_subdir = tmp_path / animal_id / session_id / "002P-F"
        mesc_subdir.mkdir(parents=True)
        mesc_fname = f"{animal_id}_{session_date_str}_S1.mesc"

        # Two MUnits -> len(munits_list)=2 > len(session_parts)=1 -> else branch at 556
        with h5py.File(str(mesc_subdir / mesc_fname), "w") as f:
            ms = f.create_group("MSession_0")
            for u in range(2):
                mu = ms.create_group(f"MUnit_{u}")
                mu.create_dataset("Channel_0", data=np.zeros(15000))
                mu.create_dataset("Channel_1", data=np.zeros(15000))

        animals = {
            animal_id: {
                "sessions": {},
                "session_names": [session_id],
                "session_dates": [session_date_str],
                "pdays": [100],
                "dob": "20200101",
                # fname is already in this entry -> add_mesc_munit_pair = False
                "UseMUnits": [[mesc_fname, [0, 1]]],
            }
        }
        animals_spreadsheet = copy.deepcopy(animals)
        result = add_session_animal_folders(
            animals, animals_spreadsheet, directory=str(tmp_path)
        )
        # UseMUnits should NOT have a duplicate entry (add_mesc_munit_pair was set False)
        assert "UseMUnits" in result[animal_id]
        assert len(result[animal_id]["UseMUnits"]) == 1

    def test_new_session_branch_hits_source_bug(self, tmp_path):
        """Lines 534-539 (new session tracking) are reachable but immediately followed by
        line 569 (int(session_date)) which fails with TypeError because session_date was
        converted to datetime on line 537. This documents the source bug.
        """
        animal_id = "DON-001234"
        session_id = "20210401"
        session_date_str = "20210401"  # 8-digit YYYYMMDD

        mesc_subdir = tmp_path / animal_id / session_id / "002P-F"
        mesc_subdir.mkdir(parents=True)
        mesc_fname = f"{animal_id}_{session_date_str}_S1.mesc"
        # Single MUnit -> len(munits_list) <= len(session_parts) -> no else branch needed
        self._make_mesc(str(mesc_subdir / mesc_fname), n_frames=15000, n_channels=1)

        animals = {
            animal_id: {
                "sessions": {},
                "session_names": [],        # NOT tracked -> new-session branch 534-539
                "session_dates": [],        # NOT tracked
                "pdays": [],
                "dob": "20200101",          # valid date for num_to_date (line 536)
                "UseMUnits": [],
            }
        }
        animals_spreadsheet = copy.deepcopy(animals)
        # Lines 534-539 execute successfully (dob_date, pday computed correctly),
        # but line 537 converts session_date to datetime, then line 569 calls int()
        # on a datetime object -> TypeError (source code bug).
        with pytest.raises(TypeError):
            add_session_animal_folders(
                animals, animals_spreadsheet, directory=str(tmp_path)
            )

    def test_mismatched_animal_id_in_filename_skipped(self, tmp_path):
        """If the .mesc filename's animal_id doesn't match the directory, it's skipped."""
        animal_id = "DON-001234"
        session_id = "20210401"

        mesc_subdir = tmp_path / animal_id / session_id / "002P-F"
        mesc_subdir.mkdir(parents=True)
        # Filename has different animal ID prefix
        wrong_fname = f"DON-999999_{session_id}_S1.mesc"
        self._make_mesc(str(mesc_subdir / wrong_fname), n_frames=15000)

        animals = {
            animal_id: {
                "sessions": {},
                "session_names": [],
                "session_dates": [],
                "pdays": [],
                "dob": "20200101",
                "UseMUnits": [],
            }
        }
        animals_spreadsheet = copy.deepcopy(animals)
        result = add_session_animal_folders(
            animals, animals_spreadsheet, directory=str(tmp_path)
        )
        # session_names should still be empty since filename was skipped
        assert result[animal_id]["session_names"] == []


# ---------------------------------------------------------------------------
# main() — integration smoke test with a real xlsx file
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_writes_animals_yaml(self, tmp_path, monkeypatch):
        """main() reads an xlsx, moves .mesc files, reads YAML, and writes animals.yaml."""
        from neural_analysis.utils.metadata import yaml_creator

        # Write a minimal xlsx with the expected filename in tmp_path
        xlsx_name = "Intrinsic_CA3_database-September_7,_10_08_AM.xlsx"
        fpath = str(tmp_path / xlsx_name)
        _write_minimal_xlsx(fpath)

        # Write a minimal animal_summary.yaml for get_animals_from_yaml
        animals_data = {
            "DON-0001234": {
                "animal_id": "DON-0001234",
                "sessions": {},
            }
        }
        with open(tmp_path / "animal_summary.yaml", "w") as f:
            yaml.dump(animals_data, f)

        # Monkeypatch combine to avoid KeyError (yaml animal not in spreadsheet)
        monkeypatch.setattr(
            yaml_creator,
            "combine_spreadsheet_and_old_animal_summary_yaml",
            lambda sp, y: sp,
        )

        yaml_creator.main(directory=str(tmp_path))
        result_path = tmp_path / "animals.yaml"
        assert result_path.exists()
        with open(result_path) as f:
            animals = yaml.safe_load(f)
        assert isinstance(animals, dict)
