"""Tests for neural_analysis.utils.file_management.paths — coverage target >= 95%."""

from __future__ import annotations

from pathlib import Path

import pytest

from neural_analysis.utils.file_management.paths import (
    get_directories,
    get_files,
    init_path_checks,
    regex_search,
    search_filedir,
)


# ---------------------------------------------------------------------------
# init_path_checks
# ---------------------------------------------------------------------------


class TestInitPathChecks:
    def test_dir_creates_missing_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "new_dir" / "nested"
        result = init_path_checks(target, check="dir")
        assert target.exists()
        assert target.is_dir()
        assert result == target

    def test_dir_returns_path_object_for_string(self, tmp_path: Path) -> None:
        target = tmp_path / "strdir"
        result = init_path_checks(str(target), check="dir")
        assert isinstance(result, Path)
        assert result.exists()

    def test_dir_existing_directory_no_error(self, tmp_path: Path) -> None:
        # Directory already exists — should not raise
        result = init_path_checks(tmp_path, check="dir")
        assert result == tmp_path
        assert result.is_dir()

    def test_file_valid_file_returns_path(self, tmp_path: Path) -> None:
        f = tmp_path / "myfile.txt"
        f.write_text("hello")
        result = init_path_checks(f, check="file")
        assert result == f

    def test_file_missing_file_raises_value_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "no_such_file.txt"
        with pytest.raises(ValueError, match="is not a valid file"):
            init_path_checks(missing, check="file")

    def test_file_string_path_missing_raises(self, tmp_path: Path) -> None:
        missing = str(tmp_path / "absent.csv")
        with pytest.raises(ValueError, match="is not a valid file"):
            init_path_checks(missing, check="file")

    def test_default_check_is_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "auto_dir"
        result = init_path_checks(target)  # default check="dir"
        assert target.is_dir()
        assert result == target


# ---------------------------------------------------------------------------
# search_filedir
# ---------------------------------------------------------------------------


class TestSearchFiledir:
    def test_nonexistent_path_returns_empty_list(self, tmp_path: Path) -> None:
        result = search_filedir(tmp_path / "does_not_exist")
        assert result == []

    def test_type_file_returns_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "subdir").mkdir()
        result = search_filedir(tmp_path, type="file")
        names = {p.name for p in result}
        assert "a.txt" in names
        assert "b.txt" in names
        assert "subdir" not in names

    def test_type_dir_returns_directories(self, tmp_path: Path) -> None:
        (tmp_path / "d1").mkdir()
        (tmp_path / "d2").mkdir()
        (tmp_path / "file.txt").write_text("x")
        result = search_filedir(tmp_path, type="dir")
        names = {p.name for p in result}
        assert "d1" in names
        assert "d2" in names
        assert "file.txt" not in names

    def test_type_dir_include_regex_filters(self, tmp_path: Path) -> None:
        (tmp_path / "alpha").mkdir()
        (tmp_path / "beta").mkdir()
        (tmp_path / "gamma").mkdir()
        result = search_filedir(tmp_path, type="dir", include_regex="^al")
        names = {p.name for p in result}
        assert "alpha" in names
        assert "beta" not in names
        assert "gamma" not in names

    def test_type_dir_exclude_regex_tuple(self, tmp_path: Path) -> None:
        (tmp_path / "keep").mkdir()
        (tmp_path / "skip").mkdir()
        result = search_filedir(tmp_path, type="dir", exclude_regex=("skip",))
        names = {p.name for p in result}
        assert "keep" in names
        assert "skip" not in names

    def test_type_dir_include_and_exclude_combined(self, tmp_path: Path) -> None:
        (tmp_path / "foo_bar").mkdir()
        (tmp_path / "foo_baz").mkdir()
        (tmp_path / "qux").mkdir()
        result = search_filedir(
            tmp_path,
            type="dir",
            include_regex="^foo",
            exclude_regex=("foo_baz",),
        )
        names = {p.name for p in result}
        assert "foo_bar" in names
        assert "foo_baz" not in names
        assert "qux" not in names

    def test_type_dir_no_include_no_exclude_returns_all(self, tmp_path: Path) -> None:
        (tmp_path / "x").mkdir()
        (tmp_path / "y").mkdir()
        result = search_filedir(tmp_path, type="dir")
        assert len(result) == 2

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        result = search_filedir(tmp_path, type="file")
        assert result == []

    def test_returns_list_of_path_objects(self, tmp_path: Path) -> None:
        (tmp_path / "f.csv").write_text("data")
        result = search_filedir(tmp_path, type="file")
        assert all(isinstance(p, Path) for p in result)

    def test_string_path_accepted(self, tmp_path: Path) -> None:
        (tmp_path / "x.txt").write_text("x")
        result = search_filedir(str(tmp_path), type="file")
        assert len(result) == 1

    def test_unknown_type_returns_empty(self, tmp_path: Path) -> None:
        # Neither "dir" nor "file" — hits the fall-through branch (33->37)
        (tmp_path / "d").mkdir()
        (tmp_path / "f.txt").write_text("x")
        result = search_filedir(tmp_path, type="symlink")
        assert result == []

    def test_type_dir_exclude_regex_item_not_in_tuple(self, tmp_path: Path) -> None:
        # exclude_regex provided but item name is NOT in it — should be included
        (tmp_path / "keep").mkdir()
        result = search_filedir(tmp_path, type="dir", exclude_regex=("other",))
        names = {p.name for p in result}
        assert "keep" in names


# ---------------------------------------------------------------------------
# regex_search
# ---------------------------------------------------------------------------


class TestRegexSearch:
    def _make_paths(self, names: list[str], base: Path) -> list[Path]:
        return [base / n for n in names]

    def test_matches_pattern(self, tmp_path: Path) -> None:
        paths = self._make_paths(["file_001.txt", "data_002.csv", "note.txt"], tmp_path)
        result = regex_search(paths, r"\d{3}")
        names = [p.name for p in result]
        assert "file_001.txt" in names
        assert "data_002.csv" in names
        assert "note.txt" not in names

    def test_no_match_returns_empty(self, tmp_path: Path) -> None:
        paths = self._make_paths(["abc.txt", "xyz.csv"], tmp_path)
        result = regex_search(paths, r"\d+")
        assert result == []

    def test_all_match(self, tmp_path: Path) -> None:
        paths = self._make_paths(["a1.txt", "b2.csv"], tmp_path)
        result = regex_search(paths, r"\w")
        assert len(result) == 2

    def test_empty_list_returns_empty(self, tmp_path: Path) -> None:
        result = regex_search([], r".*")
        assert result == []


# ---------------------------------------------------------------------------
# get_directories
# ---------------------------------------------------------------------------


class TestGetDirectories:
    def test_nonexistent_path_returns_empty(self, tmp_path: Path) -> None:
        result = get_directories(str(tmp_path / "no_such"))
        assert result == []

    def test_returns_only_directories(self, tmp_path: Path) -> None:
        (tmp_path / "d1").mkdir()
        (tmp_path / "d2").mkdir()
        (tmp_path / "file.txt").write_text("x")
        result = get_directories(str(tmp_path))
        assert set(result) == {"d1", "d2"}

    def test_regex_filters_directories(self, tmp_path: Path) -> None:
        (tmp_path / "session_01").mkdir()
        (tmp_path / "session_02").mkdir()
        (tmp_path / "other").mkdir()
        result = get_directories(str(tmp_path), regex_search=r"^session")
        assert set(result) == {"session_01", "session_02"}

    def test_no_regex_returns_all(self, tmp_path: Path) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        result = get_directories(str(tmp_path))
        assert set(result) == {"a", "b"}

    def test_empty_directory_returns_empty(self, tmp_path: Path) -> None:
        result = get_directories(str(tmp_path))
        assert result == []

    def test_returns_list_of_strings(self, tmp_path: Path) -> None:
        (tmp_path / "dir1").mkdir()
        result = get_directories(str(tmp_path))
        assert all(isinstance(d, str) for d in result)


# ---------------------------------------------------------------------------
# get_files
# ---------------------------------------------------------------------------


class TestGetFiles:
    def test_nonexistent_path_returns_empty(self, tmp_path: Path) -> None:
        result = get_files(str(tmp_path / "no_such"))
        assert result == []

    def test_returns_only_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.csv").write_text("b")
        (tmp_path / "subdir").mkdir()
        result = get_files(str(tmp_path))
        assert set(result) == {"a.txt", "b.csv"}

    def test_ending_filter(self, tmp_path: Path) -> None:
        (tmp_path / "data.csv").write_text("x")
        (tmp_path / "report.txt").write_text("y")
        (tmp_path / "notes.csv").write_text("z")
        result = get_files(str(tmp_path), ending=".csv")
        assert set(result) == {"data.csv", "notes.csv"}

    def test_regex_search_filter(self, tmp_path: Path) -> None:
        (tmp_path / "run_001.txt").write_text("a")
        (tmp_path / "run_002.txt").write_text("b")
        (tmp_path / "other.txt").write_text("c")
        result = get_files(str(tmp_path), regex_search=r"run_\d+")
        assert set(result) == {"run_001.txt", "run_002.txt"}

    def test_ending_and_regex_combined(self, tmp_path: Path) -> None:
        (tmp_path / "run_001.csv").write_text("a")
        (tmp_path / "run_002.txt").write_text("b")
        (tmp_path / "other.csv").write_text("c")
        result = get_files(str(tmp_path), ending=".csv", regex_search=r"run_\d+")
        assert set(result) == {"run_001.csv"}

    def test_no_filter_returns_all_files(self, tmp_path: Path) -> None:
        (tmp_path / "x.txt").write_text("x")
        (tmp_path / "y.txt").write_text("y")
        result = get_files(str(tmp_path))
        assert set(result) == {"x.txt", "y.txt"}

    def test_empty_directory_returns_empty(self, tmp_path: Path) -> None:
        result = get_files(str(tmp_path))
        assert result == []

    def test_returns_list_of_strings(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("data")
        result = get_files(str(tmp_path))
        assert all(isinstance(f, str) for f in result)

    def test_ending_no_match_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("data")
        result = get_files(str(tmp_path), ending=".csv")
        assert result == []

    def test_regex_no_match_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("data")
        result = get_files(str(tmp_path), regex_search=r"\d{5}")
        assert result == []
