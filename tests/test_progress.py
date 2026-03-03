"""Tests for the progress bar utility."""

from __future__ import annotations

from neural_analysis.utils.progress import get_progress_bar


class TestGetProgressBar:
    def test_iterates_all_items(self) -> None:
        items = list(range(10))
        result = list(get_progress_bar(items, desc="test", disable=True))
        assert result == items

    def test_respects_disable(self) -> None:
        bar = get_progress_bar(range(5), disable=True)
        assert list(bar) == [0, 1, 2, 3, 4]

    def test_quiet_env_var(self, monkeypatch: object) -> None:
        import pytest

        mp = pytest.MonkeyPatch()
        mp.setenv("NEURAL_ANALYSIS_QUIET", "1")
        try:
            result = list(get_progress_bar(range(3)))
            assert result == [0, 1, 2]
        finally:
            mp.undo()

    def test_with_total(self) -> None:
        result = list(get_progress_bar(iter(range(4)), total=4, disable=True))
        assert result == [0, 1, 2, 3]
