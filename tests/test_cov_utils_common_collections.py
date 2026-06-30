"""Coverage tests for neural_analysis.utils.common.collections."""

from __future__ import annotations

import numpy as np
import pytest

from neural_analysis.utils.common import collections as C


def test_flatten() -> None:
    assert C.flatten([[1, 2], [3], [4, 5]]) == [1, 2, 3, 4, 5]


def test_unique() -> None:
    assert sorted(C.unique([1, 1, 2, 3, 3])) == [1, 2, 3]


def test_is_array_like() -> None:
    assert C.is_array_like([1, 2, 3]) is True
    assert C.is_array_like((1, 2)) is True
    assert C.is_array_like("abc") is False
    assert C.is_array_like(b"abc") is False
    assert C.is_array_like(5) is False


def test_make_list_ifnot() -> None:
    assert C.make_list_ifnot([1, 2]) == [1, 2]
    assert C.make_list_ifnot((1, 2)) == [1, 2]
    assert C.make_list_ifnot(5) == [5]


def test_mean_diff() -> None:
    x = np.array([[2.0, 4.0], [4.0, 6.0]])
    y = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert np.allclose(C.mean_diff(x, y, axis=0), [2.0, 4.0])


def test_do_critical_raises_and_logs() -> None:
    messages: list[str] = []

    class _Logger:
        def critical(self, m: str) -> None:
            messages.append(m)

    with pytest.raises(ValueError, match="boom"):
        C.do_critical(ValueError, "boom", logger=_Logger())
    assert messages == ["boom"]


def test_do_critical_without_logger() -> None:
    with pytest.raises(RuntimeError, match="x"):
        C.do_critical(RuntimeError, "x")
