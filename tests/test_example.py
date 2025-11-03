import pytest

from neural_analysis.example import mean, normalize


def test_mean_simple() -> None:
    assert mean([1.0, 2.0, 3.0]) == 2.0


def test_mean_empty_raises() -> None:
    with pytest.raises(ValueError):
        mean([])


def test_normalize_range() -> None:
    vals = [2.0, 4.0, 6.0]
    norm = normalize(vals)
    assert min(norm) == 0.0
    assert max(norm) == 1.0


def test_normalize_constant() -> None:
    assert normalize([5.0, 5.0]) == [0.0, 0.0]
