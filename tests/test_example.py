from neural_analysis.example import mean, normalize


def test_mean_simple():
    assert mean([1.0, 2.0, 3.0]) == 2.0


def test_mean_empty_raises():
    import pytest

    with pytest.raises(ValueError):
        mean([])


def test_normalize_range():
    vals = [2.0, 4.0, 6.0]
    norm = normalize(vals)
    assert min(norm) == 0.0
    assert max(norm) == 1.0


def test_normalize_constant():
    assert normalize([5.0, 5.0]) == [0.0, 0.0]
