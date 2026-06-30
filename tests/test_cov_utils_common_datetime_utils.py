"""Coverage tests for neural_analysis.utils.common.datetime_utils."""

from __future__ import annotations

from neural_analysis.utils.common.datetime_utils import (
    extract_date_from_filename,
    num_to_date,
)


def test_extract_date_from_filename_found() -> None:
    assert extract_date_from_filename("session_20240115_data.h5") == "20240115"


def test_extract_date_from_filename_none() -> None:
    assert extract_date_from_filename("no_digits_here.h5") is None


def test_num_to_date_valid() -> None:
    assert num_to_date("20240115") == "20240115"


def test_num_to_date_invalid_returns_none() -> None:
    # Exercises the except ValueError branch (lines 14-15).
    assert num_to_date("notadate") is None
    assert num_to_date("20241345") is None  # invalid month/day
