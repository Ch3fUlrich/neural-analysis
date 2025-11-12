"""Tests for h5io save/load examples."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from neural_analysis.utils import h5io
from neural_analysis.utils.io import load_hdf5

if TYPE_CHECKING:
    from pathlib import Path


def test_h5io_array_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "array_demo.h5"
    data = np.random.randn(50, 4).astype(np.float32)
    labels = np.array([f"row_{i}" for i in range(data.shape[0])])
    attrs = {"info": {"k": 1}, "name": "demo"}

    h5io(path, task="save", data=data, labels=labels, attrs=attrs)
    loaded, loaded_labels = h5io(path, task="load")

    assert isinstance(loaded, np.ndarray)
    np.testing.assert_allclose(loaded, data)
    assert list(loaded_labels) == list(labels)


def test_h5io_dataframe_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "df_demo.h5"
    df = pd.DataFrame(
        {
            "id": [f"n{i}" for i in range(6)],
            "rate": np.linspace(0, 1, 6),
            "cond": ["A", "B", "A", "B", "A", "B"],
        }
    )
    labels = [f"trial_{i}" for i in range(len(df))]

    h5io(path, task="save", data=df, labels=labels)
    loaded, loaded_labels = h5io(path, task="load")

    assert hasattr(loaded, "equals")
    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True), df.reset_index(drop=True)
    )
    assert list(loaded_labels) == labels


def test_h5io_dataframe_filter_pairs(tmp_path: Path) -> None:
    path = tmp_path / "pairs_demo.h5"
    df = pd.DataFrame(
        {
            "item_i": ["A", "A", "B", "C"],
            "item_j": ["B", "C", "C", "D"],
            "score": [0.1, 0.8, 0.5, 0.9],
        }
    )

    h5io(path, task="save", data=df, labels=None)
    wanted = [("A", "C"), ("B", "C")]
    (loaded, _), _attrs = load_hdf5(path, filter_pairs=wanted, return_attrs=True)

    assert set(zip(loaded["item_i"], loaded["item_j"])) == set(wanted)
