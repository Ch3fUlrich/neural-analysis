"""Targeted coverage tests for neural_analysis.utils.io.

Focus: lines/branches not covered by existing test files.
Missing (from baseline run): 38-51, 84-85, 146, 160, 194-204, 246, 291,
294->297, 311-312, 348, 365->367, 368, 411-419, 424-425, 433->471, 476,
683, 690->689, 695-697, 799-800, 811-813, 827, 837-839, 872-874.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest

from neural_analysis.utils.io import (
    _attr_equals,
    _from_bytes_array,
    _load_dataframe,
    _normalize_attr_value,
    _resolve_storage_manager,
    _save_dataframe,
    _to_bytes_array,
    _write_attrs,
    get_hdf5_dataset_names,
    load_hdf5,
    load_results_from_hdf5_dataset,
    save_array,
    save_hdf5,
    save_result_to_hdf5_dataset,
    update_array,
)


# ---------------------------------------------------------------------------
# _resolve_storage_manager — exception path (lines 84-85)
# ---------------------------------------------------------------------------


def test_resolve_storage_manager_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """When StorageManager import raises, _resolve_storage_manager returns None."""
    import sys

    # Remove the cached module so the import block runs again
    original = sys.modules.get("neural_analysis.utils.storage.manager")
    sys.modules["neural_analysis.utils.storage.manager"] = None  # type: ignore[assignment]
    try:
        result = _resolve_storage_manager(None, use_cache=True, use_sql=True)
        # Should return None because the import raises
        assert result is None
    finally:
        if original is None:
            sys.modules.pop("neural_analysis.utils.storage.manager", None)
        else:
            sys.modules["neural_analysis.utils.storage.manager"] = original


# ---------------------------------------------------------------------------
# _save_dataframe — string index (line 146) and existing cols_grp (line 160)
# ---------------------------------------------------------------------------


def test_save_dataframe_string_index(tmp_path: Path) -> None:
    """_save_dataframe saves string index via _to_bytes_array (line 146)."""
    df = pd.DataFrame(
        {"val": [10.0, 20.0, 30.0]},
        index=pd.Index(["row_a", "row_b", "row_c"]),
    )
    h5_path = tmp_path / "str_index.h5"
    with h5py.File(h5_path, "w") as f:
        grp = f.create_group("data")
        _save_dataframe(grp, df, compression="gzip", compression_opts=4)

    # Reload and verify round-trip
    with h5py.File(h5_path, "r") as f:
        loaded = _load_dataframe(f["data"])

    assert list(loaded.index) == ["row_a", "row_b", "row_c"]
    assert list(loaded["val"]) == [10.0, 20.0, 30.0]


def test_save_dataframe_existing_cols_grp_with_children(tmp_path: Path) -> None:
    """_save_dataframe deletes existing children of columns_data group (line 160).

    Creates a columns_data group with pre-existing children so that line 160
    (del cols_grp[key]) is executed.
    """
    df = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
    h5_path = tmp_path / "existing_grp_children.h5"

    with h5py.File(h5_path, "w") as f:
        grp = f.create_group("data")
        # Pre-create columns_data with stale children to trigger del at line 160
        cols_grp = grp.create_group("columns_data")
        cols_grp.create_dataset("stale_col", data=np.array([99, 88]))
        _save_dataframe(grp, df, compression="gzip", compression_opts=4)

    with h5py.File(h5_path, "r") as f:
        loaded = _load_dataframe(f["data"])

    # stale_col must have been cleared and replaced with fresh columns
    assert "a" in loaded.columns
    assert "b" in loaded.columns
    assert loaded["a"].tolist() == [5, 6]
    assert loaded["b"].tolist() == [7, 8]


# ---------------------------------------------------------------------------
# _load_dataframe — fallback values path with object dtype (lines 194-204)
# ---------------------------------------------------------------------------


def test_load_dataframe_fallback_numeric_values(tmp_path: Path) -> None:
    """_load_dataframe falls back to flat 'values' dataset when no columns_data group."""
    h5_path = tmp_path / "fallback_values.h5"
    # Build HDF5 file that has no columns_data group but has a 'values' dataset
    with h5py.File(h5_path, "w") as f:
        grp = f.create_group("data")
        grp.create_dataset(
            "index_is_numeric", data=np.array([True])
        )
        grp.create_dataset("index", data=np.array([0, 1]))
        grp.create_dataset("columns", data=np.array([b"x", b"y"], dtype="S"))
        # Numeric values — no object dtype conversion needed
        grp.create_dataset("values", data=np.array([[1.0, 2.0], [3.0, 4.0]]))

    with h5py.File(h5_path, "r") as f:
        df = _load_dataframe(f["data"])

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert df["x"].tolist() == [1.0, 3.0]
    assert df["y"].tolist() == [2.0, 4.0]


def test_load_dataframe_fallback_object_values(tmp_path: Path) -> None:
    """_load_dataframe falls back to flat 'values' dataset with object dtype bytes (lines 193-204)."""
    h5_path = tmp_path / "fallback_obj_values.h5"
    # Build HDF5 file with variable-length bytes dataset (reads back as object dtype)
    with h5py.File(h5_path, "w") as f:
        grp = f.create_group("data")
        grp.create_dataset(
            "index_is_numeric", data=np.array([True])
        )
        grp.create_dataset("index", data=np.array([0, 1]))
        grp.create_dataset("columns", data=np.array([b"x", b"y"], dtype="S"))
        # Use variable-length bytes type — h5py reads it back as object dtype with bytes
        dt = h5py.special_dtype(vlen=bytes)
        raw = np.empty((2, 2), dtype=object)
        raw[0, 0] = b"hello"; raw[0, 1] = b"world"
        raw[1, 0] = b"foo"; raw[1, 1] = b"bar"
        grp.create_dataset("values", data=raw, dtype=dt)

    with h5py.File(h5_path, "r") as f:
        # Confirm the dataset is read as object dtype
        vals = f["data"]["values"][...]
        assert vals.dtype == object
        df = _load_dataframe(f["data"])

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)


# ---------------------------------------------------------------------------
# save_array — dict overwrite protection (line 246)
# ---------------------------------------------------------------------------


def test_save_array_dict_no_overwrite_raises(tmp_path: Path) -> None:
    """save_array raises ValueError when allow_overwrite=False and .npz exists."""
    data = {"arr": np.array([1, 2, 3])}
    p = tmp_path / "existing.npz"
    save_array(p, data)
    assert p.exists()

    with pytest.raises(ValueError, match="already exists"):
        save_array(p, data, allow_overwrite=False)


# ---------------------------------------------------------------------------
# update_array — create new file from scratch (line 291, 294->297 branch)
# ---------------------------------------------------------------------------


def test_update_array_creates_new_file(tmp_path: Path) -> None:
    """update_array works when the target .npz file does not yet exist."""
    p = tmp_path / "fresh.npz"
    assert not p.exists()

    result = update_array(p, {"k1": np.array([10, 20, 30])})
    assert result.exists()
    with np.load(result, allow_pickle=False) as data:
        assert "k1" in data.files
        np.testing.assert_array_equal(data["k1"], [10, 20, 30])


def test_update_array_auto_adds_npz_extension(tmp_path: Path) -> None:
    """update_array appends .npz when path has no extension."""
    p = tmp_path / "noext"
    result = update_array(p, {"arr": np.zeros(5)})
    assert result.suffix == ".npz"
    assert result.exists()


# ---------------------------------------------------------------------------
# _write_attrs — exception path (lines 311-312)
# ---------------------------------------------------------------------------


def test_write_attrs_exception_path(tmp_path: Path) -> None:
    """_write_attrs falls back to json.dumps(str(v)) when serialisation fails."""
    h5_path = tmp_path / "attr_exc.h5"

    class Unserializable:
        def __repr__(self) -> str:
            return "Unserializable()"

    with h5py.File(h5_path, "w") as f:
        # Pass a value that is not a basic type and whose json.dumps also works
        # by hitting the except branch and calling json.dumps(str(v))
        _write_attrs(f, {"tricky": Unserializable()})
        # The key should have been stored as the JSON-encoded string repr
        raw = f.attrs["tricky"]
        assert "Unserializable" in json.loads(raw)


# ---------------------------------------------------------------------------
# save_hdf5 — append mode with existing 'data' group (DataFrame, line 348)
# ---------------------------------------------------------------------------


def test_save_hdf5_dataframe_append_mode_replaces(tmp_path: Path) -> None:
    """save_hdf5 in append mode ('a') replaces existing DataFrame group."""
    p = tmp_path / "append_df.h5"
    df1 = pd.DataFrame({"col": [1.0, 2.0]})
    df2 = pd.DataFrame({"col": [9.0, 8.0, 7.0]})

    save_hdf5(p, df1, mode="w")
    save_hdf5(p, df2, mode="a")

    loaded, _ = load_hdf5(p)
    assert isinstance(loaded, pd.DataFrame)
    assert len(loaded) == 3
    assert loaded["col"].tolist() == [9.0, 8.0, 7.0]


# ---------------------------------------------------------------------------
# save_hdf5 — append mode replaces existing labels dataset (line 367-368)
# ---------------------------------------------------------------------------


def test_save_hdf5_labels_replaced_in_append_mode(tmp_path: Path) -> None:
    """save_hdf5 in append mode deletes and recreates the labels dataset (lines 367-368)."""
    p = tmp_path / "labels_replace.h5"
    arr = np.array([1.0, 2.0, 3.0])
    labels1 = np.array([10, 20, 30], dtype=np.int32)  # numeric labels (non-S dtype)
    labels2 = np.array([100, 200, 300], dtype=np.int32)

    save_hdf5(p, arr, labels=labels1, mode="w")
    # Second write in append mode: 'labels' already exists -> must delete and recreate
    save_hdf5(p, arr, labels=labels2, mode="a")

    loaded_arr, loaded_labels = load_hdf5(p)
    np.testing.assert_array_equal(loaded_labels, labels2)


# ---------------------------------------------------------------------------
# save_hdf5 / load_hdf5 — string/object-dtype labels (lines 365->367, 368)
# ---------------------------------------------------------------------------


def test_save_hdf5_string_labels_roundtrip(tmp_path: Path) -> None:
    """String labels are bytes-encoded on save and decoded back on load."""
    arr = np.arange(6, dtype=np.float64).reshape(2, 3)
    labels = ["alpha", "beta"]
    p = tmp_path / "str_labels.h5"

    save_hdf5(p, arr, labels=labels)
    loaded_arr, loaded_labels = load_hdf5(p)

    np.testing.assert_array_equal(loaded_arr, arr)
    assert loaded_labels == labels


def test_save_hdf5_object_array_labels(tmp_path: Path) -> None:
    """Object-dtype labels array is also bytes-encoded (line 366 branch)."""
    arr = np.ones((3, 2))
    labels = np.array(["x", "y", "z"], dtype=object)
    p = tmp_path / "obj_labels.h5"

    save_hdf5(p, arr, labels=labels)
    loaded_arr, loaded_labels = load_hdf5(p)

    assert loaded_labels == ["x", "y", "z"]


# ---------------------------------------------------------------------------
# load_hdf5 — np.bytes_ attribute path (lines 411-419)
# ---------------------------------------------------------------------------


def test_load_hdf5_np_bytes_attr(tmp_path: Path) -> None:
    """load_hdf5 correctly decodes np.bytes_ attributes stored directly."""
    p = tmp_path / "np_bytes_attr.h5"
    with h5py.File(p, "w") as f:
        # h5py stores byte-string attrs as np.bytes_
        f.attrs["meta"] = np.bytes_(b"plain text")
        f.create_dataset("data", data=np.array([1.0, 2.0]))

    (data_out, _), attrs = load_hdf5(p, return_attrs=True)
    assert "meta" in attrs
    assert attrs["meta"] == "plain text"


def test_load_hdf5_np_bytes_attr_json(tmp_path: Path) -> None:
    """load_hdf5 parses JSON-encoded np.bytes_ attributes."""
    p = tmp_path / "np_bytes_json.h5"
    encoded = json.dumps({"nested": 42}).encode("utf-8")
    with h5py.File(p, "w") as f:
        f.attrs["info"] = np.bytes_(encoded)
        f.create_dataset("data", data=np.array([0.0]))

    (_, _), attrs = load_hdf5(p, return_attrs=True)
    assert attrs["info"] == {"nested": 42}


# ---------------------------------------------------------------------------
# load_hdf5 — attr with .item() that raises (lines 424-425)
# ---------------------------------------------------------------------------


def test_load_hdf5_attr_item_raises(tmp_path: Path) -> None:
    """load_hdf5 stores the raw value when .item() raises."""
    p = tmp_path / "attr_item_exc.h5"
    # Write a numpy array attribute — h5py may read it back as an ndarray
    # whose .item() call raises ValueError when not 0-d.
    with h5py.File(p, "w") as f:
        f.attrs["arr_attr"] = np.array([1, 2, 3])
        f.create_dataset("data", data=np.array([7.0]))

    (_, _), attrs = load_hdf5(p, return_attrs=True)
    # The value must have been stored (not dropped)
    assert "arr_attr" in attrs


# ---------------------------------------------------------------------------
# load_hdf5 — no 'data' key in file (line 433->471 branch: "data" not in f)
# ---------------------------------------------------------------------------


def test_load_hdf5_no_data_key(tmp_path: Path) -> None:
    """load_hdf5 returns (None, []) when the HDF5 file has no 'data' dataset."""
    p = tmp_path / "no_data.h5"
    with h5py.File(p, "w") as f:
        f.attrs["version"] = 1

    data_out, labels_out = load_hdf5(p)
    assert data_out is None
    assert labels_out == []


# ---------------------------------------------------------------------------
# load_hdf5 — numeric (non-S) labels (line 476)
# ---------------------------------------------------------------------------


def test_load_hdf5_numeric_labels(tmp_path: Path) -> None:
    """load_hdf5 returns numeric labels array unchanged when dtype is not S."""
    arr = np.array([1.0, 2.0, 3.0])
    p = tmp_path / "num_labels.h5"

    with h5py.File(p, "w") as f:
        f.create_dataset("data", data=arr)
        f.create_dataset("labels", data=np.array([10, 20, 30], dtype=np.int32))

    loaded_arr, loaded_labels = load_hdf5(p)
    np.testing.assert_array_equal(loaded_arr, arr)
    np.testing.assert_array_equal(loaded_labels, [10, 20, 30])


# ---------------------------------------------------------------------------
# save_result_to_hdf5_dataset — existing ds_group (line 683)
#   and None value skipped (line 690->689) and non-scalar converted (695-697)
# ---------------------------------------------------------------------------


def test_save_result_existing_dataset_group(tmp_path: Path) -> None:
    """save_result_to_hdf5_dataset appends into an existing top-level group."""
    p = tmp_path / "results.h5"

    # First call creates the dataset group
    save_result_to_hdf5_dataset(
        p,
        dataset_name="session_A",
        result_key="run_1",
        scalar_data={"metric": "cos", "val": 0.5},
        array_data={"vec": np.array([1.0, 2.0])},
        use_cache=False,
        use_sql_index=False,
    )

    # Second call must reuse existing "session_A" group (line 679 branch)
    save_result_to_hdf5_dataset(
        p,
        dataset_name="session_A",
        result_key="run_2",
        scalar_data={"metric": "euc", "val": 0.8},
        array_data={"vec": np.array([3.0, 4.0])},
        use_cache=False,
        use_sql_index=False,
    )

    with h5py.File(p, "r") as f:
        assert "session_A" in f
        assert "run_1" in f["session_A"]
        assert "run_2" in f["session_A"]


def test_save_result_overwrite_existing_result_key(tmp_path: Path) -> None:
    """save_result_to_hdf5_dataset overwrites an existing result_key."""
    p = tmp_path / "overwrite.h5"
    save_result_to_hdf5_dataset(
        p,
        dataset_name="ds",
        result_key="res",
        scalar_data={"v": 1},
        array_data={"a": np.array([0.0])},
        use_cache=False,
        use_sql_index=False,
    )
    # Overwrite with different value
    save_result_to_hdf5_dataset(
        p,
        dataset_name="ds",
        result_key="res",
        scalar_data={"v": 99},
        array_data={"a": np.array([7.0])},
        use_cache=False,
        use_sql_index=False,
    )
    with h5py.File(p, "r") as f:
        assert f["ds"]["res"].attrs["v"] == 99


def test_save_result_none_scalar_skipped(tmp_path: Path) -> None:
    """None values in scalar_data are skipped (line 690->689 branch)."""
    p = tmp_path / "none_scalar.h5"
    save_result_to_hdf5_dataset(
        p,
        dataset_name="ds",
        result_key="r1",
        scalar_data={"present": "yes", "absent": None},
        array_data={},
        use_cache=False,
        use_sql_index=False,
    )
    with h5py.File(p, "r") as f:
        attrs = dict(f["ds"]["r1"].attrs)
    assert "present" in attrs
    assert "absent" not in attrs


def test_save_result_non_scalar_converted_to_str(tmp_path: Path) -> None:
    """Non-str/int/float/bool scalar values are stored as str (lines 695-697)."""
    p = tmp_path / "nonscalar.h5"
    save_result_to_hdf5_dataset(
        p,
        dataset_name="ds",
        result_key="r1",
        scalar_data={"tags": ["a", "b"]},
        array_data={},
        use_cache=False,
        use_sql_index=False,
    )
    with h5py.File(p, "r") as f:
        stored = f["ds"]["r1"].attrs["tags"]
    # Should have been converted to str representation
    assert isinstance(stored, str)


# ---------------------------------------------------------------------------
# load_results_from_hdf5_dataset — dataset_name not found (lines 799-800)
# ---------------------------------------------------------------------------


def test_load_results_dataset_name_not_found(tmp_path: Path) -> None:
    """load_results_from_hdf5_dataset returns {} when dataset_name absent."""
    p = tmp_path / "partial.h5"
    save_result_to_hdf5_dataset(
        p,
        dataset_name="exists",
        result_key="key",
        scalar_data={"v": 1},
        array_data={},
        use_cache=False,
        use_sql_index=False,
    )

    result = load_results_from_hdf5_dataset(p, dataset_name="does_not_exist")
    assert result == {}


# ---------------------------------------------------------------------------
# load_results_from_hdf5_dataset — result_key not in ds_group (lines 811-813)
# ---------------------------------------------------------------------------


def test_load_results_result_key_not_found(tmp_path: Path) -> None:
    """load_results_from_hdf5_dataset skips missing result_key (line 812 continue)."""
    p = tmp_path / "rk_miss.h5"
    save_result_to_hdf5_dataset(
        p,
        dataset_name="ds",
        result_key="present",
        scalar_data={"x": 1},
        array_data={},
        use_cache=False,
        use_sql_index=False,
    )

    result = load_results_from_hdf5_dataset(
        p, dataset_name="ds", result_key="absent"
    )
    # "ds" key exists but "absent" is not in ds_group -> continue -> empty inner dict
    assert result == {"ds": {}}


def test_load_results_specific_result_key_found(tmp_path: Path) -> None:
    """load_results_from_hdf5_dataset loads only the specified result_key (line 813)."""
    p = tmp_path / "rk_found.h5"
    save_result_to_hdf5_dataset(
        p,
        dataset_name="ds",
        result_key="run_1",
        scalar_data={"score": 0.42},
        array_data={"vec": np.array([1.0, 2.0])},
        use_cache=False,
        use_sql_index=False,
    )
    save_result_to_hdf5_dataset(
        p,
        dataset_name="ds",
        result_key="run_2",
        scalar_data={"score": 0.99},
        array_data={"vec": np.array([3.0, 4.0])},
        use_cache=False,
        use_sql_index=False,
    )

    result = load_results_from_hdf5_dataset(
        p, dataset_name="ds", result_key="run_1"
    )
    # Only run_1 should be returned (line 813 path: result_keys = [result_key])
    assert "ds" in result
    assert "run_1" in result["ds"]
    assert "run_2" not in result["ds"]
    np.testing.assert_array_almost_equal(
        result["ds"]["run_1"]["arrays"]["vec"], [1.0, 2.0]
    )


# ---------------------------------------------------------------------------
# load_results_from_hdf5_dataset — filter_attrs rejects entry (line 827)
# ---------------------------------------------------------------------------


def test_load_results_filter_attrs_skips_non_matching(tmp_path: Path) -> None:
    """filter_attrs excludes results whose attrs don't match."""
    p = tmp_path / "filter.h5"
    for i in range(3):
        save_result_to_hdf5_dataset(
            p,
            dataset_name="ds",
            result_key=f"run_{i}",
            scalar_data={"score": float(i)},
            array_data={"v": np.array([float(i)])},
            use_cache=False,
            use_sql_index=False,
        )

    result = load_results_from_hdf5_dataset(
        p, dataset_name="ds", filter_attrs={"score": 1.0}
    )
    assert "ds" in result
    assert len(result["ds"]) == 1
    assert "run_1" in result["ds"]


# ---------------------------------------------------------------------------
# load_results_from_hdf5_dataset — exception path (lines 837-839)
# ---------------------------------------------------------------------------


def test_load_results_exception_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """load_results_from_hdf5_dataset returns {} when h5py.File raises."""
    p = tmp_path / "bad.h5"
    p.write_bytes(b"not hdf5 content")

    result = load_results_from_hdf5_dataset(p, dataset_name="any")
    assert result == {}


# ---------------------------------------------------------------------------
# get_hdf5_dataset_names — exception path (lines 872-874)
# ---------------------------------------------------------------------------


def test_get_hdf5_dataset_names_exception(tmp_path: Path) -> None:
    """get_hdf5_dataset_names returns [] when h5py.File raises."""
    p = tmp_path / "corrupt.h5"
    p.write_bytes(b"not a valid hdf5 file at all")

    names = get_hdf5_dataset_names(p)
    assert names == []


# ---------------------------------------------------------------------------
# load_hdf5 — return_attrs=True when file is missing (lines 393 branch)
# ---------------------------------------------------------------------------


def test_load_hdf5_missing_file_return_attrs(tmp_path: Path) -> None:
    """load_hdf5 returns ((None, []), {}) when file missing and return_attrs=True."""
    p = tmp_path / "ghost.h5"
    result = load_hdf5(p, return_attrs=True)
    (data_out, labels_out), attrs_out = result  # type: ignore[misc]
    assert data_out is None
    assert labels_out == []
    assert attrs_out == {}


# ---------------------------------------------------------------------------
# save_hdf5 / load_hdf5 — append mode replaces existing array 'data' (line 354)
# ---------------------------------------------------------------------------


def test_save_hdf5_array_append_mode_replaces(tmp_path: Path) -> None:
    """save_hdf5 in append mode ('a') replaces existing array dataset."""
    p = tmp_path / "arr_append.h5"
    arr1 = np.array([1.0, 2.0, 3.0])
    arr2 = np.array([9.0, 8.0])

    save_hdf5(p, arr1, mode="w")
    save_hdf5(p, arr2, mode="a")

    loaded, _ = load_hdf5(p)
    np.testing.assert_array_equal(loaded, arr2)


# ---------------------------------------------------------------------------
# _load_dataframe — index_is_numeric=False (already existing path but ensure
# string-index round-trips via _from_bytes_array properly)
# ---------------------------------------------------------------------------


def test_load_dataframe_string_index_roundtrip(tmp_path: Path) -> None:
    """DataFrame with string index round-trips correctly through HDF5."""
    df = pd.DataFrame(
        {"score": [0.1, 0.2, 0.3]},
        index=pd.Index(["alpha", "beta", "gamma"]),
    )
    p = tmp_path / "str_idx_rt.h5"
    save_hdf5(p, df)
    loaded, _ = load_hdf5(p)
    assert isinstance(loaded, pd.DataFrame)
    assert list(loaded.index) == ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# _save_dataframe — string/object-dtype column values (line 163-164)
# ---------------------------------------------------------------------------


def test_save_dataframe_object_column_values(tmp_path: Path) -> None:
    """Columns with object/Unicode dtype are bytes-encoded in HDF5."""
    df = pd.DataFrame({"label": ["cat", "dog", "bird"], "score": [1.0, 2.0, 3.0]})
    p = tmp_path / "obj_col.h5"
    save_hdf5(p, df)
    loaded, _ = load_hdf5(p)
    assert loaded["label"].tolist() == ["cat", "dog", "bird"]
    assert loaded["score"].tolist() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# load_hdf5 — filter_pairs with a list (has __len__) vs generator (no __len__)
# ---------------------------------------------------------------------------


def test_load_hdf5_filter_pairs_generator(tmp_path: Path) -> None:
    """filter_pairs works correctly when provided as a generator (no __len__)."""
    df = pd.DataFrame(
        {
            "item_i": ["A", "B", "C"],
            "item_j": ["X", "Y", "Z"],
            "v": [1, 2, 3],
        }
    )
    p = tmp_path / "gen_filter.h5"
    save_hdf5(p, df)

    # Use a generator — has no __len__, hits the "N" branch
    def pair_gen():
        yield ("A", "X")
        yield ("C", "Z")

    loaded, _ = load_hdf5(p, filter_pairs=pair_gen())
    assert isinstance(loaded, pd.DataFrame)
    assert len(loaded) == 2
    assert set(loaded["item_i"]) == {"A", "C"}
