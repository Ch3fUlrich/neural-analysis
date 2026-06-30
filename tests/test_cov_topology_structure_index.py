"""Additional tests to increase coverage of neural_analysis.topology.structure_index.

These tests target the specific uncovered lines and branches identified in the
baseline run, including:
- StructureIndexConfig.to_kwargs with radius
- validate_args_types arg_provided=False branch
- _filter_noisy_outliers empty-array early return
- _cloud_overlap_neighbors geodesic path
- _cloud_overlap_radius geodesic path
- compute_structure_index: config kwarg, 1D label, data transposition,
  verbose paths, filter_noise, discrete_label list, n_bins list,
  num_shuffles=0 early return, seeded shuffling with verbose
- draw_overlap_graph with numeric node_names and with custom node_color
- _array_to_key hash path (>10 elements)
- compute_structure_index_sweep: data_indices, regenerate, storage fallback
- load_structure_index_results: non-existent file, filters
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neural_analysis.topology.structure_index import (
    StructureIndexConfig,
    _array_to_key,
    _cloud_overlap_neighbors,
    _cloud_overlap_radius,
    _filter_noisy_outliers,
    _structure_index_cache_key,
    _parse_loaded_results,
    compute_structure_index,
    draw_overlap_graph,
    validate_args_types,
)
from neural_analysis.topology import (
    compute_structure_index_sweep,
    load_structure_index_results,
)


# ---------------------------------------------------------------------------
# StructureIndexConfig
# ---------------------------------------------------------------------------


class TestStructureIndexConfig:
    """Tests for StructureIndexConfig dataclass and to_kwargs."""

    def test_to_kwargs_defaults(self) -> None:
        cfg = StructureIndexConfig()
        kw = cfg.to_kwargs()
        assert kw["distance_metric"] == "euclidean"
        assert kw["n_neighbors"] == 15
        assert kw["num_shuffles"] == 100
        assert kw["discrete_label"] is False
        assert kw["verbose"] is False
        # radius is None by default → no radius key, n_neighbors present
        assert "radius" not in kw
        assert "n_neighbors" in kw

    def test_to_kwargs_with_radius(self) -> None:
        """Cover lines 163-165: radius branch pops n_neighbors."""
        cfg = StructureIndexConfig(radius=2.5, n_neighbors=10)
        kw = cfg.to_kwargs()
        assert "radius" in kw
        assert kw["radius"] == 2.5
        # n_neighbors should have been popped
        assert "n_neighbors" not in kw

    def test_to_kwargs_with_radius_zero(self) -> None:
        """radius=0 is falsy but StructureIndexConfig stores it; None check is explicit."""
        cfg = StructureIndexConfig(radius=None)
        kw = cfg.to_kwargs()
        assert "radius" not in kw

    def test_config_passed_to_compute_structure_index(self) -> None:
        """Cover lines 616-618: config kwarg merging in compute_structure_index."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(80, 4))
        label = rng.uniform(size=(80, 1))

        cfg = StructureIndexConfig(
            n_neighbors=5,
            num_shuffles=0,
            verbose=False,
        )
        si, bin_info, overlap_mat, shuf_si = compute_structure_index(
            data, label, n_bins=4, config=cfg
        )
        assert isinstance(si, (float, np.floating))
        assert len(shuf_si) == 0  # num_shuffles=0

    def test_config_kwargs_override(self) -> None:
        """kwargs override config fields."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(80, 4))
        label = rng.uniform(size=(80, 1))

        cfg = StructureIndexConfig(num_shuffles=50, n_neighbors=5)
        # Override num_shuffles via kwargs
        si, _, _, shuf_si = compute_structure_index(
            data, label, n_bins=4, config=cfg, num_shuffles=3
        )
        assert len(shuf_si) == 3


# ---------------------------------------------------------------------------
# validate_args_types
# ---------------------------------------------------------------------------


class TestValidateArgsTypes:
    """Test the validate_args_types decorator."""

    def test_type_check_passes(self) -> None:
        """Decorator allows valid types."""
        @validate_args_types(x=int)
        def f(x: int) -> int:
            return x * 2

        assert f(3) == 6

    def test_type_check_fails(self) -> None:
        """Decorator raises TypeError on wrong type."""
        @validate_args_types(x=int)
        def f(x: int) -> int:
            return x * 2

        with pytest.raises(TypeError, match="type is"):
            f("hello")  # type: ignore[arg-type]

    def test_arg_not_provided(self) -> None:
        """Cover line 192: arg_provided=False when arg not in names or kwargs."""
        @validate_args_types(missing_arg=int)
        def f(x: int) -> int:
            return x

        # 'missing_arg' is not a parameter of f → arg_provided=False → no error
        assert f(5) == 5

    def test_kwarg_type_check(self) -> None:
        """Decorator also checks kwargs-passed arguments."""
        @validate_args_types(x=str)
        def f(x: str) -> str:
            return x

        with pytest.raises(TypeError, match="type is"):
            f(x=42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _filter_noisy_outliers
# ---------------------------------------------------------------------------


class TestFilterNoisyOutliers:
    """Test _filter_noisy_outliers, including edge cases."""

    def test_empty_array(self) -> None:
        """Cover line 247: early return for empty array."""
        result = _filter_noisy_outliers(np.array([]))
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int64
        assert len(result) == 0

    def test_empty_2d_array(self) -> None:
        result = _filter_noisy_outliers(np.zeros((0, 3)))
        assert len(result) == 0

    def test_with_clear_outlier(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.normal(size=(100, 2))
        data[0, :] = [100.0, 100.0]  # extreme outlier
        result = _filter_noisy_outliers(data, zscore_thresh=3.5)
        assert 0 in result  # index 0 must be flagged

    def test_returns_sorted_unique(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.normal(size=(60, 2))
        result = _filter_noisy_outliers(data)
        assert np.all(result[:-1] < result[1:])  # sorted


# ---------------------------------------------------------------------------
# _cloud_overlap_neighbors – geodesic path
# ---------------------------------------------------------------------------


class TestCloudOverlapNeighborsGeodesic:
    """Cover lines 437-441 (geodesic path in _cloud_overlap_neighbors)."""

    def test_geodesic_overlap(self) -> None:
        rng = np.random.default_rng(0)
        cloud1 = rng.normal(size=(20, 2))
        cloud2 = rng.normal(size=(20, 2)) + 3.0

        ov12, ov21 = _cloud_overlap_neighbors(
            cloud1, cloud2, k=5, distance_metric="geodesic"
        )
        assert 0.0 <= ov12 <= 1.0
        assert 0.0 <= ov21 <= 1.0

    def test_invalid_metric_neighbors(self) -> None:
        rng = np.random.default_rng(0)
        cloud1 = rng.normal(size=(15, 2))
        cloud2 = rng.normal(size=(15, 2))
        with pytest.raises(ValueError, match="Unknown distance metric"):
            _cloud_overlap_neighbors(cloud1, cloud2, k=3, distance_metric="bad")


# ---------------------------------------------------------------------------
# _cloud_overlap_radius – geodesic path
# ---------------------------------------------------------------------------


class TestCloudOverlapRadiusGeodesic:
    """Cover lines 502-507 (geodesic path in _cloud_overlap_radius)."""

    def test_geodesic_radius_overlap(self) -> None:
        rng = np.random.default_rng(0)
        cloud1 = rng.normal(size=(20, 2))
        cloud2 = rng.normal(size=(20, 2)) + 1.5

        ov12, ov21 = _cloud_overlap_radius(
            cloud1, cloud2, r=2.0, distance_metric="geodesic"
        )
        assert isinstance(ov12, float)
        assert isinstance(ov21, float)

    def test_invalid_metric_radius(self) -> None:
        rng = np.random.default_rng(0)
        cloud1 = rng.normal(size=(15, 2))
        cloud2 = rng.normal(size=(15, 2))
        with pytest.raises(ValueError, match="Unknown distance metric"):
            _cloud_overlap_radius(cloud1, cloud2, r=1.0, distance_metric="bad")


# ---------------------------------------------------------------------------
# compute_structure_index – uncovered branches
# ---------------------------------------------------------------------------


class TestComputeStructureIndexCoverage:
    """Cover remaining branches in compute_structure_index."""

    def test_1d_label_is_reshaped(self) -> None:
        """Cover line 631: label.ndim==1 → reshape(-1,1)."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(80, 4))
        label_1d = rng.uniform(size=(80,))  # 1D label

        si, _, overlap_mat, _ = compute_structure_index(
            data, label_1d, n_bins=4, n_neighbors=5, num_shuffles=0
        )
        assert isinstance(si, (float, np.floating))

    def test_data_transposed_when_more_features_than_samples(self) -> None:
        """Cover lines 635-643: data auto-transposition via logger.warning.

        When data has more features than samples, the code transposes it.
        To make the function succeed after transposition: data shape (n_feat, n_samples)
        is transposed to (n_samples, n_feat), so label must have n_samples rows.
        """
        rng = np.random.default_rng(0)
        # data shape (4, 80): 4 'samples', 80 'features' → features > samples
        # After transpose: (80, 4) → 80 samples, 4 features
        # label must match 80 rows (the actual samples before transposition)
        data = rng.normal(size=(4, 80))
        label = rng.uniform(size=(80, 1))  # 80 rows = true sample count

        si, _, _, _ = compute_structure_index(
            data, label, n_bins=4, n_neighbors=5, num_shuffles=0
        )
        assert isinstance(si, (float, np.floating))

    def test_n_bins_list_path(self) -> None:
        """Cover the list branch of n_bins processing (line 652→656)."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(100, 4))
        label = rng.uniform(size=(100, 2))

        si, _, overlap_mat, _ = compute_structure_index(
            data, label, n_bins=[5, 4], n_neighbors=5, num_shuffles=0
        )
        assert isinstance(si, (float, np.floating))

    def test_discrete_label_list_all_bool(self) -> None:
        """Cover line 687: discrete_label as list of bool."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(80, 4))
        label = rng.uniform(size=(80, 1))

        si, _, _, _ = compute_structure_index(
            data, label, n_bins=4, n_neighbors=5, num_shuffles=0,
            discrete_label=[False]
        )
        assert isinstance(si, (float, np.floating))

    def test_verbose_printing(self) -> None:
        """Cover verbose print branches (lines 705, 716, 778, 789, 808, 816, etc.)."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(80, 4))
        label = rng.uniform(size=(80, 1))

        # Small num_shuffles so it finishes fast
        si, _, _, shuf_si = compute_structure_index(
            data, label, n_bins=4, n_neighbors=5, num_shuffles=3, verbose=True, seed=0
        )
        assert isinstance(si, (float, np.floating))
        assert len(shuf_si) == 3

    def test_num_shuffles_zero_early_return(self) -> None:
        """Cover line 821: return when num_shuffles==0."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(80, 4))
        label = rng.uniform(size=(80, 1))

        si, bin_info, overlap_mat, shuf_si = compute_structure_index(
            data, label, n_bins=4, n_neighbors=5, num_shuffles=0
        )
        assert isinstance(si, (float, np.floating))
        assert isinstance(shuf_si, np.ndarray)
        assert len(shuf_si) == 0

    def test_filter_noise_path(self) -> None:
        """Cover lines 754-757: filter_noise kwarg."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(120, 4))
        label = rng.uniform(size=(120, 1))

        si, _, _, _ = compute_structure_index(
            data, label, n_bins=4, n_neighbors=5, num_shuffles=0, filter_noise=True
        )
        assert isinstance(si, (float, np.floating))

    def test_few_unique_labels_warning(self) -> None:
        """Cover lines 723-729: warning when n_bins >= num_unique values."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(60, 3))
        # Only 3 unique label values, but n_bins=10 > 3
        label = np.array([0.0, 1.0, 2.0] * 20).reshape(-1, 1)

        with pytest.warns(UserWarning, match="fewer unique values"):
            si, _, _, _ = compute_structure_index(
                data, label, n_bins=10, n_neighbors=5, num_shuffles=0
            )
        assert isinstance(si, (float, np.floating))

    def test_seeded_shuffle(self) -> None:
        """Cover seed kwarg for reproducibility in shuffling (line 828)."""
        rng = np.random.default_rng(1)
        data = rng.normal(size=(80, 4))
        label = rng.uniform(size=(80, 1))

        si1, _, _, shuf1 = compute_structure_index(
            data, label, n_bins=4, n_neighbors=5, num_shuffles=5, seed=42
        )
        si2, _, _, shuf2 = compute_structure_index(
            data, label, n_bins=4, n_neighbors=5, num_shuffles=5, seed=42
        )
        np.testing.assert_array_equal(shuf1, shuf2)

    def test_invalid_distance_metric(self) -> None:
        """Cover assertion for invalid distance_metric."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(80, 4))
        label = rng.uniform(size=(80, 1))

        with pytest.raises(AssertionError):
            compute_structure_index(
                data, label, n_bins=4, n_neighbors=5, distance_metric="bad"
            )

    def test_n_bins_too_small(self) -> None:
        """Cover assertion for n_bins <= 1."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(80, 4))
        label = rng.uniform(size=(80, 1))

        with pytest.raises(AssertionError):
            compute_structure_index(
                data, label, n_bins=1, n_neighbors=5
            )

    def test_n_neighbors_too_small(self) -> None:
        """Cover assertion for n_neighbors <= 2."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(80, 4))
        label = rng.uniform(size=(80, 1))

        with pytest.raises(AssertionError):
            compute_structure_index(
                data, label, n_bins=4, n_neighbors=2
            )

    def test_type_error_on_wrong_data_type(self) -> None:
        """validate_args_types decorator raises TypeError for wrong data type."""
        with pytest.raises(TypeError):
            compute_structure_index(
                [[1, 2], [3, 4]],  # type: ignore[arg-type]  # list, not ndarray
                np.array([[0.1], [0.2]]),
                n_bins=2,
                n_neighbors=3,
            )

    def test_radius_based_with_config(self) -> None:
        """Cover radius path using a StructureIndexConfig."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(80, 4))
        label = rng.uniform(size=(80, 1))

        cfg = StructureIndexConfig(radius=0.5, num_shuffles=0, verbose=False)
        si, _, _, _ = compute_structure_index(data, label, n_bins=4, config=cfg)
        assert isinstance(si, (float, np.floating))

    def test_num_shuffles_zero_assertion(self) -> None:
        """num_shuffles < 0 raises AssertionError."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(60, 3))
        label = rng.uniform(size=(60, 1))

        with pytest.raises(AssertionError):
            compute_structure_index(data, label, n_bins=4, n_neighbors=5, num_shuffles=-1)

    def test_min_max_label_kwargs(self) -> None:
        """Cover lines 733-740 with custom min_label/max_label kwarg."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(100, 3))
        label = rng.uniform(0, 1, size=(100, 1))

        si, _, _, _ = compute_structure_index(
            data, label, n_bins=4, n_neighbors=5, num_shuffles=0,
            min_label=[0.1], max_label=[0.9]
        )
        assert isinstance(si, (float, np.floating))


# ---------------------------------------------------------------------------
# draw_overlap_graph – additional branches
# ---------------------------------------------------------------------------


class TestDrawOverlapGraphCoverage:
    """Cover remaining branches in draw_overlap_graph."""

    def teardown_method(self) -> None:
        plt.close("all")

    def test_with_numeric_node_names(self) -> None:
        """Cover line 929: node_names not str → node_val = node_names."""
        overlap_mat = np.array([[0.0, 0.3, 0.1],
                                [0.2, 0.0, 0.4],
                                [0.15, 0.35, 0.0]])
        fig, ax = plt.subplots()
        # Numeric names: isinstance(node_names[0], str) == False → node_val = node_names
        draw_overlap_graph(overlap_mat, ax=ax, node_names=[10, 20, 30])
        plt.close(fig)

    def test_with_custom_node_color(self) -> None:
        """Cover the branch where node_color is provided (skip norm_cmap computation)."""
        overlap_mat = np.array([[0.0, 0.2], [0.3, 0.0]])
        colors = ["red", "blue"]
        fig, ax = plt.subplots()
        draw_overlap_graph(overlap_mat, ax=ax, node_color=colors)
        plt.close(fig)

    def test_with_string_node_names(self) -> None:
        """node_names as strings → node_val = range(number_nodes)."""
        overlap_mat = np.array([[0.0, 0.2, 0.1],
                                [0.3, 0.0, 0.15],
                                [0.05, 0.25, 0.0]])
        fig, ax = plt.subplots()
        draw_overlap_graph(overlap_mat, ax=ax, node_names=["A", "B", "C"])
        plt.close(fig)

    def test_custom_layout(self) -> None:
        """Cover layout_type kwarg."""
        import networkx as nx
        overlap_mat = np.array([[0.0, 0.5], [0.6, 0.0]])
        fig, ax = plt.subplots()
        draw_overlap_graph(overlap_mat, ax=ax, layout_type=nx.spring_layout)
        plt.close(fig)


# ---------------------------------------------------------------------------
# _array_to_key
# ---------------------------------------------------------------------------


class TestArrayToKey:
    """Test _array_to_key function."""

    def test_small_array(self) -> None:
        """Small arrays (<=10) use actual indices."""
        arr = np.array([1, 2, 3])
        key = _array_to_key(arr)
        assert key == "1_2_3"

    def test_large_array_uses_hash(self) -> None:
        """Cover lines 970-972: large arrays (>10) use hash."""
        arr = np.arange(11, dtype=np.int_)
        key = _array_to_key(arr)
        assert key.startswith("hash_")

    def test_exactly_ten_elements(self) -> None:
        """Boundary: 10 elements → uses actual indices."""
        arr = np.arange(10, dtype=np.int_)
        key = _array_to_key(arr)
        assert "_" in key
        assert not key.startswith("hash_")


# ---------------------------------------------------------------------------
# _structure_index_cache_key
# ---------------------------------------------------------------------------


class TestStructureIndexCacheKey:
    """Test _structure_index_cache_key helper."""

    def test_format(self) -> None:
        key = _structure_index_cache_key("ds1", 10, 15, "all")
        assert "structure_index" in key
        assert "ds1" in key
        assert "n_bins=10" in key
        assert "n_neighbors=15" in key
        assert "indices=all" in key


# ---------------------------------------------------------------------------
# _parse_loaded_results
# ---------------------------------------------------------------------------


class TestParseLoadedResults:
    """Test _parse_loaded_results parsing function."""

    def test_missing_dataset_name(self) -> None:
        """If dataset_name not in loaded_data, return empty dict."""
        result = _parse_loaded_results({}, "missing_ds")
        assert result == {}

    def test_skips_entries_without_bins_or_neighbors(self) -> None:
        """Entries without n_bins or n_neighbors are skipped."""
        loaded_data = {
            "ds": {
                "item1": {
                    "attributes": {"SI": 0.5},  # missing n_bins and n_neighbors
                    "arrays": {},
                }
            }
        }
        result = _parse_loaded_results(loaded_data, "ds")
        assert result == {}

    def test_valid_entry_parsed(self) -> None:
        """A valid entry is parsed into the correct structure."""
        overlap = np.zeros((3, 3))
        shuf = np.array([0.1, 0.2])
        loaded_data = {
            "ds": {
                "item1": {
                    "attributes": {"n_bins": 5, "n_neighbors": 10, "SI": 0.7},
                    "arrays": {"overlap_mat": overlap, "shuf_SI": shuf},
                }
            }
        }
        result = _parse_loaded_results(loaded_data, "ds")
        assert (5, 10) in result
        assert result[(5, 10)]["SI"] == 0.7
        np.testing.assert_array_equal(result[(5, 10)]["overlap_mat"], overlap)


# ---------------------------------------------------------------------------
# load_structure_index_results
# ---------------------------------------------------------------------------


class TestLoadStructureIndexResults:
    """Cover load_structure_index_results branches."""

    def test_nonexistent_file_returns_empty(self, tmp_path: Path) -> None:
        """Cover lines 1339-1341: file does not exist → return {}."""
        result = load_structure_index_results(
            save_path=tmp_path / "missing.h5",
            dataset_name="test",
        )
        assert result == {}

    def test_load_with_n_bins_filter(self, tmp_path: Path) -> None:
        """Cover lines 1345-1350: filter_attrs built from n_bins and n_neighbors."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(80, 4))
        labels = rng.uniform(size=(80, 1))
        save_path = tmp_path / "si.h5"

        compute_structure_index_sweep(
            data=data,
            labels=labels,
            dataset_name="ds",
            save_path=save_path,
            n_neighbors_list=[5, 8],
            n_bins_list=[4, 6],
            num_shuffles=0,
            verbose=False,
        )

        # Filter by n_bins=4
        results = load_structure_index_results(
            save_path=save_path,
            dataset_name="ds",
            n_bins=4,
        )
        for key in results:
            assert key[0] == 4

    def test_load_with_indices_key_filter(self, tmp_path: Path) -> None:
        """Cover indices_key filter path (line 1350)."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(80, 4))
        labels = rng.uniform(size=(80, 1))
        save_path = tmp_path / "si2.h5"

        compute_structure_index_sweep(
            data=data,
            labels=labels,
            dataset_name="ds",
            save_path=save_path,
            n_neighbors_list=[5],
            n_bins_list=[4],
            num_shuffles=0,
            verbose=False,
        )

        results = load_structure_index_results(
            save_path=save_path,
            dataset_name="ds",
            indices_key="all",
        )
        assert isinstance(results, dict)


# ---------------------------------------------------------------------------
# compute_structure_index_sweep – additional paths
# ---------------------------------------------------------------------------


class TestComputeStructureIndexSweepCoverage:
    """Cover additional branches in compute_structure_index_sweep."""

    def test_with_data_indices(self, tmp_path: Path) -> None:
        """Cover lines 1135-1138: data_indices subset path."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(100, 4))
        labels = rng.uniform(size=(100, 1))
        save_path = tmp_path / "si_idx.h5"
        indices = np.arange(60, dtype=np.int_)

        results = compute_structure_index_sweep(
            data=data,
            labels=labels,
            dataset_name="ds_idx",
            save_path=save_path,
            n_neighbors_list=[5],
            n_bins_list=[4],
            data_indices=indices,
            num_shuffles=0,
            verbose=False,
        )
        assert (4, 5) in results

    def test_large_data_indices_key(self, tmp_path: Path) -> None:
        """data_indices with >10 elements → hash key (covers _array_to_key hash path)."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(100, 4))
        labels = rng.uniform(size=(100, 1))
        save_path = tmp_path / "si_hash.h5"
        indices = np.arange(80, dtype=np.int_)  # 80 > 10 → hash key

        results = compute_structure_index_sweep(
            data=data,
            labels=labels,
            dataset_name="ds_hash",
            save_path=save_path,
            n_neighbors_list=[5],
            n_bins_list=[4],
            data_indices=indices,
            num_shuffles=0,
            verbose=False,
        )
        assert (4, 5) in results

    def test_regenerate_true(self, tmp_path: Path) -> None:
        """Cover lines 1172->1191: regenerate=True skips cache check."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(80, 4))
        labels = rng.uniform(size=(80, 1))
        save_path = tmp_path / "si_regen.h5"

        # First call
        compute_structure_index_sweep(
            data=data, labels=labels, dataset_name="ds_r",
            save_path=save_path, n_neighbors_list=[5], n_bins_list=[4],
            num_shuffles=0, verbose=False,
        )
        # Second call with regenerate=True forces recomputation
        results = compute_structure_index_sweep(
            data=data, labels=labels, dataset_name="ds_r",
            save_path=save_path, n_neighbors_list=[5], n_bins_list=[4],
            num_shuffles=0, verbose=False, regenerate=True,
        )
        assert (4, 5) in results
        assert isinstance(results[(4, 5)]["SI"], (float, np.floating))

    def test_default_save_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cover line 1116: default save_path = './output/structure_indices.h5'."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(80, 4))
        labels = rng.uniform(size=(80, 1))

        # Change working directory to tmp_path so default path goes there
        monkeypatch.chdir(tmp_path)

        results = compute_structure_index_sweep(
            data=data,
            labels=labels,
            dataset_name="ds_def",
            save_path=None,  # triggers default path
            n_neighbors_list=[5],
            n_bins_list=[4],
            num_shuffles=0,
            verbose=False,
        )
        assert (4, 5) in results
        assert (tmp_path / "output" / "structure_indices.h5").exists()

    def test_cache_hit_path(self, tmp_path: Path) -> None:
        """Cover cache hit: second call with same params returns from existing_results."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(80, 4))
        labels = rng.uniform(size=(80, 1))
        save_path = tmp_path / "si_cache.h5"

        r1 = compute_structure_index_sweep(
            data=data, labels=labels, dataset_name="ds_c",
            save_path=save_path, n_neighbors_list=[5], n_bins_list=[4],
            num_shuffles=0, verbose=False,
        )
        r2 = compute_structure_index_sweep(
            data=data, labels=labels, dataset_name="ds_c",
            save_path=save_path, n_neighbors_list=[5], n_bins_list=[4],
            num_shuffles=0, verbose=False, regenerate=False,
        )
        assert r1[(4, 5)]["SI"] == pytest.approx(r2[(4, 5)]["SI"])

    def test_verbose_sweep(self, tmp_path: Path) -> None:
        """Cover verbose=True path in sweep (tqdm enabled)."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(80, 4))
        labels = rng.uniform(size=(80, 1))
        save_path = tmp_path / "si_verbose.h5"

        results = compute_structure_index_sweep(
            data=data, labels=labels, dataset_name="ds_v",
            save_path=save_path, n_neighbors_list=[5], n_bins_list=[4],
            num_shuffles=0, verbose=True,
        )
        assert (4, 5) in results

    def test_storage_manager_cache_hit_path(
        self, tmp_path: Path
    ) -> None:
        """Cover storage_manager cache_get returning non-None (lines 1173-1178).

        We inject a fake StorageManager into sys.modules so the function-local
        `from neural_analysis.utils.storage.manager import StorageManager`
        picks up our fake, which returns pre-cached results from cache_get.
        Exercises: 1122->1129, 1148->1157, 1151->1149, 1173->1179,
        1176-1178, 1246->1249.
        """
        import sys
        import types

        rng = np.random.default_rng(0)
        data = rng.normal(size=(80, 4))
        labels = rng.uniform(size=(80, 1))
        save_path = tmp_path / "si_sm.h5"

        # First run to populate the HDF5 file (no fake SM)
        first_results = compute_structure_index_sweep(
            data=data, labels=labels, dataset_name="ds_sm",
            save_path=save_path, n_neighbors_list=[5], n_bins_list=[4],
            num_shuffles=0, verbose=False,
        )
        cached_item = first_results[(4, 5)]

        # Build fake SM that pre-caches the result so cache_get returns it
        _cache: dict = {}

        class FakeSM:
            def cache_set(self, key: str, value: object) -> None:
                _cache[key] = value

            def cache_get(self, key: str) -> object:
                return _cache.get(key)

            # io.py also calls these on the storage_manager
            def index_comparison(self, *args: object, **kwargs: object) -> None:
                pass

            def save_data(self, *args: object, **kwargs: object) -> None:
                pass

        fake_sm_instance = FakeSM()
        # Pre-populate so cache_get returns immediately
        _cache["structure_index::ds_sm2::n_bins=4::n_neighbors=5::indices=all"] = cached_item

        class FakeSMCls:
            def __new__(cls) -> "FakeSM":  # type: ignore[misc]
                return fake_sm_instance

        fake_module = types.ModuleType("neural_analysis.utils.storage.manager")
        fake_module.StorageManager = FakeSMCls  # type: ignore[attr-defined]
        original_module = sys.modules.get("neural_analysis.utils.storage.manager")
        sys.modules["neural_analysis.utils.storage.manager"] = fake_module

        try:
            results = compute_structure_index_sweep(
                data=data, labels=labels, dataset_name="ds_sm2",
                save_path=save_path, n_neighbors_list=[5], n_bins_list=[4],
                num_shuffles=0, verbose=False, regenerate=False,
            )
        finally:
            if original_module is not None:
                sys.modules["neural_analysis.utils.storage.manager"] = original_module
            elif "neural_analysis.utils.storage.manager" in sys.modules:
                del sys.modules["neural_analysis.utils.storage.manager"]

        assert (4, 5) in results

    def test_storage_manager_cache_set_after_compute(
        self, tmp_path: Path
    ) -> None:
        """Cover storage_manager.cache_set after new computation (lines 1246-1249)
        and pre-loading of existing results into storage_manager (lines 1148-1157)."""
        import sys
        import types

        rng = np.random.default_rng(0)
        data = rng.normal(size=(80, 4))
        labels = rng.uniform(size=(80, 1))
        save_path = tmp_path / "si_sm3.h5"

        # First run to populate the HDF5 file (provides existing_results)
        compute_structure_index_sweep(
            data=data, labels=labels, dataset_name="ds_sm3",
            save_path=save_path, n_neighbors_list=[5], n_bins_list=[4],
            num_shuffles=0, verbose=False,
        )

        _cache: dict = {}

        class FakeSM:
            def cache_set(self, key: str, value: object) -> None:
                _cache[key] = value

            def cache_get(self, key: str) -> object:
                return None  # Force recomputation (but SM is present for cache_set)

            def index_comparison(self, *args: object, **kwargs: object) -> None:
                pass

            def save_data(self, *args: object, **kwargs: object) -> None:
                pass

        class FakeSMCls:
            def __new__(cls) -> "FakeSM":  # type: ignore[misc]
                return FakeSM()

        fake_module = types.ModuleType("neural_analysis.utils.storage.manager")
        fake_module.StorageManager = FakeSMCls  # type: ignore[attr-defined]
        original_module = sys.modules.get("neural_analysis.utils.storage.manager")
        sys.modules["neural_analysis.utils.storage.manager"] = fake_module

        try:
            results = compute_structure_index_sweep(
                data=data, labels=labels, dataset_name="ds_sm3",
                save_path=save_path, n_neighbors_list=[5], n_bins_list=[4],
                num_shuffles=0, verbose=False, regenerate=True,
            )
        finally:
            if original_module is not None:
                sys.modules["neural_analysis.utils.storage.manager"] = original_module
            elif "neural_analysis.utils.storage.manager" in sys.modules:
                del sys.modules["neural_analysis.utils.storage.manager"]

        assert (4, 5) in results

    def test_storage_manager_init_exception(
        self, tmp_path: Path
    ) -> None:
        """Cover lines 1125-1126: StorageManager() raises → storage_manager=None."""
        import sys
        import types

        rng = np.random.default_rng(0)
        data = rng.normal(size=(80, 4))
        labels = rng.uniform(size=(80, 1))
        save_path = tmp_path / "si_sm4.h5"

        class BrokenSMCls:
            def __new__(cls) -> "BrokenSMCls":  # type: ignore[misc]
                raise RuntimeError("Simulated init failure")

        fake_module = types.ModuleType("neural_analysis.utils.storage.manager")
        fake_module.StorageManager = BrokenSMCls  # type: ignore[attr-defined]
        original_module = sys.modules.get("neural_analysis.utils.storage.manager")
        sys.modules["neural_analysis.utils.storage.manager"] = fake_module

        try:
            results = compute_structure_index_sweep(
                data=data, labels=labels, dataset_name="ds_sm4",
                save_path=save_path, n_neighbors_list=[5], n_bins_list=[4],
                num_shuffles=0, verbose=False,
            )
        finally:
            if original_module is not None:
                sys.modules["neural_analysis.utils.storage.manager"] = original_module
            elif "neural_analysis.utils.storage.manager" in sys.modules:
                del sys.modules["neural_analysis.utils.storage.manager"]

        assert (4, 5) in results
