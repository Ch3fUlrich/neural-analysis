"""Tests for structure index computation module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from neural_analysis.topology.structure_index import (
    _cloud_overlap_neighbors,
    _cloud_overlap_radius,
    _create_ndim_grid,
    _filter_noisy_outliers,
    _meshgrid2,
    compute_structure_index,
    draw_overlap_graph,
)
from neural_analysis.topology import (
    compute_structure_index_sweep,
    load_structure_index_results,
)


class TestHelperFunctions:
    """Test suite for helper functions."""

    def test_filter_noisy_outliers(self):
        """Test outlier filtering."""
        # Create data with clear outliers
        np.random.seed(42)
        data = np.random.randn(100, 3)
        # Add outliers
        data[0, :] = [10, 10, 10]
        data[1, :] = [-10, -10, -10]

        noise_idx = _filter_noisy_outliers(data)

        assert isinstance(noise_idx, np.ndarray)
        assert noise_idx.dtype == np.int64 or noise_idx.dtype == np.intp
        assert len(noise_idx) <= len(data)

    def test_meshgrid2(self):
        """Test meshgrid creation."""
        arrs = (np.array([1, 2]), np.array([3, 4, 5]))
        result = _meshgrid2(arrs)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].shape == (2, 3)
        assert result[1].shape == (2, 3)

    def test_create_ndim_grid_1d(self):
        """Test 1D grid creation."""
        label = np.linspace(0, 1, 100).reshape(-1, 1)
        n_bins = [10]
        min_label = [0.0]
        max_label = [1.0]
        discrete_label = [False]

        grid, coords = _create_ndim_grid(
            label, n_bins, min_label, max_label, discrete_label
        )

        assert len(grid) == 10
        assert coords.shape[0] == 10
        assert coords.shape[1] == 1
        assert coords.shape[2] == 3

    def test_create_ndim_grid_2d(self):
        """Test 2D grid creation."""
        np.random.seed(42)
        label = np.random.rand(100, 2)
        n_bins = [5, 4]
        min_label = [0.0, 0.0]
        max_label = [1.0, 1.0]
        discrete_label = [False, False]

        grid, coords = _create_ndim_grid(
            label, n_bins, min_label, max_label, discrete_label
        )

        assert len(grid) == 20  # 5 * 4
        assert coords.shape[0] == 20
        assert coords.shape[1] == 2

    def test_create_ndim_grid_discrete(self):
        """Test grid creation with discrete labels."""
        label = np.array([[0], [1], [0], [2], [1], [2]])
        n_bins = [3]
        min_label = [0]
        max_label = [2]
        discrete_label = [True]

        grid, coords = _create_ndim_grid(
            label, n_bins, min_label, max_label, discrete_label
        )

        assert len(grid) == 3
        # Check that each discrete value has its points
        assert len(grid[0]) == 2  # Two 0s
        assert len(grid[1]) == 2  # Two 1s
        assert len(grid[2]) == 2  # Two 2s


class TestCloudOverlap:
    """Test suite for cloud overlap functions."""

    def test_cloud_overlap_neighbors_euclidean(self):
        """Test neighbor-based overlap with Euclidean distance."""
        np.random.seed(42)
        cloud1 = np.random.randn(50, 3)
        cloud2 = np.random.randn(50, 3) + 2.0  # Shifted cloud

        overlap_1_2, overlap_2_1 = _cloud_overlap_neighbors(
            cloud1, cloud2, k=10, distance_metric="euclidean"
        )

        assert isinstance(overlap_1_2, float)
        assert isinstance(overlap_2_1, float)
        assert 0 <= overlap_1_2 <= 1
        assert 0 <= overlap_2_1 <= 1

    def test_cloud_overlap_neighbors_large_k(self):
        """Test overlap with k larger than total points."""
        np.random.seed(42)
        cloud1 = np.random.randn(10, 2)
        cloud2 = np.random.randn(10, 2)

        overlap_1_2, overlap_2_1 = _cloud_overlap_neighbors(
            cloud1, cloud2, k=30, distance_metric="euclidean"
        )

        # Should handle gracefully
        assert isinstance(overlap_1_2, float)
        assert isinstance(overlap_2_1, float)

    def test_cloud_overlap_radius_euclidean(self):
        """Test radius-based overlap with Euclidean distance."""
        np.random.seed(42)
        cloud1 = np.random.randn(50, 3)
        cloud2 = np.random.randn(50, 3) + 1.0

        overlap_1_2, overlap_2_1 = _cloud_overlap_radius(
            cloud1, cloud2, r=1.0, distance_metric="euclidean"
        )

        assert isinstance(overlap_1_2, float)
        assert isinstance(overlap_2_1, float)
        assert 0 <= overlap_1_2 <= 1
        assert 0 <= overlap_2_1 <= 1

    def test_cloud_overlap_invalid_metric(self):
        """Test that invalid distance metric raises error."""
        np.random.seed(42)
        cloud1 = np.random.randn(20, 2)
        cloud2 = np.random.randn(20, 2)

        with pytest.raises(ValueError, match="Unknown distance metric"):
            _cloud_overlap_neighbors(
                cloud1, cloud2, k=5, distance_metric="invalid_metric"
            )


class TestComputeStructureIndex:
    """Test suite for main structure index computation."""

    def test_compute_structure_index_basic(self):
        """Test basic structure index computation."""
        np.random.seed(42)
        # Create structured data
        n_samples = 200
        theta = np.linspace(0, 2 * np.pi, n_samples)
        data = np.column_stack(
            [np.cos(theta), np.sin(theta), np.random.randn(n_samples)]
        )
        label = theta.reshape(-1, 1)

        si, bin_info, overlap_mat, shuf_si = compute_structure_index(
            data, label, n_bins=10, n_neighbors=15, num_shuffles=10, verbose=False
        )

        assert isinstance(si, (float, np.floating))
        assert 0 <= si <= 1
        assert isinstance(bin_info, tuple)
        assert len(bin_info) == 2
        assert isinstance(overlap_mat, np.ndarray)
        assert overlap_mat.ndim == 2
        assert isinstance(shuf_si, np.ndarray)
        assert len(shuf_si) == 10

    def test_compute_structure_index_2d_labels(self):
        """Test with 2D labels."""
        np.random.seed(42)
        data = np.random.randn(150, 5)
        label = np.random.randn(150, 2)

        si, _, overlap_mat, _ = compute_structure_index(
            data, label, n_bins=[5, 4], n_neighbors=10, num_shuffles=5, verbose=False
        )

        assert isinstance(si, (float, np.floating))
        assert isinstance(overlap_mat, np.ndarray)

    def test_compute_structure_index_discrete_labels(self):
        """Test with discrete labels."""
        np.random.seed(42)
        data = np.random.randn(100, 3)
        label = np.random.choice([0, 1, 2], size=(100, 1))

        si, _, overlap_mat, _ = compute_structure_index(
            data,
            label,
            n_bins=3,
            discrete_label=True,
            n_neighbors=10,
            num_shuffles=5,
            verbose=False,
        )

        assert isinstance(si, (float, np.floating))
        assert overlap_mat.shape[0] <= 3  # May be fewer if bins are filtered

    def test_compute_structure_index_with_nans(self):
        """Test handling of NaN values."""
        np.random.seed(42)
        data = np.random.randn(100, 3)
        label = np.random.randn(100, 1)
        # Add some NaNs
        data[0, 0] = np.nan
        label[1, 0] = np.nan

        si, _, _, _ = compute_structure_index(
            data, label, n_bins=5, n_neighbors=10, num_shuffles=0, verbose=False
        )

        assert isinstance(si, (float, np.floating))
        # Should handle NaNs gracefully

    def test_compute_structure_index_radius(self):
        """Test structure index with radius-based overlap."""
        np.random.seed(42)
        data = np.random.randn(100, 3)
        label = np.random.randn(100, 1)

        si, _, _, _ = compute_structure_index(
            data, label, n_bins=5, radius=1.0, num_shuffles=0, verbose=False
        )

        assert isinstance(si, (float, np.floating))

    def test_compute_structure_index_invalid_both_k_and_r(self):
        """Test that specifying both n_neighbors and radius raises error."""
        np.random.seed(42)
        data = np.random.randn(50, 2)
        label = np.random.randn(50, 1)

        with pytest.raises(ValueError, match="Specify either n_neighbors or radius"):
            compute_structure_index(
                data, label, n_bins=3, n_neighbors=10, radius=1.0, verbose=False
            )

    def test_compute_structure_index_few_bins(self):
        """Test with very few unique labels."""
        np.random.seed(42)
        data = np.random.randn(100, 3)
        label = np.zeros((100, 1))  # All same label

        si, _, overlap_mat, _ = compute_structure_index(
            data, label, n_bins=5, n_neighbors=10, num_shuffles=0, verbose=False
        )

        # Should return NaN when only one bin
        assert np.isnan(si)
        assert np.isnan(overlap_mat).all()

    def test_compute_structure_index_dims_subset(self):
        """Test computing SI on subset of dimensions."""
        np.random.seed(42)
        data = np.random.randn(100, 5)
        label = np.random.randn(100, 1)

        si, _, _, _ = compute_structure_index(
            data,
            label,
            n_bins=5,
            dims=[0, 2, 4],  # Use only 3 out of 5 dimensions
            n_neighbors=10,
            num_shuffles=0,
            verbose=False,
        )

        assert isinstance(si, (float, np.floating))


class TestSweepFunctionality:
    """Test suite for parameter sweep and batch processing functionality."""

    def test_compute_structure_index_sweep_basic(self):
        """Test basic parameter sweep."""
        np.random.seed(42)
        data = np.random.randn(100, 3)
        labels = np.random.randn(100, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "si_sweep.h5"

            results = compute_structure_index_sweep(
                data=data,
                labels=labels,
                dataset_name="test_session",
                save_path=save_path,
                n_neighbors_list=[10, 15],
                n_bins_list=[5, 8],
                num_shuffles=5,
                verbose=False,
            )

            # Check results structure
            assert len(results) == 4  # 2 x 2 combinations
            assert (5, 10) in results
            assert (8, 15) in results

            # Check result content
            for key, result in results.items():
                assert "SI" in result
                assert "overlap_mat" in result
                assert "metadata" in result
                assert isinstance(result["SI"], (float, np.floating))

            # Verify file was created
            assert save_path.exists()

    def test_compute_structure_index_sweep_caching(self):
        """Test that sweep uses cached results."""
        np.random.seed(42)
        data = np.random.randn(100, 3)
        labels = np.random.randn(100, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "si_sweep.h5"

            # First computation
            results1 = compute_structure_index_sweep(
                data=data,
                labels=labels,
                dataset_name="test_session",
                save_path=save_path,
                n_neighbors_list=[10],
                n_bins_list=[5],
                num_shuffles=0,
                verbose=False,
            )

            si_first = results1[(5, 10)]["SI"]

            # Second computation (should load from cache)
            results2 = compute_structure_index_sweep(
                data=data,
                labels=labels,
                dataset_name="test_session",
                save_path=save_path,
                n_neighbors_list=[10],
                n_bins_list=[5],
                num_shuffles=0,
                verbose=False,
                regenerate=False,
            )

            si_second = results2[(5, 10)]["SI"]

            # Should get same result from cache
            assert si_first == pytest.approx(si_second)

    def test_load_structure_index_results(self):
        """Test loading saved results."""
        np.random.seed(42)
        data = np.random.randn(100, 3)
        labels = np.random.randn(100, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "si_sweep.h5"

            # Compute and save
            original_results = compute_structure_index_sweep(
                data=data,
                labels=labels,
                dataset_name="test_session",
                save_path=save_path,
                n_neighbors_list=[10, 15],
                n_bins_list=[5],
                num_shuffles=5,
                verbose=False,
            )

            # Load all results
            loaded_results = load_structure_index_results(
                save_path=save_path,
                dataset_name="test_session",
            )

            assert len(loaded_results) == 2
            assert (5, 10) in loaded_results
            assert (5, 15) in loaded_results

            # Check values match
            for key in loaded_results:
                assert loaded_results[key]["SI"] == pytest.approx(
                    original_results[key]["SI"]
                )

    def test_load_structure_index_results_filtered(self):
        """Test loading with filters."""
        np.random.seed(42)
        data = np.random.randn(100, 3)
        labels = np.random.randn(100, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "si_sweep.h5"

            # Compute and save multiple combinations
            compute_structure_index_sweep(
                data=data,
                labels=labels,
                dataset_name="test_session",
                save_path=save_path,
                n_neighbors_list=[10, 15, 20],
                n_bins_list=[5, 8],
                num_shuffles=0,
                verbose=False,
            )

            # Load with filter
            filtered_results = load_structure_index_results(
                save_path=save_path,
                dataset_name="test_session",
                n_bins=5,
            )

            # Should only get results with n_bins=5
            assert len(filtered_results) == 3  # 3 n_neighbors values
            for key in filtered_results:
                assert key[0] == 5  # n_bins is first in tuple


class TestDrawOverlapGraph:
    """Test suite for graph drawing function."""

    def test_draw_overlap_graph_basic(self):
        """Test basic graph drawing."""
        import matplotlib.pyplot as plt

        overlap_mat = np.array([[0, 0.2, 0.3], [0.25, 0, 0.4], [0.35, 0.45, 0]])

        fig, ax = plt.subplots()
        # draw_overlap_graph returns None (nx.draw_networkx doesn't return anything)
        # Just check it runs without error
        draw_overlap_graph(overlap_mat, ax=ax)

        # Check that the axes has some content (artists were added)
        assert len(ax.collections) > 0 or len(ax.patches) > 0
        plt.close(fig)

    def test_draw_overlap_graph_with_names(self):
        """Test graph drawing with node names."""
        import matplotlib.pyplot as plt

        overlap_mat = np.random.rand(4, 4)
        np.fill_diagonal(overlap_mat, 0)

        fig, ax = plt.subplots()
        draw_overlap_graph(overlap_mat, ax=ax, node_names=["A", "B", "C", "D"])

        # Check that the axes has some content (artists were added)
        assert len(ax.collections) > 0 or len(ax.patches) > 0
        plt.close(fig)

    def test_draw_overlap_graph_custom_params(self):
        """Test graph drawing with custom parameters."""
        import matplotlib.pyplot as plt

        overlap_mat = np.random.rand(3, 3)

        fig, ax = plt.subplots()
        draw_overlap_graph(
            overlap_mat,
            ax=ax,
            node_size=500,
            scale_edges=10,
            arrow_size=15,
            edge_vmin=0.1,
            edge_vmax=0.8,
        )

        # Check that the axes has some content (artists were added)
        assert len(ax.collections) > 0 or len(ax.patches) > 0
        plt.close(fig)


class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_small_sample_size(self):
        """Test with very small sample size."""
        np.random.seed(42)
        data = np.random.randn(20, 2)
        label = np.random.randn(20, 1)

        si, _, _, _ = compute_structure_index(
            data, label, n_bins=3, n_neighbors=5, num_shuffles=0, verbose=False
        )

        # Should handle small datasets
        assert isinstance(si, (float, np.floating))

    def test_high_dimensional_data(self):
        """Test with high-dimensional data."""
        np.random.seed(42)
        data = np.random.randn(100, 20)
        label = np.random.randn(100, 1)

        si, _, _, _ = compute_structure_index(
            data, label, n_bins=5, n_neighbors=10, num_shuffles=0, verbose=False
        )

        assert isinstance(si, (float, np.floating))

    def test_single_feature_data(self):
        """Test with 1D data."""
        np.random.seed(42)
        data = np.random.randn(100, 1)
        label = np.random.randn(100, 1)

        si, _, _, _ = compute_structure_index(
            data, label, n_bins=5, n_neighbors=10, num_shuffles=0, verbose=False
        )

        assert isinstance(si, (float, np.floating))
