import numpy as np

from neural_analysis.metrics.distributions import shape_distance_procrustes
from neural_analysis.metrics.pairwise_core import compare_datasets
from neural_analysis.topology.structure_index import compute_structure_index


def test_feature_similarity_migration():
    """
    Test that the new `compare_datasets` (mode='all-pairs') correctly replaces
    the legacy `feature_similarity` functionality.
    """
    # Dummy data representing groups of population vectors
    group1 = np.random.rand(10, 5)
    group2 = np.random.rand(10, 5)

    data_dict = {
        "group1": group1,
        "group2": group2
    }

    # In legacy feature_similarity, mode='between' or 'inside' was handled by compare_distribution_groups
    # Now handled by compare_datasets mode='all-pairs', but metric must return scalar
    # Since Euclidean between arrays returns a distance matrix, we use 'wasserstein' which returns scalar
    result = compare_datasets(data_dict, mode="all-pairs", metric="wasserstein")

    assert isinstance(result, dict)
    assert "group1" in result
    assert "group2" in result["group1"]
    assert isinstance(result["group1"]["group2"], float)

def test_shape_distance_migration():
    """
    Test that `shape_distance_procrustes` correctly replaces the old `shape_distance`
    and `calc_shape_similarity` with 'procrustes' method.
    """
    data1 = np.random.rand(20, 3)
    data2 = np.random.rand(20, 3)

    # In legacy, shape_distance returned dist, pairs, metadata
    # shape_distance_procrustes returns dist, pairs (list of tuples or array)
    dist, pairs = shape_distance_procrustes(data1, data2, return_pairs=True)

    assert isinstance(dist, float)
    assert dist >= 0
    # pairs could be a list of tuples, or dict, let's just check length
    assert len(pairs) == 20

def test_structure_index_migration():
    """
    Test that the new `compute_structure_index` replaces the legacy `structure_index` function.
    """
    data = np.random.rand(50, 3)
    labels = np.random.rand(50, 2)

    # New signature
    si, overlap_mats, _, _ = compute_structure_index(
        data=data,
        label=labels,
        n_bins=10,
        distance_metric="euclidean",
        n_neighbors=5,
        num_shuffles=0,
        discrete_label=False,
        verbose=False
    )

    assert isinstance(si, float)
    assert isinstance(overlap_mats, tuple)
    assert len(overlap_mats) == 2
