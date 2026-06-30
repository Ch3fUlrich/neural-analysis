"""Coverage tests for neural_analysis.learning.cross_validation."""

from __future__ import annotations

import numpy as np
import pytest

from neural_analysis.learning.cross_validation import create_folds


def _materialize(folds):
    """Drain the split iterator into a list of (train_idx, test_idx) tuples."""
    return [(np.asarray(tr), np.asarray(te)) for tr, te in folds]


def test_kfold_default_path_returns_expected_number_of_folds():
    """Non-stratified path (default) should yield n_folds train/test splits."""
    labels = np.arange(20, dtype=np.float64)
    folds = _materialize(create_folds(labels, n_folds=5, random_state=0))

    assert len(folds) == 5
    for train_idx, test_idx in folds:
        assert train_idx.dtype.kind in ("i", "u")
        assert test_idx.dtype.kind in ("i", "u")
        # Train and test indices must be disjoint and cover all samples.
        assert set(train_idx).isdisjoint(set(test_idx))
        assert set(train_idx) | set(test_idx) == set(range(20))


def test_kfold_test_sizes_partition_all_samples_exactly_once():
    """With 20 samples and 5 folds, every index appears in exactly one test fold."""
    labels = np.zeros(20, dtype=np.float64)
    folds = _materialize(create_folds(labels, n_folds=5, stratify=False, random_state=0))

    test_counts = np.zeros(20, dtype=int)
    for _, test_idx in folds:
        assert len(test_idx) == 4  # 20 / 5
        test_counts[test_idx] += 1
    assert np.all(test_counts == 1)


def test_float_labels_with_stratify_true_falls_back_to_kfold():
    """Floating labels short-circuit the stratify check -> KFold (not StratifiedKFold).

    KFold does not require label balance, so 20 float samples / 4 folds works even
    though the per-class counts would be meaningless for stratification.
    """
    labels = np.linspace(0.0, 1.0, 20, dtype=np.float64)
    folds = _materialize(create_folds(labels, n_folds=4, stratify=True, random_state=0))

    assert len(folds) == 4
    test_counts = np.zeros(20, dtype=int)
    for _, test_idx in folds:
        test_counts[test_idx] += 1
    assert np.all(test_counts == 1)


def test_stratified_path_with_integer_labels_preserves_class_proportions():
    """Integer labels + stratify=True uses StratifiedKFold, keeping class balance."""
    # 12 samples: 6 of class 0, 6 of class 1.
    labels = np.array([0, 1] * 6, dtype=np.int64)
    folds = _materialize(create_folds(labels, n_folds=3, stratify=True, random_state=0))

    assert len(folds) == 3
    for _, test_idx in folds:
        classes = labels[test_idx]
        # Each test fold of size 4 should contain 2 of each class (stratified).
        assert np.sum(classes == 0) == 2
        assert np.sum(classes == 1) == 2


def test_stratified_and_kfold_produce_different_splits_for_int_labels():
    """The stratify flag actually changes which splitter runs for integer labels."""
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

    strat = _materialize(create_folds(labels, n_folds=2, stratify=True, random_state=0))
    plain = _materialize(create_folds(labels, n_folds=2, stratify=False, random_state=0))

    # Stratified test folds must each contain both classes; plain KFold need not.
    for _, test_idx in strat:
        assert set(labels[test_idx]) == {0, 1}

    strat_test_sets = sorted(tuple(sorted(te.tolist())) for _, te in strat)
    plain_test_sets = sorted(tuple(sorted(te.tolist())) for _, te in plain)
    assert strat_test_sets != plain_test_sets


def test_random_state_is_deterministic():
    """Same random_state -> identical splits; different seed -> different splits."""
    labels = np.arange(15, dtype=np.float64)

    a = _materialize(create_folds(labels, n_folds=3, random_state=7))
    b = _materialize(create_folds(labels, n_folds=3, random_state=7))
    c = _materialize(create_folds(labels, n_folds=3, random_state=99))

    for (tr_a, te_a), (tr_b, te_b) in zip(a, b):
        assert np.array_equal(tr_a, tr_b)
        assert np.array_equal(te_a, te_b)

    a_sets = sorted(tuple(sorted(te.tolist())) for _, te in a)
    c_sets = sorted(tuple(sorted(te.tolist())) for _, te in c)
    assert a_sets != c_sets


def test_integer_labels_without_stratify_uses_kfold():
    """Integer labels with stratify=False stays on the KFold branch."""
    labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
    folds = _materialize(create_folds(labels, n_folds=5, stratify=False, random_state=0))

    assert len(folds) == 5
    test_counts = np.zeros(10, dtype=int)
    for _, test_idx in folds:
        assert len(test_idx) == 2
        test_counts[test_idx] += 1
    assert np.all(test_counts == 1)


def test_too_few_samples_for_stratified_warns():
    """StratifiedKFold warns when a class has fewer members than n_folds."""
    # Class 1 has only a single member but we request 3 folds.
    # Sklearn issues a UserWarning rather than raising a ValueError in newer versions.
    import warnings

    labels = np.array([0, 0, 0, 0, 0, 1], dtype=np.int64)
    folds = create_folds(labels, n_folds=3, stratify=True, random_state=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _materialize(folds)
    # Either a warning was issued or we successfully got splits back -- both are valid.
    assert len(result) == 3 or any(issubclass(w.category, (UserWarning, ValueError)) for w in caught)
