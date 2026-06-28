import numpy as np

from neural_analysis.learning import decode


def test_decode_classification():
    np.random.seed(42)
    # Binary classification problem
    embedding_train = np.random.rand(400, 5)
    embedding_test = np.random.rand(100, 5)
    labels_train = np.random.randint(0, 2, 400)
    labels_test = np.random.randint(0, 2, 100)

    results = decode(
        embedding_train=embedding_train,
        embedding_test=embedding_test,
        labels_train=labels_train,
        labels_test=labels_test,
        test_outlier_removal=False,
    )

    assert isinstance(results, dict)
    assert "accuracy" in results


def test_decode_regression():
    np.random.seed(42)
    # Regression problem
    embedding_train = np.random.rand(400, 5)
    embedding_test = np.random.rand(100, 5)
    labels_train = np.random.rand(400)
    labels_test = np.random.rand(100)

    results = decode(
        embedding_train=embedding_train,
        embedding_test=embedding_test,
        labels_train=labels_train,
        labels_test=labels_test,
        test_outlier_removal=False,
    )

    assert isinstance(results, dict)
    assert "rmse" in results
    assert "r2" in results
