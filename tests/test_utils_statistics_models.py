import numpy as np

from neural_analysis.utils.statistics.models import FunctionModel, get_auc, get_best_fit


def test_function_model_linear():
    x = np.array([1.0, 2.0, 3.0])
    y = FunctionModel.linear(x, 2.0, 1.0, {"x_shift": 0.0, "y_shift": 0.0, "inverse": False})
    np.testing.assert_array_equal(y, np.array([3.0, 5.0, 7.0]))

def test_get_auc():
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 0])
    res = get_auc(x, y, functions=["gaussian"])
    assert res > 0

def test_get_best_fit():
    x = np.linspace(0.1, 5, 20)
    y = 3.0 * x + 2.0

    results = get_best_fit(x, y, functions=["linear"])
    assert results["function_name"] == "linear"
    assert "mse" in results
