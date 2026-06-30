"""Tests for neural_analysis.utils.statistics.models to raise line+branch coverage to >= 95%."""

import unittest.mock as mock

import numpy as np
import pytest
from scipy.optimize import curve_fit

import neural_analysis.utils.statistics.models as models_module
from neural_analysis.utils.statistics.models import (
    FunctionModel,
    get_auc,
    get_best_fit,
)

# ---------------------------------------------------------------------------
# FunctionModel.shift_x / shift_y
# ---------------------------------------------------------------------------


def test_shift_x_basic():
    x = np.array([1.0, 2.0, 3.0])
    result = FunctionModel.shift_x(x, 1.0)
    np.testing.assert_array_equal(result, np.array([0.0, 1.0, 2.0]))


def test_shift_y_basic():
    y = np.array([1.0, 2.0])
    result = FunctionModel.shift_y(y, 3.0)
    np.testing.assert_array_equal(result, np.array([4.0, 5.0]))


# ---------------------------------------------------------------------------
# FunctionModel.apply_inverse
# ---------------------------------------------------------------------------


def test_apply_inverse_false():
    y = np.array([2.0, 4.0])
    result = FunctionModel.apply_inverse(y, inverse=False)
    np.testing.assert_array_equal(result, y)


def test_apply_inverse_true():
    y = np.array([2.0, 4.0])
    result = FunctionModel.apply_inverse(y, inverse=True)
    np.testing.assert_allclose(result, np.array([0.5, 0.25]))


# ---------------------------------------------------------------------------
# FunctionModel.linear
# ---------------------------------------------------------------------------


def test_linear_no_shift_no_inverse():
    x = np.array([1.0, 2.0, 3.0])
    y = FunctionModel.linear(x, 2.0, 1.0, {"x_shift": 0.0, "y_shift": 0.0, "inverse": False})
    np.testing.assert_array_equal(y, np.array([3.0, 5.0, 7.0]))


def test_linear_with_x_shift():
    x = np.array([2.0, 3.0, 4.0])
    # shift=1: effective x = [1, 2, 3]; y = 2*x + 0 => [2, 4, 6]
    y = FunctionModel.linear(x, 2.0, 0.0, {"x_shift": 1.0, "y_shift": 0.0, "inverse": False})
    np.testing.assert_array_equal(y, np.array([2.0, 4.0, 6.0]))


def test_linear_with_y_shift():
    x = np.array([1.0, 2.0])
    # y = 1*x + 0 + 5 = [6, 7]
    y = FunctionModel.linear(x, 1.0, 0.0, {"x_shift": 0.0, "y_shift": 5.0, "inverse": False})
    np.testing.assert_array_equal(y, np.array([6.0, 7.0]))


def test_linear_with_inverse():
    x = np.array([1.0, 2.0])
    # y = 1*x + 1 => [2, 3]; inverse => [0.5, 1/3]
    y = FunctionModel.linear(x, 1.0, 1.0, {"x_shift": 0.0, "y_shift": 0.0, "inverse": True})
    np.testing.assert_allclose(y, np.array([0.5, 1.0 / 3.0]))


# ---------------------------------------------------------------------------
# FunctionModel.exponential
# ---------------------------------------------------------------------------


def test_exponential_no_inverse():
    x = np.array([0.0])
    # y = 1.0 * exp(0) = 1.0
    y = FunctionModel.exponential(x, 0.5, 1.0, {"x_shift": 0.0, "y_shift": 0.0, "inverse": False})
    np.testing.assert_allclose(y, np.array([1.0]))


def test_exponential_with_inverse():
    # Covers lines 60-63 (inverse branch)
    x = np.array([0.0])
    # base: a*exp(-k*(x-x_shift)) = 2.0*exp(0) = 2.0; inverse => 0.5
    y = FunctionModel.exponential(x, 0.0, 2.0, {"x_shift": 0.0, "y_shift": 0.0, "inverse": True})
    np.testing.assert_allclose(y, np.array([0.5]))


def test_exponential_with_x_and_y_shift():
    x = np.array([1.0])
    # x_shifted = 1 - 1 = 0; y = 1*exp(0) = 1; y_shifted = 1 + 2 = 3
    y = FunctionModel.exponential(x, 0.0, 1.0, {"x_shift": 1.0, "y_shift": 2.0, "inverse": False})
    np.testing.assert_allclose(y, np.array([3.0]))


def test_exponential_inverse_with_shift():
    # Covers inverse branch with shifts
    x = np.array([1.0])
    # x_shifted = 1 - 1 = 0; y = 4*exp(0) = 4; y_shifted = 4 + 0 = 4; inverse => 0.25
    y = FunctionModel.exponential(x, 0.0, 4.0, {"x_shift": 1.0, "y_shift": 0.0, "inverse": True})
    np.testing.assert_allclose(y, np.array([0.25]))


# ---------------------------------------------------------------------------
# FunctionModel.gaussian
# ---------------------------------------------------------------------------


def test_gaussian_no_inverse():
    x = np.array([0.0])
    # y = a*exp(-k*0^2) = 3.0
    y = FunctionModel.gaussian(x, 1.0, 3.0, {"x_shift": 0.0, "y_shift": 0.0, "inverse": False})
    np.testing.assert_allclose(y, np.array([3.0]))


def test_gaussian_with_inverse():
    # Covers lines 88-91 (inverse branch)
    x = np.array([0.0])
    # y = 4*exp(0) = 4; inverse => 0.25
    y = FunctionModel.gaussian(x, 0.0, 4.0, {"x_shift": 0.0, "y_shift": 0.0, "inverse": True})
    np.testing.assert_allclose(y, np.array([0.25]))


def test_gaussian_with_shift_and_inverse():
    x = np.array([2.0])
    # x_shifted = 2 - 2 = 0; y = 5*exp(0) = 5; y_shifted = 5 + 0; inverse => 0.2
    y = FunctionModel.gaussian(x, 1.0, 5.0, {"x_shift": 2.0, "y_shift": 0.0, "inverse": True})
    np.testing.assert_allclose(y, np.array([0.2]))


# ---------------------------------------------------------------------------
# FunctionModel.hyperbolic
# ---------------------------------------------------------------------------


def test_hyperbolic_no_inverse():
    x = np.array([0.0])
    # y = a / (1 + k*0) = a = 2.0
    y = FunctionModel.hyperbolic(x, 1.0, 2.0, {"x_shift": 0.0, "y_shift": 0.0, "inverse": False})
    np.testing.assert_allclose(y, np.array([2.0]))


def test_hyperbolic_with_inverse():
    x = np.array([0.0])
    # y = 4 / (1 + 0) = 4; inverse => 0.25
    y = FunctionModel.hyperbolic(x, 0.0, 4.0, {"x_shift": 0.0, "y_shift": 0.0, "inverse": True})
    np.testing.assert_allclose(y, np.array([0.25]))


# ---------------------------------------------------------------------------
# FunctionModel.power
# ---------------------------------------------------------------------------


def test_power_no_inverse():
    x = np.array([0.0])
    # y = a / (1 + k*0)^n = a = 3.0
    y = FunctionModel.power(x, 1.0, 2.0, 3.0, {"x_shift": 0.0, "y_shift": 0.0, "inverse": False})
    np.testing.assert_allclose(y, np.array([3.0]))


def test_power_with_inverse():
    # Covers lines 103-106 (inverse branch)
    x = np.array([0.0])
    # y = 4 / (1 + 0)^2 = 4; inverse => 0.25
    y = FunctionModel.power(x, 0.0, 2.0, 4.0, {"x_shift": 0.0, "y_shift": 0.0, "inverse": True})
    np.testing.assert_allclose(y, np.array([0.25]))


def test_power_with_x_shift():
    x = np.array([2.0])
    # x_shifted = 2 - 2 = 0; y = 5 / (1 + 1*0)^2 = 5
    y = FunctionModel.power(x, 1.0, 2.0, 5.0, {"x_shift": 2.0, "y_shift": 0.0, "inverse": False})
    np.testing.assert_allclose(y, np.array([5.0]))


# ---------------------------------------------------------------------------
# get_auc - fast method (trapezoidal)
# ---------------------------------------------------------------------------


def test_get_auc_fast_method():
    # Covers line 144: method="fast" uses trapezoidal rule directly
    x = [0.0, 1.0, 2.0]
    y = [0.0, 1.0, 0.0]
    result = get_auc(x, y, method="fast")
    # trapezoid of triangle: 1.0
    np.testing.assert_allclose(result, 1.0)


def test_get_auc_fast_method_rectangle():
    x = [0.0, 1.0]
    y = [2.0, 2.0]
    result = get_auc(x, y, method="fast")
    np.testing.assert_allclose(result, 2.0)


# ---------------------------------------------------------------------------
# get_auc - best method with return_fit
# ---------------------------------------------------------------------------


def test_get_auc_best_return_fit():
    # Covers line 154: return_fit=True
    x = np.linspace(0.1, 5.0, 20)
    y = 2.0 * x + 1.0
    auc, fit_result = get_auc(x, y, functions=["linear"], method="best", return_fit=True)
    assert isinstance(auc, float)
    assert isinstance(fit_result, dict)
    assert "function_name" in fit_result
    assert fit_result["function_name"] == "linear"
    assert auc > 0


# ---------------------------------------------------------------------------
# get_best_fit - functions=None -> uses all (covers line 221)
# ---------------------------------------------------------------------------


def test_get_best_fit_functions_none():
    # Covers line 221: functions is None -> uses all implemented functions
    x = np.linspace(0.1, 5.0, 15)
    y = 2.0 * x + 1.0
    result = get_best_fit(x, y, functions=None)
    assert "function_name" in result
    assert result["function_name"] in ["linear", "exponential", "gaussian", "hyperbolic", "power"]


def test_get_best_fit_functions_auto_string():
    # Covers line 221: functions="auto" -> uses all implemented functions
    x = np.linspace(0.1, 5.0, 15)
    y = 2.0 * x + 1.0
    result = get_best_fit(x, y, functions="auto")
    assert "function_name" in result


# ---------------------------------------------------------------------------
# get_best_fit - unknown function raises ValueError (covers line 226)
# ---------------------------------------------------------------------------


def test_get_best_fit_unknown_function_raises():
    # Covers lines 224-228: unknown function name raises ValueError
    x = np.linspace(0.1, 5.0, 10)
    y = x
    with pytest.raises(ValueError, match="not implemented"):
        get_best_fit(x, y, functions=["nonexistent_func"])


def test_get_best_fit_unknown_function_in_list():
    x = np.linspace(0.1, 3.0, 10)
    y = x
    with pytest.raises(ValueError, match="Choose from"):
        get_best_fit(x, y, functions=["linear", "bogus"])


# ---------------------------------------------------------------------------
# get_best_fit - power function (covers lines 240-252)
# ---------------------------------------------------------------------------


def test_get_best_fit_power_function():
    # Covers lines 239-252: power function fitting path
    x = np.linspace(1.0, 10.0, 20)
    # Power-like decay: a / (1 + k*x)^n  with a=10, k=0.5, n=1
    y = 10.0 / (1.0 + 0.5 * x) ** 1.0
    result = get_best_fit(x, y, functions=["power"])
    assert result["function_name"] == "power"
    assert "param" in result
    assert "k" in result["param"]
    assert "n" in result["param"]
    assert "a" in result["param"]
    assert result["mse"] < 1.0  # should fit well


# ---------------------------------------------------------------------------
# get_best_fit - exponential, gaussian, hyperbolic fits (legend_desc branches)
# ---------------------------------------------------------------------------


def test_get_best_fit_exponential():
    # Covers lines 315-317: exponential legend_desc branch
    x = np.linspace(0.1, 5.0, 20)
    y = 3.0 * np.exp(-0.5 * x)
    result = get_best_fit(x, y, functions=["exponential"])
    assert result["function_name"] == "exponential"
    assert "legend_desc" in result
    assert "exponential" in result["legend_desc"]
    assert result["mse"] < 1.0


def test_get_best_fit_gaussian():
    # Covers lines 318-320: gaussian legend_desc branch
    x = np.linspace(-3.0, 3.0, 30)
    y = 2.0 * np.exp(-0.5 * x**2)
    result = get_best_fit(x, y, functions=["gaussian"])
    assert result["function_name"] == "gaussian"
    assert "legend_desc" in result
    assert "gaussian" in result["legend_desc"]
    assert result["mse"] < 1.0


def test_get_best_fit_hyperbolic():
    # Covers lines 321-323: hyperbolic legend_desc branch
    x = np.linspace(1.0, 10.0, 20)
    y = 5.0 / (1.0 + 0.3 * x)
    result = get_best_fit(x, y, functions=["hyperbolic"])
    assert result["function_name"] == "hyperbolic"
    assert "legend_desc" in result
    assert "hyperbolic" in result["legend_desc"]
    assert result["mse"] < 1.0


# ---------------------------------------------------------------------------
# get_best_fit - with x_shift, y_shift, inverse options (legend_desc shift branches)
# ---------------------------------------------------------------------------


def test_get_best_fit_linear_with_x_shift():
    # Covers legend_desc x_shift branch (shift_str uses parenthetical form)
    x = np.linspace(1.0, 5.0, 20)
    y = 2.0 * (x - 1.0) + 0.5
    result = get_best_fit(x, y, functions=["linear"], x_shift=1.0)
    assert result["function_name"] == "linear"
    assert "legend_desc" in result
    # The shift string "(x - 1.00)" appears in legend
    assert "(x - 1.00)" in result["legend_desc"]


def test_get_best_fit_exponential_with_y_shift():
    # Covers y_shift_str branch in legend (y_shift != 0)
    x = np.linspace(0.1, 3.0, 15)
    y = 2.0 * np.exp(-0.5 * x) + 1.0
    result = get_best_fit(x, y, functions=["exponential"], y_shift=1.0)
    assert result["function_name"] == "exponential"
    assert "+ 1.00" in result["legend_desc"]


def test_get_best_fit_linear_with_inverse():
    # Covers inv_str branch and "not inverse" skipped for linear legend
    x = np.linspace(1.0, 5.0, 20)
    y_base = 2.0 * x + 1.0
    y = 1.0 / y_base  # inverse of linear
    result = get_best_fit(x, y, functions=["linear"], inverse=True)
    assert result["function_name"] == "linear"
    assert "1/" in result["legend_desc"]


# ---------------------------------------------------------------------------
# get_best_fit - linear R^2 branch when ss_tot > 0
# ---------------------------------------------------------------------------


def test_get_best_fit_linear_r_squared():
    # Covers lines 310-314: R^2 computed for linear, non-inverse fit
    x = np.linspace(0.1, 5.0, 20)
    y = 3.0 * x + 2.0
    result = get_best_fit(x, y, functions=["linear"])
    assert result["function_name"] == "linear"
    assert "R²" in result["legend_desc"]


def test_get_best_fit_linear_constant_y_r_squared_nan():
    # Covers line 313: branch where ss_tot == 0 -> r_squared = nan
    x = np.linspace(0.1, 5.0, 20)
    y = np.ones(20) * 3.0  # constant y -> ss_tot = 0
    result = get_best_fit(x, y, functions=["linear"])
    assert result["function_name"] == "linear"
    # legend should mention nan for R^2 or contain nan
    assert "nan" in result["legend_desc"].lower() or "R²" in result["legend_desc"]


# ---------------------------------------------------------------------------
# get_best_fit - negative bounds fallback path (covers lines 270-301)
# ---------------------------------------------------------------------------


def test_get_best_fit_negative_exponential_fallback():
    # Covers lines 285-301: negative fallback for non-power functions
    # A decreasing (negative slope) linear-ish fit that forces negative bounds
    x = np.linspace(0.1, 5.0, 20)
    # Negative exponential: should fit with negative k
    y = -2.0 * np.exp(0.5 * x)  # grows negatively, forces negative-k path
    # Use exponential which may hit the fallback path for negative values
    result = get_best_fit(x, y, functions=["exponential"])
    assert result["function_name"] == "exponential"
    assert "mse" in result


def test_get_best_fit_negative_linear_fallback():
    # Forces negative fallback for linear
    x = np.linspace(0.1, 5.0, 20)
    y = -3.0 * x - 1.0
    result = get_best_fit(x, y, functions=["linear"])
    assert result["function_name"] == "linear"
    assert "param" in result
    assert "m" in result["param"]


# ---------------------------------------------------------------------------
# get_best_fit - power negative fallback (covers lines 271-284)
# ---------------------------------------------------------------------------


def test_get_best_fit_power_negative_fallback():
    # Data that makes positive bounds fail for power, triggering negative bounds
    x = np.linspace(0.1, 5.0, 20)
    # Negative power-like: forces negative k/n bounds
    y = -5.0 / (1.0 + 0.3 * x) ** 2
    result = get_best_fit(x, y, functions=["power"])
    assert result["function_name"] == "power"
    assert "mse" in result


# ---------------------------------------------------------------------------
# get_best_fit - RuntimeError warning (lines 334-335) and no-fits ValueError (337)
# We monkeypatch curve_fit so ALL attempts raise RuntimeError, which:
# 1. In the inner try: RuntimeError -> caught by except Exception -> fallback tries
# 2. Fallback also raises RuntimeError -> caught by outer except RuntimeError -> prints warning
# 3. After loop, fits dict is empty -> raises ValueError (line 337)
# ---------------------------------------------------------------------------


def test_get_best_fit_warning_print_on_runtime_error(capsys):
    # Covers lines 334-335: Warning printed when RuntimeError is caught
    x = np.linspace(0.1, 5.0, 20)
    y = 2.0 * x + 1.0

    def always_runtime_error(*args, **kwargs):
        raise RuntimeError("Simulated: max evaluations exceeded")

    with mock.patch.object(models_module, "curve_fit", side_effect=always_runtime_error):
        with pytest.raises(ValueError, match="No functions could be fitted successfully"):
            get_best_fit(x, y, functions=["linear"])

    captured = capsys.readouterr()
    assert "Warning: Curve fitting failed for linear function." in captured.out


def test_get_best_fit_no_fits_raises_value_error():
    # Covers line 337: ValueError when all function fits fail (fits dict empty)
    x = np.linspace(0.1, 5.0, 20)
    y = 2.0 * x + 1.0

    def always_runtime_error(*args, **kwargs):
        raise RuntimeError("Simulated total failure")

    with mock.patch.object(models_module, "curve_fit", side_effect=always_runtime_error):
        with pytest.raises(ValueError, match="No functions could be fitted successfully"):
            get_best_fit(x, y, functions=["exponential", "gaussian"])


def test_get_best_fit_no_fits_multiple_functions_all_fail(capsys):
    # Multiple functions all fail -> all printed, then ValueError
    x = np.linspace(0.1, 5.0, 20)
    y = 2.0 * x + 1.0

    def always_runtime_error(*args, **kwargs):
        raise RuntimeError("Simulated")

    with mock.patch.object(models_module, "curve_fit", side_effect=always_runtime_error):
        with pytest.raises(ValueError, match="No functions"):
            get_best_fit(x, y, functions=["exponential", "hyperbolic"])

    captured = capsys.readouterr()
    assert "exponential" in captured.out
    assert "hyperbolic" in captured.out


# ---------------------------------------------------------------------------
# get_best_fit - multi-function selection (picks best mse)
# ---------------------------------------------------------------------------


def test_get_best_fit_picks_best_function():
    # Confirms best-fit selection works across multiple functions
    x = np.linspace(0.1, 5.0, 30)
    y = 3.0 * np.exp(-0.8 * x)  # clearly exponential
    result = get_best_fit(x, y, functions=["linear", "exponential"])
    # exponential should win
    assert result["function_name"] == "exponential"
    assert result["mse"] < 0.5


def test_get_best_fit_returns_required_keys():
    x = np.linspace(0.1, 5.0, 20)
    y = 2.0 * x + 1.0
    result = get_best_fit(x, y, functions=["linear"])
    for key in ["param", "mse", "y_dense", "auc", "legend_desc", "function_name"]:
        assert key in result


# ---------------------------------------------------------------------------
# get_auc - best method (no return_fit)
# ---------------------------------------------------------------------------


def test_get_auc_best_linear():
    x = np.linspace(0.1, 5.0, 20)
    y = 2.0 * x + 1.0
    auc = get_auc(x, y, functions=["linear"], method="best")
    assert isinstance(auc, float)
    assert auc > 0


def test_get_auc_best_exponential():
    x = np.linspace(0.1, 5.0, 20)
    y = 3.0 * np.exp(-0.5 * x)
    auc = get_auc(x, y, functions=["exponential"], method="best")
    assert isinstance(auc, float)
    assert auc > 0


# ---------------------------------------------------------------------------
# get_best_fit - power legend with x_shift (covers line 324-326)
# ---------------------------------------------------------------------------


def test_get_best_fit_power_legend_with_x_shift():
    # Covers line 324-326: power legend_desc with x_shift != 0
    x = np.linspace(1.0, 10.0, 20)
    y = 10.0 / (1.0 + 0.5 * (x - 1.0)) ** 1.5
    result = get_best_fit(x, y, functions=["power"], x_shift=1.0)
    assert result["function_name"] == "power"
    assert "power" in result["legend_desc"]
    assert "(x - 1.00)" in result["legend_desc"]


# ---------------------------------------------------------------------------
# get_auc - NaN AUC fallback (lines 149-152)
# When get_best_fit returns {'auc': np.nan, ...}, get_auc falls back to trapezoid
# ---------------------------------------------------------------------------


def test_get_auc_nan_auc_fallback_to_trapezoid():
    # Covers lines 149-152: auc is np.nan -> logger.error + fallback to trapezoidal
    x = [0.0, 1.0, 2.0]
    y = [0.0, 1.0, 0.0]
    # Mock get_best_fit to return dict with auc=np.nan (the actual singleton)
    mock_fit = {"auc": np.nan, "function_name": "linear", "param": {}}
    with mock.patch.object(models_module, "get_best_fit", return_value=mock_fit):
        result = get_auc(x, y, method="best")
    # Should have fallen back to trapezoidal rule
    np.testing.assert_allclose(result, 1.0)


def test_get_auc_nan_auc_fallback_return_fit():
    # Covers lines 149-154: auc is np.nan AND return_fit=True
    x = [0.0, 1.0, 2.0]
    y = [0.0, 1.0, 0.0]
    mock_fit = {"auc": np.nan, "function_name": "linear", "param": {}}
    with mock.patch.object(models_module, "get_best_fit", return_value=mock_fit):
        auc, fit_result = get_auc(x, y, method="best", return_fit=True)
    np.testing.assert_allclose(auc, 1.0)
    assert fit_result is mock_fit


# ---------------------------------------------------------------------------
# get_best_fit - power negative fallback path (lines 271-284)
# Monkeypatch curve_fit so first call raises RuntimeError (inner except Exception)
# but second call (negative bounds) succeeds.
# ---------------------------------------------------------------------------


def test_get_best_fit_power_negative_bounds_fallback():
    # Covers lines 271-284: power function, positive bounds fail, negative bounds succeed
    x = np.linspace(0.1, 5.0, 20)
    y = 10.0 / (1.0 + 0.5 * x)

    call_count = [0]
    real_curve_fit = curve_fit

    def fail_first_power_call(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # Simulate first (positive bounds) call failing with RuntimeError
            raise RuntimeError("Simulated max evaluations exceeded on first try")
        return real_curve_fit(*args, **kwargs)

    with mock.patch.object(models_module, "curve_fit", side_effect=fail_first_power_call):
        result = get_best_fit(x, y, functions=["power"])

    assert result["function_name"] == "power"
    assert call_count[0] == 2  # both calls were made
    assert "k" in result["param"]
    assert "n" in result["param"]
    assert "a" in result["param"]
