from collections.abc import Callable
from typing import Any

import numpy as np

FunctionType = Callable[[np.ndarray, float, float | None, float | None], np.ndarray]
import logging
from typing import Literal

from scipy.optimize import curve_fit

from neural_analysis.utils.common.collections import (
    make_list_ifnot,
)

logger = logging.getLogger(__name__)


class FunctionModel:
    """Base class for function models with shift and inverse options."""

    @staticmethod
    def shift_x(x: np.ndarray, x_shift: float) ->np.ndarray:
        """Apply horizontal shift to x values."""
        return x - x_shift

    @staticmethod
    def shift_y(y: np.ndarray, y_shift: float) ->np.ndarray:
        """Apply vertical shift to y values."""
        return y + y_shift

    @classmethod
    def apply_inverse(cls, y: np.ndarray, inverse: bool) ->np.ndarray:
        """Apply inverse transformation if requested."""
        return 1.0 / y if inverse else y

    @classmethod
    def linear(cls, x: np.ndarray, m: float, c: float, config: dict[str, Any]
        ) ->np.ndarray:
        """Linear function: y = m*(x - x_shift) + c or its inverse."""
        x_shifted = cls.shift_x(x, config.get('x_shift', 0.0))
        y = m * x_shifted + c
        y_shifted = cls.shift_y(y, config.get('y_shift', 0.0))
        return cls.apply_inverse(y_shifted, config.get('inverse', False))

    @classmethod
    def exponential(cls, x: np.ndarray, k: float, a: float, config: dict[
        str, Any]) ->np.ndarray:
        """Exponential function: y = a*exp(-k*(x - x_shift)) or its inverse."""
        x_shifted = cls.shift_x(x, config.get('x_shift', 0.0))
        y = a * np.exp(-k * x_shifted)
        y_shifted = cls.shift_y(y, config.get('y_shift', 0.0))
        return cls.apply_inverse(y_shifted, config.get('inverse', False))

    @classmethod
    def gaussian(cls, x: np.ndarray, k: float, a: float, config: dict[str, Any]
        ) ->np.ndarray:
        """Gaussian function: y = a*exp(-k*(x - x_shift)^2) or its inverse."""
        x_shifted = cls.shift_x(x, config.get('x_shift', 0.0))
        y = a * np.exp(-k * x_shifted ** 2)
        y_shifted = cls.shift_y(y, config.get('y_shift', 0.0))
        return cls.apply_inverse(y_shifted, config.get('inverse', False))

    @classmethod
    def hyperbolic(cls, x: np.ndarray, k: float, a: float, config: dict[str,
        Any]) ->np.ndarray:
        """Hyperbolic function: y = a/(1 + k*(x - x_shift)) or its inverse."""
        x_shifted = cls.shift_x(x, config.get('x_shift', 0.0))
        y = a / (1 + k * x_shifted)
        y_shifted = cls.shift_y(y, config.get('y_shift', 0.0))
        return cls.apply_inverse(y_shifted, config.get('inverse', False))

    @classmethod
    def power(cls, x: np.ndarray, k: float, n: float, a: float, config:
        dict[str, Any]) ->np.ndarray:
        """Power function: y = a/(1 + k*(x - x_shift))^n or its inverse."""
        x_shifted = cls.shift_x(x, config.get('x_shift', 0.0))
        y = a / (1 + k * x_shifted) ** n
        y_shifted = cls.shift_y(y, config.get('y_shift', 0.0))
        return cls.apply_inverse(y_shifted, config.get('inverse', False))


def get_auc(x: list[float], y: list[float], functions: str | list[Literal['auto', 'linear', 'exponential', 'gaussian', 'hyperbolic', 'power']]='auto', method: Literal['best', 'fast']='best', return_fit:
    bool=False) ->float:
    """Calculate area under the curve based on the best fit of given functions or using trapezoidal rule.

    Parameters:
    ----------
    x : List[float]
        Independent variable values (x-axis).
    y : List[float]
        Dependent variable values (y-axis).
    functions : List[str], optional
        List of function names to fit: 'linear', 'exponential', 'gaussian', 'hyperbolic', 'power'.
        If "auto", all implemented functions are used for fitting and selecting the best fit.
        Whether to plot the best fit curve along with data points.
    method : {"best", "fast"}, default="best"
        Method to calculate AUC:
        - "best": Fit multiple functions and use the best fit for AUC calculation.
        - "fast": Use trapezoidal rule for quick AUC calculation.
    return_fit : bool, default=False
        If True and method is "best", return the fit result dictionary along with AUC.

    Returns:
    -------
    float
        Calculated area under the curve (AUC).
    """
    if method == 'fast':
        auc = np.trapezoid(y, x)
    else:
        fit_result = get_best_fit(x, y, functions=functions)
        auc = fit_result.get('auc', np.nan)
        if auc is np.nan:
            logger.error(
                'Failed to compute AUC from best fit; falling back to trapezoidal rule.'
                )
            auc = np.trapezoid(y, x)
        if return_fit:
            return auc, fit_result
    return auc


def get_best_fit(x: list[float], y: list[float], functions: str | list[Literal['auto', 'linear', 'exponential', 'gaussian', 'hyperbolic', 'power']]='auto', x_shift: float=0.0, y_shift: float=0.0, inverse:
    bool=False, maxfev: int=10000) ->dict[str, float | list[float] | np.ndarray | str]:
    """
    Fit multiple mathematical models to (x, y) data points and determine the best-fitting model
    based on Mean Squared Error (MSE). Supports horizontal and vertical shifts and inverse function options.

    Parameters:
    ----------
    x : List[float]
        Independent variable values (x-axis).
    y : List[float]
        Dependent variable values (y-axis).
    functions : List[str], optional
        List of function names to fit: 'linear', 'exponential', 'gaussian', 'hyperbolic', 'power'.
        If None, all implemented functions are used.
        Directory to save the plot of the best-fitting model. If None, plot is not saved.
        Whether to display the plot of the data and best-fitting model.
    x_shift : float, default=0.0
        Horizontal shift applied to x values for all functions.
    y_shift : float, default=0.0
        Vertical shift applied to y values for all functions.
    inverse : bool, default=False
        If True, fits the inverse of each function (1/f(x)).

    Returns:
    -------
    Dict[str, Union[float, List[float], np.ndarray, str]]
        Dictionary containing:
        - 'param': Fitted parameter(s).
        - 'mse': Mean squared error of the fit.
        - 'y_dense': Model output over a dense x-range for plotting.
        - 'auc': Area under the fitted curve.
        - 'function_name': Name of the best-fitting function.
        - 'legend_desc': Description for plot legend.

    Raises:
    ------
    ValueError
        If any function name is not implemented.
    """
    x = np.array(x)
    y = np.array(y)
    x_dense = np.linspace(min(x), max(x), 100)
    fits: dict[str, dict] = {}
    implemented_functions = ['linear', 'exponential', 'gaussian',
        'hyperbolic', 'power']
    function_map: dict[str, FunctionType] = {'linear': FunctionModel.linear,
        'exponential': FunctionModel.exponential, 'gaussian': FunctionModel
        .gaussian, 'hyperbolic': FunctionModel.hyperbolic, 'power':
        FunctionModel.power}
    if functions is None or isinstance(functions, str) and functions == 'auto':
        functions = implemented_functions
    else:
        functions = make_list_ifnot(functions)
        for func in functions:
            if func not in implemented_functions:
                raise ValueError(
                    f"Function '{func}' is not implemented. Choose from {implemented_functions} or use 'auto'."
                    )
    config: dict[str, Any] = {'x_shift': x_shift, 'y_shift': y_shift,
        'inverse': inverse}
    initial_guesses = {'positive': [0.1, 2.0, 1.0], 'negative': [-0.1, -2.0,
        1.0]}
    for func_name in functions:
        func = function_map[func_name]
        try:
            try:
                if func_name == 'power':
                    p0 = initial_guesses['positive']
                    bounds = [0, 0, -np.inf], [np.inf, np.inf, np.inf]
                    popt, _ = curve_fit(lambda x, k, n, a: func(x, k, n, a,
                        config), x, y, p0=p0, bounds=bounds, maxfev=maxfev)
                    y_pred = func(x, *popt, config)
                    y_dense = func(x_dense, *popt, config)
                    param = {'k': popt[0], 'n': popt[1], 'a': popt[2]}
                else:
                    p0 = [0.1, 1.0]
                    bounds = [0, -np.inf], [np.inf, np.inf]
                    popt, _ = curve_fit(lambda x, k, a: func(x, k, a,
                        config), x, y, p0=p0, bounds=bounds, maxfev=maxfev)
                    y_pred = func(x, *popt, config)
                    y_dense = func(x_dense, *popt, config)
                    if func_name == 'linear':
                        param = {'m': popt[0], 'c': popt[1]}
                    else:
                        param = {'k': popt[0], 'a': popt[1]}
            except:
                if func_name == 'power':
                    p0 = initial_guesses['negative']
                    bounds = [-np.inf, -np.inf, -np.inf], [0, 0, np.inf]
                    popt, _ = curve_fit(lambda x, k, n, a: func(x, k, n, a,
                        config), x, y, p0=p0, bounds=bounds, maxfev=maxfev)
                    y_pred = func(x, *popt, config)
                    y_dense = func(x_dense, *popt, config)
                    param = {'k': popt[0], 'n': popt[1], 'a': popt[2]}
                else:
                    p0 = [0.1, -1.0]
                    bounds = [-np.inf, -np.inf], [0, np.inf]
                    popt, _ = curve_fit(lambda x, k, a: func(x, k, a,
                        config), x, y, p0=p0, bounds=bounds, maxfev=maxfev)
                    y_pred = func(x, *popt, config)
                    y_dense = func(x_dense, *popt, config)
                    if func_name == 'linear':
                        param = {'m': popt[0], 'c': popt[1]}
                    else:
                        param = {'k': popt[0], 'a': popt[1]}
            mse = np.mean((y_pred - y) ** 2)
            auc = np.trapezoid(y_dense, x=x_dense)
            shift_str = f'(x - {x_shift:.2f})' if x_shift != 0 else 'x'
            y_shift_str = f' + {y_shift:.2f}' if y_shift != 0 else ''
            inv_str = '1/' if inverse else ''
            if func_name == 'linear':
                legend_desc = f"""Function: linear
Fit: y = {inv_str}({popt[0]:.4f}{shift_str} + {popt[1]:.4f}){y_shift_str}"""
                if not inverse:
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    ss_res = np.sum((y - y_pred) ** 2)
                    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                    legend_desc += f'\nR²: {r_squared:.4f}'
            elif func_name == 'exponential':
                legend_desc = f"""Function: exponential
Fit: y = {inv_str}({popt[1]:.4f}exp(-{popt[0]:.4f}{shift_str})){y_shift_str}"""
            elif func_name == 'gaussian':
                legend_desc = f"""Function: gaussian
Fit: y = {inv_str}({popt[1]:.4f}exp(-{popt[0]:.4f}{shift_str}^2)){y_shift_str}"""
            elif func_name == 'hyperbolic':
                legend_desc = f"""Function: hyperbolic
Fit: y = {inv_str}({popt[1]:.4f}/(1 + {popt[0]:.4f}{shift_str})){y_shift_str}"""
            elif func_name == 'power':
                legend_desc = f"""Function: power
Fit: y = {inv_str}({popt[2]:.4f}/(1 + {popt[0]:.4f}{shift_str})^{popt[1]:.4f}){y_shift_str}"""
            fits[func_name] = {'param': param, 'mse': mse, 'y_dense':
                y_dense, 'auc': auc, 'legend_desc': legend_desc}
        except RuntimeError:
            print(f'Warning: Curve fitting failed for {func_name} function.')
    if not fits:
        raise ValueError('No functions could be fitted successfully.')
    best_fit_name = min(fits, key=lambda x: fits[x]['mse'])
    best_fit_dict = fits[best_fit_name]
    best_fit_dict['function_name'] = best_fit_name
    return best_fit_dict
