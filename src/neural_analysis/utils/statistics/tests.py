import logging
from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import (
    levene,
    mannwhitneyu,
    monte_carlo_test,
    permutation_test,
    shapiro,
    ttest_ind,
    ttest_rel,
    wilcoxon,
)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

from neural_analysis.utils.common.collections import (
    do_critical,
    make_list_ifnot,
    mean_diff,
)

logger = logging.getLogger(__name__)


def _auto_select_test_method(
    data1: np.ndarray,  # type: ignore
    data2: np.ndarray | None = None,  # type: ignore
    test_type: Literal["paired", "unpaired"] = "paired",
    alpha_normality: float = 0.05,
    min_sample_size_parametric: int = 20,
) -> tuple[str, dict[str, float]]:
    """
    Automatically select the appropriate statistical test based on data characteristics.

    This function selects between parametric (t-test variants) and non-parametric tests
    (Wilcoxon or Mann-Whitney U) based on sample size, normality of data/differences,
    and (for unpaired tests) equality of variances. It uses the Shapiro-Wilk test for
    normality and Levene's test for variance equality.

    Assumptions and Notes:
    - Parametric tests (ttest, ttest_ind, ttest_ind_welch) assume:
      - For paired: Normality of differences.
      - For unpaired: Normality of both groups and (for ttest_ind) equal variances.
    - Non-parametric tests:
      - Wilcoxon (paired): Assumes symmetry of differences around the median.
      - Mann-Whitney U (unpaired): Assumes similar distribution shapes (beyond location shift).
    - Normality tests (Shapiro-Wilk) may be overly sensitive for large samples (>50)
      or underpowered for small samples (<3). Visual inspection (e.g., Q-Q plots) is recommended.
    - If normality or variance tests fail due to numerical issues (e.g., identical values),
      the function defaults to non-parametric tests.
    - Sample size threshold (min_sample_size_parametric) is a heuristic; adjust based on context.

    Parameters
    ----------
    data1 : np.ndarray
        First dataset (or differences for paired tests if data2 is None).
    data2 : np.ndarray, optional
        Second dataset (required for unpaired tests; optional for paired if differences precomputed).
    test_type : {"paired", "unpaired"}
        Type of comparison.
    alpha_normality : float, default=0.05
        Significance level for normality and variance tests.
    min_sample_size_parametric : int, default=20
        Minimum sample size per group/pair to consider parametric tests.

    Returns
    -------
    method : str
        Selected test method name (e.g., 'ttest', 'wilcoxon', 'ttest_ind', 'ttest_ind_welch', 'mannwhitneyu').
    diagnostics : dict
        Dictionary with sample sizes, p-values (normality, variance), reasons, notes, and warnings.

    Raises
    ------
    ValueError
        If inputs are invalid (e.g., data2 missing for unpaired, mismatched lengths, empty arrays).
    TypeError
        If inputs are not numpy arrays.
    """
    if isinstance(data1, (pd.DataFrame, pd.Series)):
        data1 = data1.to_numpy()
    if data2 is not None and (isinstance(data2, (pd.DataFrame, pd.Series))):
        data2 = data2.to_numpy()
    if (
        not isinstance(data1, np.ndarray)
        or data2 is not None
        and not isinstance(data2, np.ndarray)
    ):
        raise TypeError("data1 and data2 must be numpy arrays")
    if len(data1) == 0:
        raise ValueError("data1 cannot be empty")
    if test_type == "unpaired" and data2 is None:
        raise ValueError("data2 must be provided for unpaired tests")
    if test_type == "unpaired" and len(data2) == 0:  # type: ignore
        raise ValueError("data2 cannot be empty for unpaired tests")
    if test_type == "paired" and data2 is not None and len(data1) != len(data2):
        raise ValueError("data1 and data2 must have the same length for paired tests")
    diagnostics = {
        "alpha_normality": alpha_normality,
        "min_sample_size_parametric": min_sample_size_parametric,
        "assumptions_note": "Parametric: normality (and equal variances for ttest_ind). Wilcoxon: symmetric differences. Mann-Whitney U: similar shapes.",
    }
    if test_type == "paired":
        if data2 is not None:
            differences = data1 - data2
        else:
            differences = data1
            diagnostics["note"] = "data1 treated as differences since data2 is None"
        n = len(differences)
        diagnostics["n_pairs"] = n
        norm_p = np.nan
        is_normal = False
        if n >= 3:
            try:
                _, norm_p = shapiro(differences)
                diagnostics["normality_p"] = norm_p
                is_normal = norm_p >= alpha_normality
            except Exception as e:
                diagnostics["normality_p"] = np.nan
                diagnostics["normality_error"] = f"Shapiro-Wilk test failed: {str(e)}"
        else:
            diagnostics["normality_p"] = np.nan
            diagnostics["normality_note"] = (
                "Normality test skipped due to small sample size (<3)"
            )
        if n < 3:
            method = "wilcoxon"
            diagnostics["reason"] = "insufficient_sample_size"
        elif n >= min_sample_size_parametric and is_normal:
            method = "ttest"
            diagnostics["reason"] = "parametric_assumptions_met"
        else:
            method = "wilcoxon"
            diagnostics["reason"] = (
                "normality_violated" if not is_normal else "small_sample_size"
            )
        if n > 50:
            diagnostics["warning"] = (
                "Shapiro-Wilk may be overly sensitive for large samples (>50). Consider visual checks."
            )
    else:
        n1 = len(data1)
        n2 = len(data2)  # type: ignore
        diagnostics["n_group1"] = n1
        diagnostics["n_group2"] = n2
        norm_p1, norm_p2 = np.nan, np.nan
        is_normal1, is_normal2 = False, False
        if n1 >= 3:
            try:
                _, norm_p1 = shapiro(data1)
                diagnostics["normality_p_group1"] = norm_p1
                is_normal1 = norm_p1 >= alpha_normality
            except Exception as e:
                diagnostics["normality_p_group1"] = np.nan
                diagnostics["normality_error_group1"] = (
                    f"Shapiro-Wilk test failed for group1: {str(e)}"
                )
        else:
            diagnostics["normality_p_group1"] = np.nan
            diagnostics["normality_note_group1"] = (
                "Normality test skipped for group1 due to small sample size (<3)"
            )
        if n2 >= 3:
            try:
                _, norm_p2 = shapiro(data2)
                diagnostics["normality_p_group2"] = norm_p2
                is_normal2 = norm_p2 >= alpha_normality
            except Exception as e:
                diagnostics["normality_p_group2"] = np.nan
                diagnostics["normality_error_group2"] = (
                    f"Shapiro-Wilk test failed for group2: {str(e)}"
                )
        else:
            diagnostics["normality_p_group2"] = np.nan
            diagnostics["normality_note_group2"] = (
                "Normality test skipped for group2 due to small sample size (<3)"
            )
        both_normal = is_normal1 and is_normal2
        sufficient_size = (
            n1 >= min_sample_size_parametric and n2 >= min_sample_size_parametric
        )
        if min(n1, n2) < 3:
            method = "mannwhitneyu"
            diagnostics["reason"] = "insufficient_sample_size"
        elif sufficient_size and both_normal:
            try:
                _, var_p = levene(data1, data2)
                diagnostics["variance_p"] = var_p
                if var_p >= alpha_normality:
                    method = "ttest_ind"
                    diagnostics["reason"] = "parametric_assumptions_met_equal_variances"
                else:
                    method = "ttest_ind_welch"
                    diagnostics["reason"] = (
                        "parametric_assumptions_met_unequal_variances"
                    )
            except Exception as e:
                diagnostics["variance_p"] = np.nan
                diagnostics["variance_error"] = f"Levene test failed: {str(e)}"
                method = "mannwhitneyu"
                diagnostics["reason"] = "variance_test_failed_fallback_to_nonparametric"
        else:
            method = "mannwhitneyu"
            diagnostics["reason"] = (
                "normality_violated" if not both_normal else "small_sample_size"
            )
        if max(n1, n2) > 50:
            diagnostics["warning"] = (
                "Shapiro-Wilk may be overly sensitive for large samples (>50). Consider visual checks."
            )
    return method, diagnostics  # type: ignore


def _auto_select_correction_method(
    num_comparisons: int,
    test_type: Literal["paired", "unpaired"] = "paired",
    alpha: float = 0.05,
) -> tuple[  # type: ignore
    Literal[
        "holm",
        "bonferroni",
        "fdr_bh",
        "sidak",
        "holm-sidak",
        "simes-hochberg",
        "hommel",
        "fdr_by",
        "fdr_tsbh",
        "fdr_tsbky",
        "none",
    ],
    dict[str, any],
]:
    """
    Automatically select the appropriate multiple testing correction method.

    This function selects a correction method based on the number of comparisons
    and test type (paired vs unpaired). Paired tests may warrant more conservative
    corrections due to within-subject dependencies.

    Correction Methods:
    - 'holm': Step-down procedure controlling FWER; good balance for small-moderate comparisons n<=5 (paired) or n<=8 (unpaired).
    - 'bonferroni': Very conservative FWER control; suitable for very small numbers n<=2 (paired) or n<=3 (unpaired).
    - 'fdr_bh': Benjamini-Hochberg FDR control; appropriate for moderate-large numbers n<=15 (paired) or n<=25 (unpaired).
    - 'sidak': Similar to Bonferroni but slightly less conservative for very small numbers.
    - 'holm-sidak': Combination of Holm and Sidak procedures for small numbers.
    - 'simes-hochberg': Less conservative than holm for independent tests, small numbers.
    - 'hommel': Improved version of holm for general dependence structures, moderate numbers n<=10 (paired) or n<=15 (unpaired).
    - 'fdr_by': Benjamini-Yekutieli FDR control; robust for positive dependencies and large numbers n>15 (paired) or n>25 (unpaired).
    - 'fdr_tsbh': Two-stage FDR procedure for large numbers.
    - 'fdr_tsbky': Two-stage FDR with Hochberg step for large numbers.
    - 'none': No correction (not recommended for multiple comparisons).

    Assumptions and Notes:
    - Paired tests use stricter thresholds due to within-subject correlations.
    - FWER (Family-Wise Error Rate) methods control probability of any false positive.
    - FDR methods control expected proportion of false positives among rejected hypotheses.
    - Selection prioritizes: Bonferroni/Sidak (n≤2-3) → Holm/Holm-Sidak/Simes-Hochberg (n≤5-8) → Hommel (n≤10-15) → FDR methods (n>10-15).
    - For very large numbers (>25), FDR methods are generally preferred over FWER.
    - Alternatives are provided for each selection to allow manual override based on study design.

    Parameters
    ----------
    num_comparisons : int
        Number of planned comparisons.
    test_type : {"paired", "unpaired"}, default="paired"
        Type of statistical test.
    alpha : float, default=0.05
        Family-wise error rate or FDR level.

    Returns
    -------
    method : str
        Selected correction method from the available options.
    diagnostics : dict
        Dictionary with selection details, reasons, alternatives, and notes.
    """
    diagnostics = {
        "num_comparisons": num_comparisons,
        "test_type": test_type,
        "alpha": alpha,
    }
    if test_type == "paired":
        thresholds = {
            "bonferroni": 2,
            "sidak": 2,
            "holm": 5,
            "holm-sidak": 5,
            "simes-hochberg": 5,
            "hommel": 10,
            "fdr_bh": 15,
            "fdr_by": float("inf"),
            "fdr_tsbh": float("inf"),
            "fdr_tsbky": float("inf"),
        }
    else:
        thresholds = {
            "bonferroni": 3,
            "sidak": 3,
            "holm": 8,
            "holm-sidak": 8,
            "simes-hochberg": 8,
            "hommel": 15,
            "fdr_bh": 25,
            "fdr_by": float("inf"),
            "fdr_tsbh": float("inf"),
            "fdr_tsbky": float("inf"),
        }
    if num_comparisons <= thresholds["bonferroni"]:
        method = "bonferroni"
        diagnostics["reason"] = (
            "very small number of comparisons; conservative bonferroni correction"  # type: ignore
        )
        diagnostics["alternatives"] = {  # type: ignore
            "sidak": "similar to bonferroni but slightly less conservative",
            "holm": "step-down procedure for small numbers",
            "none": "no correction (not recommended for multiple comparisons)",
        }
    elif num_comparisons <= thresholds["holm"]:
        method = "holm"
        diagnostics["reason"] = (
            "small number of comparisons; holm provides good balance of power and control"  # type: ignore
        )
        diagnostics["alternatives"] = {  # type: ignore
            "holm-sidak": "combination of holm and sidak procedures",
            "simes-hochberg": "less conservative than holm for independent tests",
            "bonferroni": "more conservative option",
            "fdr_bh": "FDR control instead of FWER",
        }
    elif num_comparisons <= thresholds["hommel"]:
        method = "hommel"
        diagnostics["reason"] = (
            "moderate number of comparisons; hommel improves on holm for general dependence"  # type: ignore
        )
        diagnostics["alternatives"] = {  # type: ignore
            "fdr_bh": "FDR control with good power",
            "holm": "more conservative FWER control",
            "fdr_by": "robust FDR for dependencies",
        }
    elif num_comparisons <= thresholds["fdr_bh"]:
        method = "fdr_bh"
        diagnostics["reason"] = "moderate-large number; FDR control with fdr_bh"  # type: ignore
        diagnostics["alternatives"] = {  # type: ignore
            "fdr_by": "more robust for positive dependencies",
            "hommel": "FWER control with good power",
            "fdr_tsbh": "two-stage FDR procedure",
            "fdr_tsbky": "two-stage FDR with Hochberg step",
        }
    else:
        method = "fdr_by"
        diagnostics["reason"] = (
            "large number of comparisons; robust FDR control with fdr_by"  # type: ignore
        )
        diagnostics["alternatives"] = {  # type: ignore
            "fdr_tsbh": "two-stage FDR procedure for large numbers",
            "fdr_tsbky": "two-stage FDR with Hochberg step",
            "fdr_bh": "standard FDR (less robust to dependencies)",
            "hommel": "FWER control (more conservative)",
        }
    return method, diagnostics  # type: ignore


def check_normality(data: np.ndarray, min_size: int = 3, context: str = "") -> float:  # type: ignore
    """
    Perform Shapiro-Wilk normality test on data if sufficient samples are available.

    Parameters
    ----------
    data : np.ndarray
        Data array to test for normality.
    min_size : int, default=3
        Minimum number of samples required to perform the normality test.
    context : str, optional
        Additional context string for warning messages to provide more detail about the test.

    Returns
    -------
    float
        p-value from the Shapiro-Wilk test, or np.nan if insufficient data is provided.
    """
    if len(data) < min_size:
        return np.nan
    _, p = shapiro(data)
    if p < 0.05:
        logger.warning(f"Warning: Normality violated (p={p:.4f}) {context}.")
    return p  # type: ignore


def apply_multiple_correction(
    p_values: list[float], method: str, alpha: float = 0.05
) -> np.ndarray:  # type: ignore
    """
    Apply multiple testing correction to a list of p-values.

    Parameters
    ----------
    p_values : List[float]
        List of raw p-values to be corrected.
    method : str
        Multiple testing correction method (e.g., 'holm', 'bonferroni', 'fdr_bh', 'none').
    alpha : float, default=0.05
        Family-wise error rate for correction.

    Returns
    -------
    np.ndarray
        Array of corrected p-values, preserving np.nan for invalid entries.
    """
    p_values_array = np.array(p_values)
    if method == "none" or not p_values_array.size:
        return p_values_array
    valid_mask = ~np.isnan(p_values_array)
    corrected = np.full_like(p_values_array, np.nan)
    if valid_mask.any():
        corrected[valid_mask] = multipletests(
            p_values_array[valid_mask], alpha=alpha, method=method
        )[1]
    return corrected


def _compute_paired_pvalue(
    values1: np.ndarray,  # type: ignore
    values2: np.ndarray,  # type: ignore
    method: str,
    n_permutations: int = 5000,
    labels: tuple[str, str] | None = None,
    group_name: str | None = None,
    col: str | None = None,
) -> tuple[float, float]:
    """
    Compute p-value and normality p-value for paired test data, including method selection and logging.


    Parameters
    ----------
    values1 : np.ndarray
        First array of paired values.
    values2 : np.ndarray
        Second array of paired values.
    method : str
        Statistical test method ('auto', 'wilcoxon', 'ttest', 'permutation', 'monte_carlo_test', 'monte_carlo_test_normal', 'tukey').
    n_permutations : int, optional
        Number of resamples for permutation tests.
    labels : Tuple[str, str], optional
        Labels for the compared conditions (for logging).
    group_name : str, optional
        Group name (for logging).
    col : str, optional
        Column name (for logging).


    Returns
    -------
    Tuple[float, float]
        p_value: The p-value from the statistical test.
        norm_p: The p-value from the normality test on differences.
    """
    if len(values1) < 2 or len(values2) < 2:
        logger.warning(
            f"Insufficient data for paired test (n1={len(values1)}, n2={len(values2)})"
        )
        return np.nan, np.nan
    differences = values1 - values2
    norm_p = check_normality(
        differences,
        context=f"for difference '{labels[0]}' vs '{labels[1]}' in '{group_name}' for '{col}'"
        if labels and group_name and col
        else "for paired difference",
    )
    current_method = method
    if method == "auto":
        current_method, auto_diagnostics = _auto_select_test_method(
            data1=values1, data2=values2, test_type="paired"
        )
        if labels and group_name and col:
            logger.info(
                f"Auto-selected '{current_method}' for '{labels[0]}' vs '{labels[1]}' in '{group_name}' for '{col}' (n={auto_diagnostics.get('n_pairs', 'N/A')}, norm_p={auto_diagnostics.get('normality_p', 'N/A'):.4f}, reason={auto_diagnostics.get('reason', 'N/A')})"
            )
    if current_method == "tukey":
        data_list = [values1, values2]
        tukey_labels = [labels[0], labels[1]] if labels else ["group1", "group2"]
        p_dict = compute_tukey_pvalues(data_list, tukey_labels)
        comb = tuple(sorted(tukey_labels))
        p_value = p_dict.get(comb) or p_dict.get(comb[::-1], np.nan)  # type: ignore
    else:
        try:
            if current_method == "wilcoxon":
                _, p_value = wilcoxon(values1, values2)
            elif current_method == "ttest":
                _, p_value = ttest_rel(values1, values2)
            elif current_method == "permutation":
                res = permutation_test(
                    (values1, values2),
                    mean_diff,
                    vectorized=False,
                    permutation_type="samples",
                    alternative="two-sided",
                    n_resamples=n_permutations,
                )
                p_value = res.pvalue
            elif current_method in ["monte_carlo_test", "monte_carlo_test_normal"]:
                combined = np.concatenate([values1, values2])
                mu, sigma = np.mean(combined), np.std(combined)
                if current_method == "monte_carlo_test_normal":
                    rvs1 = rvs2 = lambda size: np.random.normal(mu, sigma, size)
                else:
                    rvs1 = rvs2 = lambda size, c=combined: np.random.choice(  # type: ignore
                        c,
                        size=size,
                        replace=True,
                    )
                res = monte_carlo_test(
                    (values1, values2),
                    statistic=mean_diff,
                    vectorized=False,
                    alternative="two-sided",
                    rvs=(rvs1, rvs2),
                )
                p_value = res.pvalue
            else:
                raise ValueError(f"Unknown paired method: {current_method}")
        except ValueError as e:
            logger.warning(f"Paired test failed: {e}")
            p_value = np.nan
    return p_value, norm_p


def _compute_unpaired_pvalue(
    values1: np.ndarray,  # type: ignore
    values2: np.ndarray,  # type: ignore
    method: str,
    n_permutations: int = 5000,
    labels: tuple[str, str] | None = None,
    group_name: str | None = None,
    col: str | None = None,
    normality: bool = False,
) -> tuple[float, float | None, float | None]:
    """
    Compute p-value and normality p-values for unpaired statistical test, including method selection and logging.


    Parameters
    ----------
    values1, values2 : np.ndarray
        Unpaired data arrays to compare.
    method : str
        Statistical test method ('auto', 'mannwhitneyu', 'ttest_ind', 'tukey').
    n_permutations : int, optional
        Number of resamples for permutation tests (unused currently).
    labels : Tuple[str, str], optional
        Labels for the compared groups (for logging).
    group_name : str, optional
        Group name (for logging).
    col : str, optional
        Column name (for logging).
    normality : bool, optional
        Whether to perform normality tests (default: False).


    Returns
    -------
    Tuple[float, Optional[float], Optional[float]]
        p_value: The p-value from the statistical test.
        norm_p1: The normality p-value for the first group (None if normality=False).
        norm_p2: The normality p-value for the second group (None if normality=False).
    """
    if len(values1) < 2 or len(values2) < 2:
        logger.warning(
            f"Insufficient data for unpaired test (n1={len(values1)}, n2={len(values2)})"
        )
        return np.nan, None, None
    norm_p1 = norm_p2 = None
    if normality:
        norm_p1 = check_normality(
            values1,
            context=f"for group '{labels[0] if labels else 'group1'}' in '{group_name}' for '{col}'"
            if labels and group_name and col
            else "for first group",
        )
        norm_p2 = check_normality(
            values2,
            context=f"for group '{labels[1] if labels else 'group2'}' in '{group_name}' for '{col}'"
            if labels and group_name and col
            else "for second group",
        )
    current_method = method
    if method == "auto":
        current_method, auto_diagnostics = _auto_select_test_method(
            data1=values1, data2=values2, test_type="unpaired"
        )
        if labels and group_name and col:
            logger.info(
                f"Auto-selected '{current_method}' for '{group_name}' between groups '{labels[0]}' (n={auto_diagnostics.get('n_group1', 'N/A')}) and '{labels[1]}' (n={auto_diagnostics.get('n_group2', 'N/A')}) for '{col}' (norm_p1={auto_diagnostics.get('normality_p_group1', 'N/A'):.4f}, norm_p2={auto_diagnostics.get('normality_p_group2', 'N/A'):.4f}, reason={auto_diagnostics.get('reason', 'N/A')})"
            )
    if current_method == "tukey":
        data_list = [values1, values2]
        tukey_labels = [labels[0], labels[1]] if labels else ["group1", "group2"]
        p_dict = compute_tukey_pvalues(data_list, tukey_labels)
        comb = tuple(sorted(tukey_labels))
        p_value = p_dict.get(comb) or p_dict.get(comb[::-1], np.nan)  # type: ignore
    else:
        try:
            if current_method == "mannwhitneyu":
                _, p_value = mannwhitneyu(values1, values2)
            elif current_method == "ttest_ind":
                _, p_value = ttest_ind(values1, values2)
            elif current_method == "permutation":
                res = permutation_test(
                    (values1, values2),
                    mean_diff,
                    vectorized=False,
                    permutation_type="samples",
                    alternative="two-sided",
                    n_resamples=n_permutations,
                )
                p_value = res.pvalue
            else:
                raise ValueError(f"Unknown unpaired method: {current_method}")
        except ValueError as e:
            logger.warning(f"Unpaired test failed: {e}")
            p_value = np.nan
    return p_value, norm_p1, norm_p2


def compute_tukey_pvalues(
    data_list: list[np.ndarray],  # type: ignore
    labels: list[str],
) -> dict[tuple[str, str], float]:
    """
    Compute p-values using Tukey's HSD test for multiple group comparisons.

    Parameters
    ----------
    data_list : List[np.ndarray]
        List of data arrays for each group.
    labels : List[str]
        Labels corresponding to each data group.

    Returns
    -------
    Dict[Tuple[str, str], float]
        Dictionary mapping pairs of labels to their corresponding p-values.
    """
    if len(labels) < 2 or any(len(d) == 0 for d in data_list):
        return {}
    all_values = np.concatenate(data_list)
    all_groups = np.concatenate(
        [([labels[i]] * len(data_list[i])) for i in range(len(labels))]
    )
    res = pairwise_tukeyhsd(all_values, all_groups, alpha=0.05)
    combs = list(combinations(sorted(labels), 2))
    return dict(zip(combs, res.pvalues))


def statistical_comparison(
    df: pd.DataFrame,
    pair_name_col: str,
    compare_by: str,
    value_col: list[str],
    test_type: Literal["paired", "unpaired", "both"] = "both",
    group_by: str | None = None,
    groups: list[str] = None,  # type: ignore
    labels: list[str] = None,  # type: ignore
    correction_method: Literal[
        "auto",
        "holm",
        "bonferroni",
        "fdr_bh",
        "sidak",
        "holm-sidak",
        "simes-hochberg",
        "hommel",
        "fdr_by",
        "fdr_tsbh",
        "fdr_tsbky",
        "none",
    ] = "auto",
    method: Literal[
        "auto",
        "wilcoxon",
        "ttest",
        "mannwhitneyu",
        "ttest_ind",
        "permutation",
        "monte_carlo_test",
        "monte_carlo_test_normal",
        "tukey",
    ] = "auto",
    n_permutations: int = 5000,
    normality: bool = True,
) -> dict[str, dict[str, pd.DataFrame]] | dict[str, dict[str, dict[str, pd.DataFrame]]]:
    """
    Unified statistical comparison function supporting paired and unpaired tests with
    integrated heatmap and violin plot visualizations.

    This function consolidates paired and unpaired statistical testing into a single
    interface with consistent multiple testing correction logic. It automatically
    generates p-value heatmaps and interactive violin plots with significance annotations.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame in long format.
    pair_name_col : str
        Column identifying paired samples (e.g., 'animal_id', 'subject_id').
    compare_by : str
        Column containing conditions/categories to compare (e.g., 'task', 'timepoint').
    value_col : List[str]
        List of dependent variable column names to test.
    test_type : {"paired", "unpaired", "both"}, default="both"
        Type of statistical test to perform.
    group_by : str, optional
        Column to group data by.
    groups : List[str], optional
        Specific groups to analyze.
    labels : List[str], optional
        Specific conditions to compare.
        Custom plot title.
        Additional text appended to titles.
        Directory to save plots.
        Generate p-value heatmaps.
        Generate violin plots with significance annotations.
        Save static plots as PDF (True) or PNG (False).
    correction_method : str, default="auto"
        Multiple testing correction method.
    method : str, default="auto"
        Statistical test method.
    n_permutations : int, default=5000
        Number of permutations for permutation/Monte Carlo tests.
    normality : bool, default=True
        Perform Shapiro-Wilk normality test.

    Returns
    -------
    Dict[str, Dict[str, pd.DataFrame]] or Dict[str, Dict[str, Dict[str, pd.DataFrame]]]
        Results structure depends on test_type and normality.
    """
    value_col = make_list_ifnot(value_col)
    test_types = [test_type] if isinstance(test_type, str) else list(test_type)
    paired_methods = [
        "wilcoxon",
        "ttest",
        "permutation",
        "monte_carlo_test",
        "monte_carlo_test_normal",
        "tukey",
    ]
    unpaired_methods = ["mannwhitneyu", "ttest_ind", "tukey"]
    results = {}
    paired_results = None
    unpaired_results = None
    unpaired_possible = True
    if "paired" in test_types:
        logger.info(f"Running paired statistical tests using {method}...")
        paired_method = method if method in paired_methods else "wilcoxon"
        if method not in paired_methods and method != "auto":
            logger.warning(
                f"Method '{method}' not valid for paired tests. Using 'wilcoxon'."
            )
        paired_results = _run_statistical_tests(
            df=df,
            pair_name_col=pair_name_col,
            compare_by=compare_by,
            value_col=value_col,
            group_by=group_by,
            groups=groups,
            labels=labels,
            method=paired_method,
            correction_method=correction_method,
            n_permutations=n_permutations,
            normality=normality,
            paired=True,
        )
        results["paired"] = paired_results
    if "unpaired" in test_types:
        if group_by is None:
            logger.warning(
                "Unpaired test requested but group_by is None. Skipping unpaired test."
            )
            unpaired_possible = False
        else:
            unique_groups = df[group_by].unique() if groups is None else groups
            if len(unique_groups) < 2:
                logger.warning(
                    f"Unpaired test requested but only {len(unique_groups)} group(s) present. Skipping unpaired test."
                )
                unpaired_possible = False
        if unpaired_possible:
            logger.info(f"Running unpaired statistical tests using {method}...")
            unpaired_method = method if method in unpaired_methods else "mannwhitneyu"
            if method not in unpaired_methods and method != "auto":
                logger.warning(
                    f"Method '{method}' not valid for unpaired tests. Using 'mannwhitneyu'."
                )
            unpaired_results = _run_statistical_tests(
                df=df,
                pair_name_col=pair_name_col,
                compare_by=compare_by,
                value_col=value_col,
                group_by=group_by,
                groups=groups,
                labels=labels,
                method=unpaired_method,
                correction_method=correction_method,
                n_permutations=n_permutations,
                normality=normality,
                paired=False,
            )
            results["unpaired"] = unpaired_results
    return results  # type: ignore


def _initialize_heatmaps(
    value_col: list[str],
    groups: list[str],
    labels: list[str],
    normality: bool,
    paired: bool,
) -> tuple[
    dict[str, dict[str, pd.DataFrame]], dict[str, dict[str, pd.DataFrame]] | None
]:
    """
    Initialize heatmaps for statistical tests.

    Parameters
    ----------
    value_col : List[str]
        List of dependent variable column names.
    groups : List[str]
        List of group names to analyze.
    labels : List[str]
        List of condition labels to compare.
    normality : bool
        Whether to include normality test results.
    paired : bool
        Whether the test is paired or unpaired.

    Returns
    -------
    Tuple[Dict[str, Dict[str, pd.DataFrame]], Optional[Dict[str, Dict[str, pd.DataFrame]]]]
        test_heatmaps: Dictionary of p-value DataFrames.
        normality_heatmaps: Dictionary of normality test DataFrames (if normality is True).
    """
    if paired:
        test_heatmaps = {
            col: {
                group: pd.DataFrame(np.nan, index=labels, columns=labels)
                for group in groups
            }
            for col in value_col
        }
        normality_heatmaps = None
        if normality:
            normality_heatmaps = {
                col: {
                    group: pd.DataFrame(np.nan, index=labels, columns=labels)
                    for group in groups
                }
                for col in value_col
            }
    else:
        test_heatmaps = {
            col: pd.DataFrame(np.nan, index=labels, columns=[col])  # type: ignore
            for col in value_col
        }
        normality_heatmaps = None
        if normality:
            normality_heatmaps = {
                col: pd.DataFrame(np.nan, index=labels, columns=groups)  # type: ignore
                for col in value_col
            }
    return test_heatmaps, normality_heatmaps


def _run_statistical_tests(
    df: pd.DataFrame,
    pair_name_col: str,
    compare_by: str,
    value_col: list[str],
    group_by: str | None,
    groups: list[str] | None,
    labels: list[str] | None,
    method: str,
    correction_method: str,
    n_permutations: int,
    normality: bool,
    paired: bool,
) -> dict[str, dict[str, pd.DataFrame]] | dict[str, dict[str, dict[str, pd.DataFrame]]]:
    """
    Run statistical tests (paired or unpaired) with multiple testing correction.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame in long format.
    pair_name_col : str
        Column identifying samples/pairs.
    compare_by : str
        Column containing conditions to compare.
    value_col : List[str]
        List of dependent variable column names.
    group_by : str, optional
        Column to group data by.
    groups : List[str], optional
        Specific groups to analyze.
    labels : List[str], optional
        Specific conditions to compare.
    method : str
        Statistical test method.
    correction_method : str
        Multiple testing correction method.
    n_permutations : int
        Number of resamples for permutation tests.
    normality : bool
        Whether to perform normality tests.
    paired : bool
        Whether to perform paired or unpaired tests.

    Returns
    -------
    Union[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, Dict[str, pd.DataFrame]]]]
        Dictionary containing p-values and optionally normality test results.
    """
    labels = df[compare_by].unique() if labels is None else labels  # type: ignore
    groups = (
        ["all"]  # type: ignore
        if group_by is None
        else df[group_by].unique()
        if groups is None
        else groups
    )
    if not paired and len(groups) != 2:  # type: ignore
        raise ValueError(
            f"Unpaired tests require exactly 2 groups. Got {len(groups)}: {groups}"  # type: ignore
        )
    num_value_cols = len(value_col)
    groups_count = len(groups)  # type: ignore
    labels_count = len(labels)  # type: ignore
    num_comparisons = (
        num_value_cols * groups_count * (labels_count * (labels_count - 1) // 2)
        if paired
        else num_value_cols * labels_count
    )
    selected_correction_method = correction_method
    if correction_method == "auto":
        selected_correction_method, auto_diagnostics = _auto_select_correction_method(
            num_comparisons, test_type="paired" if paired else "unpaired"
        )
        logger.info(
            f"Auto-selected correction method '{selected_correction_method}' for {'paired' if paired else 'unpaired'} tests (estimated {num_comparisons} comparisons): {auto_diagnostics['reason']}"
        )
    test_heatmaps, normality_heatmaps = _initialize_heatmaps(
        value_col,
        groups,  # type: ignore
        labels,  # type: ignore
        normality,
        paired,
    )
    for group_name in groups:  # type: ignore
        group_df = df if group_by is None else df[df[group_by] == group_name]
        if paired:
            group_df = group_df.sort_values(by=pair_name_col)
        all_p_values = []
        all_comparisons = []
        if paired:
            combs = list(combinations(labels, 2))  # type: ignore
            for label1, label2 in combs:
                group_label1_df = group_df[group_df[compare_by] == label1]
                group_label2_df = group_df[group_df[compare_by] == label2]
                if group_label1_df.empty or group_label2_df.empty:
                    for col in value_col:
                        all_p_values.append(np.nan)
                        all_comparisons.append((col, label1, label2))
                    continue
                common_names = set(group_label1_df[pair_name_col]) & set(
                    group_label2_df[pair_name_col]
                )
                if not common_names or len(common_names) < 2:
                    for col in value_col:
                        all_p_values.append(np.nan)
                        all_comparisons.append((col, label1, label2))
                    continue
                for col in value_col:
                    values1 = group_label1_df[
                        group_label1_df[pair_name_col].isin(common_names)
                    ][col].values
                    values2 = group_label2_df[
                        group_label2_df[pair_name_col].isin(common_names)
                    ][col].values
                    p_value, norm_p = _compute_paired_pvalue(
                        values1,  # type: ignore
                        values2,  # type: ignore
                        method,
                        n_permutations,
                        (label1, label2),
                        group_name,
                        col,
                    )
                    if normality:
                        normality_heatmaps[col][group_name].loc[label1, label2] = norm_p  # type: ignore
                        normality_heatmaps[col][group_name].loc[label2, label1] = norm_p  # type: ignore
                    all_p_values.append(p_value)
                    all_comparisons.append((col, label1, label2))
        else:
            for col in value_col:
                for label in labels:  # type: ignore
                    label_df = group_df[group_df[compare_by] == label]
                    groups_data = [
                        label_df[label_df[group_by] == g][col].dropna().values
                        for g in groups  # type: ignore
                    ]
                    if any(len(g) == 0 for g in groups_data):
                        all_p_values.append(np.nan)
                        all_comparisons.append((col, label))  # type: ignore
                        continue
                    p_value, norm_p1, norm_p2 = _compute_unpaired_pvalue(
                        groups_data[0],  # type: ignore
                        groups_data[1],  # type: ignore
                        method,
                        groups,  # type: ignore
                        label,  # type: ignore
                        col,
                        normality,  # type: ignore
                    )
                    if normality:
                        normality_heatmaps[col].loc[label, groups[0]] = norm_p1  # type: ignore
                        normality_heatmaps[col].loc[label, groups[1]] = norm_p2  # type: ignore
                    all_p_values.append(p_value)
                    all_comparisons.append((col, label))  # type: ignore
        corrected_p = apply_multiple_correction(
            all_p_values, selected_correction_method
        )
        if paired:
            for idx, (col, label1, label2) in enumerate(all_comparisons):
                test_heatmaps[col][group_name].loc[label1, label2] = corrected_p[idx]
                test_heatmaps[col][group_name].loc[label2, label1] = corrected_p[idx]
        else:
            for idx, (col, label) in enumerate(all_comparisons):  # type: ignore
                test_heatmaps[col].loc[label, col] = corrected_p[idx]  # type: ignore
    if normality:
        return {"pvalues": test_heatmaps, "normality": normality_heatmaps}  # type: ignore
    return test_heatmaps


def mannwhitneyu_cross_df(
    df: pd.DataFrame,
    group_by: str,
    compare_by: str,
    value_col: list[str],
    groups: list[str] = None,  # type: ignore
    labels: list[str] = None,  # type: ignore
) -> dict[str, pd.DataFrame]:
    """
    Perform Mann-Whitney U test across specified groups in a DataFrame for multiple value columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to analyze.
    group_by : str
        Column name to group data by (e.g., 'group').
    compare_by : str
        Column name to compare within groups (e.g., 'condition').
    value_col : str
        Column name containing values to test (e.g., 'intercept').
    groups : List[str], optional
        List of group names to analyze. If None, uses unique values in group_by.
    labels : List[str], optional
        List of labels for compare_by. If None, uses unique values in compare_by.
        Title for plots. If None, generated automatically.
        Additional text for plot titles.
        Directory to save plots.
        Whether to generate heatmaps of p-values.
        Save plots as PDF if True.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with group names as keys and DataFrames of p-values as values.
        Each DataFrame contains p-values for each label pair.
    """
    value_col = make_list_ifnot(value_col)
    labels = df[compare_by].unique() if labels is None else labels
    groups = df[group_by].unique() if groups is None else groups
    if len(groups) != 2:
        do_critical(
            ValueError,  # type: ignore
            f"Mann-Whitney U test requires exactly two groups. Got {len(groups)} groups: {groups}.",
        )
    pvalues_df = pd.DataFrame(index=labels, columns=value_col, dtype=float)
    for label in labels:
        if label not in df[compare_by].unique():
            do_critical(
                ValueError,  # type: ignore
                f"Label '{label}' not found in column '{compare_by}'.",
            )
        label_df = df[df[compare_by] == label]
        for col in value_col:
            if col not in label_df.columns:
                do_critical(ValueError, f"Column '{col}' not found in DataFrame.")  # type: ignore
            g1 = label_df[label_df[group_by] == groups[0]][col].values
            g2 = label_df[label_df[group_by] == groups[1]][col].values
            stat, p_value = mannwhitneyu(g1, g2)
            pvalues_df.loc[label, col] = p_value
        if len(value_col) > 1:
            p_values = pvalues_df.loc[label].values
            valid_mask = ~np.isnan(p_values)  # type: ignore
            if valid_mask.any():
                corrected_p = multipletests(
                    p_values[valid_mask], method="holm", alpha=0.05
                )[1]
                pvalues_df.loc[label, value_col] = np.where(
                    valid_mask, corrected_p, np.nan
                )
            else:
                pvalues_df.loc[label, value_col] = np.nan
    return pvalues_df  # type: ignore


def sigtest_cross_df(
    df: pd.DataFrame,
    pair_name_col: str,
    compare_by: str,
    value_col: list[str],
    group_by: str | None = None,
    groups: list[str] = None,  # type: ignore
    labels: list[str] = None,  # type: ignore
    correction_method: Literal["holm", "bonferroni", "fdr_bh", "none"] = "holm",
    method: Literal[
        "wilcoxon",
        "ttest",
        "permutation",
        "monte_carlo_test",
        "monte_carlo_test_normal",
        "tukey",
    ] = "wilcoxon",
    n_permutations: int = 5000,
    normality: bool = True,
) -> dict[str, dict[str, pd.DataFrame]] | dict[str, dict[str, dict[str, pd.DataFrame]]]:
    """Performs paired statistical tests across groups and conditions in a DataFrame.
        This function automates running paired statistical tests (e.g., Wilcoxon,
        paired t-test) on long-form data. It can operate on the entire dataset
        or on distinct subgroups. For each pairwise comparison of conditions
        (e.g., 'pre' vs. 'post'), it runs the test for multiple dependent
        variables (`value_col`).
        A key feature is the application of multiple testing correction (e.g., Holm,
        Bonferroni) across the p-values obtained from the different `value_col`
        variables for each specific condition pair. The results can be optionally
        visualized as heatmaps of p-values.
        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame in a long format.
        pair_name_col : str
            The column name that identifies paired samples, such as a subject ID.
        compare_by : str
            The column name containing the conditions to compare (e.g., 'treatment').
        value_col : List[str]
            A list of column names for the dependent variables to be tested.
        group_by : str, optional
            The column name to group the DataFrame by. If provided, tests are run
            independently for each group. If None, the test is run on the entire
            DataFrame, by default None.
        groups : List[str], optional
            A specific list of groups from the `group_by` column to analyze. If
            None, all unique groups are used, by default None.
        labels : List[str], optional
            A specific list of conditions from the `compare_by` column to compare.
            If None, all unique conditions are used, by default None.
            A custom title for the generated plots. If None, a title is
            automatically generated, by default None.
            Additional text to append to the plot title, by default None.
            The directory path to save the plots. If None, plots are not saved to
            disk, by default None.
            If True, generates and displays heatmaps of the resulting p-values,
            by default False.
            If True, saves the plots in PDF format. Relevant only if `plot` is
            True and `save_dir` is provided, by default False.
        correction_method : {"holm", "bonferroni", "fdr_bh", "none"}, optional
            The method for multiple testing correction applied across `value_col`
            for each pair of conditions.
            - "none": No correction is applied.
            - "holm": Holm-Bonferroni method.
            - "bonferroni": Bonferroni correction.
            - "fdr_bh": Benjamini/Hochberg for FDR control.
            - `sidak` : one-step correction
            - `holm-sidak` : step down method using Sidak adjustments
            - `simes-hochberg` : step-up method  (independent)
            - `hommel` : closed method based on Simes tests (non-negative)
            - `fdr_by` : Benjamini/Yekutieli (negative)
            - `fdr_tsbh` : two stage fdr correction (non-negative)
            - `fdr_tsbky` : two stage fdr correction (non-negative)
            Defaults to "holm".
        method : {"wilcoxon", "ttest", "permutation", "monte_carlo_test"}, optional
            The statistical test to perform.
            - "wilcoxon": Wilcoxon signed-rank test.
            - "ttest": Paired t-test.
            - "permutation": Permutation test on the mean difference.
            - "monte_carlo_test": Monte Carlo test (resampling).
            - "monte_carlo_test_normal": Monte Carlo test assuming normality.
            - "tukey": Tukey's Honestly Significant Difference test (not paired).
            Defaults to "wilcoxon".
        n_permutations : int, optional
            The number of resampling permutations to perform for the permutation
            test, by default 5000. Ignored for other methods.
        normality : bool, optional
            If True, performs Shapiro-Wilk test for normality on the differences (for paired tests)
            or on each group (for unpaired tests like Tukey) and returns the p-values in a separate
            structure. Defaults to False.
        Returns
        -------
        Dict[str, Dict[str, pd.DataFrame]] or Dict[str, Dict[str, Dict[str, pd.DataFrame]]]
            If normality is False, returns a nested dictionary containing the p-value results.
            The structure is: `{group_name: {value_col_name: p_value_dataframe}}`.
            - The outer keys are group names from the `group_by` column (or "all"
              if `group_by` is None).
            - The inner keys are the names of the dependent variables from `value_col`.
            - Each value is a DataFrame where the index and columns are the
              conditions from `compare_by`, and the cells contain the corrected
              p-values.
            If normality is True, returns a dictionary with keys 'pvalues' and 'normality',
            each containing the above structure (normality contains Shapiro-Wilk p-values).
        Notes
        -----
        - The multiple testing correction is applied as follows: for a given group
          (e.g., 'Group A') and a specific comparison of conditions (e.g.,
          'baseline' vs. 'treatment'), the p-values from the tests on `value_col`
          ['score1', 'score2', ...] are corrected together as a single family of
          tests.
        - The function requires that for any given condition and group, each
          identifier in `pair_name_col` is unique.
        - For a test to be performed between two conditions, there must be at least
          two common identifiers in `pair_name_col` present in both conditions.
        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> # 1. Create a sample DataFrame
        >>> data = {
        ...     'subject_id': ['s1', 's1', 's2', 's2', 's3', 's3', 's4', 's4'] * 2,
        ...     'group': ['A'] * 8 + ['B'] * 8,
        ...     'condition': ['baseline', 'treatment'] * 8,
        ...     'score1': np.random.randn(16) + np.tile([0, 1], 8),
        ...     'score2': np.random.randn(16) + np.tile([10, 8], 8)
        ... }
        >>> df = pd.DataFrame(data)
        >>>
        >>> # 2. Run the statistical analysis
        >>> results = sigtest_cross_df(
        ...     df=df,
        ...     pair_name_col='subject_id',
        ...     compare_by='condition',
        ...     value_col=['score1', 'score2'],
        ...     group_by='group',
        ...     correction_method='holm',
        ...     method='wilcoxon'
        ... )
        >>>
        >>> # 3. Print the results for one group
        >>> print("Results for Group A:")
        >>> for value_name, p_value_df in results['A'].items():
        ...     print(f"
    --- P-values for {value_name} ---")
        ...     print(p_value_df)
        Results for Group A:
        --- P-values for score1 ---
                  baseline  treatment
        baseline        NaN   0.093799
        treatment   0.093799        NaN
        --- P-values for score2 ---
                  baseline  treatment
        baseline        NaN   0.093799
        treatment   0.093799        NaN
    """
    value_col = make_list_ifnot(value_col)
    {
        "wilcoxon": "Wilcoxon Signed-Rank",
        "ttest": "Paired t",
        "permutation": "Permutation",
        "monte_carlo_test": "Monte Carlo",
        "monte_carlo_test_normal": "Monte Carlo Normal",
        "tukey": "Tukey HSD",
    }[method]
    labels = df[compare_by].unique() if labels is None else labels
    groups = (
        ["all"]
        if group_by is None
        else df[group_by].unique()
        if groups is None
        else groups
    )
    test_heatmaps = {col: dict.fromkeys(groups) for col in value_col}
    if normality:
        normality_heatmaps = {
            col: {
                group_name: pd.DataFrame(np.nan, index=labels, columns=labels)
                for group_name in groups
            }
            for col in value_col
        }
    num_animals_dict = {
        group_name: pd.DataFrame(0, index=labels, columns=labels, dtype=int)
        for group_name in groups
    }
    for group_name in groups:
        group_df = df if group_by is None else df[df[group_by] == group_name]
        group_df = group_df.sort_values(by=pair_name_col)
        for col in value_col:
            test_df = pd.DataFrame(index=labels, columns=labels, dtype=float)
            test_df = test_df.fillna(np.nan)
            test_heatmaps[col][group_name] = test_df
        all_p_values = []
        all_comparisons = []
        if method == "tukey":
            tukey_results = {}  # type: ignore
            if normality:
                for col in value_col:
                    for label in labels:
                        values = (
                            group_df[group_df[compare_by] == label][col].dropna().values
                        )
                        if len(values) >= 3:
                            _, norm_p = shapiro(values)
                            normality_heatmaps[col][group_name].loc[label, label] = (
                                norm_p
                            )
                            if norm_p < 0.05:
                                logger.warning(
                                    f"Warning: Normality test p-value {norm_p:.4f} < 0.05 for difference between '{label}' and its baseline in group '{group_name}' for column '{col}'. Consider using a non-parametric test."
                                )
            for col in value_col:
                labels_sorted = np.sort(labels)
                groups_data = [
                    group_df[group_df[compare_by] == label][col].dropna().values
                    for label in labels_sorted
                ]
                if len(labels_sorted) < 2 or any(len(g) == 0 for g in groups_data):
                    tukey_results[col] = None  # type: ignore
                    continue
                all_values = np.concatenate(groups_data)
                all_groups = np.concatenate(
                    [
                        ([labels_sorted[i]] * len(groups_data[i]))
                        for i in range(len(groups_data))
                    ]
                )
                res = pairwise_tukeyhsd(all_values, all_groups, alpha=0.05)
                p_matrix = pd.DataFrame(
                    np.nan, index=labels_sorted, columns=labels_sorted
                )
                combs = list(combinations(labels_sorted, 2))
                for (l1, l2), p in zip(combs, res.pvalues):
                    p_matrix.loc[l1, l2] = p
                    p_matrix.loc[l2, l1] = p
                    all_p_values.append(p)
                    all_comparisons.append((col, l1, l2))
                tukey_results[col] = p_matrix
        else:
            combs = list(combinations(labels, 2))
            for label1, label2 in combs:
                group_label1_df = group_df[group_df[compare_by] == label1]
                group_label2_df = group_df[group_df[compare_by] == label2]
                if group_label1_df.empty or group_label2_df.empty:
                    continue
                common_names = set(group_label1_df[pair_name_col]) & set(
                    group_label2_df[pair_name_col]
                )
                num_animals_dict[group_name].loc[label1, label2] = len(common_names)
                num_animals_dict[group_name].loc[label2, label1] = len(common_names)
                if not common_names or len(common_names) < 2:
                    for col in value_col:
                        all_p_values.append(np.nan)
                        all_comparisons.append((col, label1, label2))
                    continue
                for common_name in common_names:
                    if (
                        group_label1_df[
                            group_label1_df[pair_name_col] == common_name
                        ].shape[0]
                        != 1
                    ):
                        do_critical(
                            ValueError,  # type: ignore
                            f"Column '{pair_name_col}' has more than one unique value in group '{label1}' for common name '{common_name}'.",
                        )
                    if (
                        group_label2_df[
                            group_label2_df[pair_name_col] == common_name
                        ].shape[0]
                        != 1
                    ):
                        do_critical(
                            ValueError,  # type: ignore
                            f"Column '{pair_name_col}' has more than one unique value in group '{label2}' for common name '{common_name}'.",
                        )
                for col in value_col:
                    values1 = group_label1_df[
                        group_label1_df[pair_name_col].isin(common_names)
                    ][col].values
                    values2 = group_label2_df[
                        group_label2_df[pair_name_col].isin(common_names)
                    ][col].values
                    if normality:
                        differences = values1 - values2
                        if len(differences) >= 3:
                            _, norm_p = shapiro(differences)
                            normality_heatmaps[col][group_name].loc[label1, label2] = (
                                norm_p
                            )
                            normality_heatmaps[col][group_name].loc[label2, label1] = (
                                norm_p
                            )
                            if norm_p < 0.05:
                                logger.warning(
                                    f"Warning: Normality test p-value {norm_p:.4f} < 0.05 for difference between '{label}' and its baseline in group '{group_name}' for column '{col}'. Consider using a non-parametric test."
                                )
                        else:
                            normality_heatmaps[col][group_name].loc[label1, label2] = (
                                np.nan
                            )
                            normality_heatmaps[col][group_name].loc[label2, label1] = (
                                np.nan
                            )
                    try:
                        if method == "wilcoxon":
                            _, p_value = wilcoxon(values1, values2)
                        elif method == "ttest":
                            p_value = ttest_rel(values1, values2).pvalue
                        elif method == "permutation":
                            res = permutation_test(
                                (values1, values2),
                                mean_diff,
                                vectorized=False,
                                permutation_type="samples",
                                alternative="two-sided",
                                n_resamples=n_permutations,
                            )
                            p_value = res.pvalue
                        elif (
                            method == "monte_carlo_test"
                            or method == "monte_carlo_test_normal"
                        ):
                            combined = np.concatenate([values1, values2])
                            mu, sigma = np.mean(combined), np.std(combined)
                            if "normal" in method:
                                rvs = (
                                    lambda size, m=mu, s=sigma: np.random.normal(
                                        m, s, size=size
                                    ),
                                    lambda size, m=mu, s=sigma: np.random.normal(
                                        m, s, size=size
                                    ),
                                )
                            else:
                                rvs = (
                                    lambda size, c=combined: np.random.choice(  # type: ignore
                                        c,
                                        size=size,
                                        replace=True,
                                    ),
                                    lambda size, c=combined: np.random.choice(  # type: ignore
                                        c,
                                        size=size,
                                        replace=True,
                                    ),
                                )
                            res = monte_carlo_test(
                                data=(values1, values2),
                                rvs=rvs,
                                statistic=mean_diff,
                                vectorized=False,
                                alternative="two-sided",
                            )
                            p_value = res.pvalue
                        else:
                            raise ValueError(f"Unknown method: {method}")
                        all_p_values.append(p_value)
                        all_comparisons.append((col, label1, label2))
                    except ValueError:
                        all_p_values.append(np.nan)
                        all_comparisons.append((col, label1, label2))
        if correction_method != "none" and all_p_values:
            all_p_values = np.array(all_p_values)  # type: ignore
            valid_mask = ~np.isnan(all_p_values)
            corrected_p = np.full(all_p_values.shape, np.nan)  # type: ignore
            if valid_mask.any():
                corrected_p[valid_mask] = multipletests(
                    all_p_values[valid_mask], method=correction_method, alpha=0.05
                )[1]
            for idx, (col, label1, label2) in enumerate(all_comparisons):
                test_heatmaps[col][group_name].loc[label1, label2] = corrected_p[idx]  # type: ignore
                test_heatmaps[col][group_name].loc[label2, label1] = corrected_p[idx]  # type: ignore
        else:
            for idx, (col, label1, label2) in enumerate(all_comparisons):
                test_heatmaps[col][group_name].loc[label1, label2] = all_p_values[idx]  # type: ignore
                test_heatmaps[col][group_name].loc[label2, label1] = all_p_values[idx]  # type: ignore
    if normality:
        return {"pvalues": test_heatmaps, "normality": normality_heatmaps}  # type: ignore
    return test_heatmaps  # type: ignore
