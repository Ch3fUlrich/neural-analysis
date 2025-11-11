from typing import List, Literal, Dict, Union, Optional, Callable, Tuple
from Helper import *
from Visualizer import Vizualizer, violin_plot

from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from statsmodels.stats.multitest import multipletests
from scipy.stats import (
    wilcoxon,
    ttest_rel,
    ttest_ind,
    permutation_test,
    mannwhitneyu,
    probplot,
    monte_carlo_test,
    shapiro,
)
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations

from matplotlib import colors as mcolors

################# Statistical Tests ########################
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


import numpy as np
from typing import Optional, Tuple, Dict, Literal
from scipy.stats import shapiro, levene


def _auto_select_test_method(
    data1: np.ndarray,
    data2: Optional[np.ndarray] = None,
    test_type: Literal["paired", "unpaired"] = "paired",
    alpha_normality: float = 0.05,
    min_sample_size_parametric: int = 20,
) -> Tuple[str, Dict[str, float]]:
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
    # Input validation
    if isinstance(data1, pd.DataFrame) or isinstance(data1, pd.Series):
        data1 = data1.to_numpy()
    if data2 is not None and (
        isinstance(data2, pd.DataFrame) or isinstance(data2, pd.Series)
    ):
        data2 = data2.to_numpy()
    if not isinstance(data1, np.ndarray) or (
        data2 is not None and not isinstance(data2, np.ndarray)
    ):
        raise TypeError("data1 and data2 must be numpy arrays")
    if len(data1) == 0:
        raise ValueError("data1 cannot be empty")
    if test_type == "unpaired" and data2 is None:
        raise ValueError("data2 must be provided for unpaired tests")
    if test_type == "unpaired" and len(data2) == 0:
        raise ValueError("data2 cannot be empty for unpaired tests")
    if test_type == "paired" and data2 is not None and len(data1) != len(data2):
        raise ValueError("data1 and data2 must have the same length for paired tests")

    diagnostics = {
        "alpha_normality": alpha_normality,
        "min_sample_size_parametric": min_sample_size_parametric,
        "assumptions_note": (
            "Parametric: normality (and equal variances for ttest_ind). "
            "Wilcoxon: symmetric differences. Mann-Whitney U: similar shapes."
        ),
    }

    if test_type == "paired":
        # For paired tests, check normality of differences
        if data2 is not None:
            differences = data1 - data2
        else:
            differences = data1
            diagnostics["note"] = "data1 treated as differences since data2 is None"

        n = len(differences)
        diagnostics["n_pairs"] = n

        # Check normality if sufficient sample size
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

        # Select test based on sample size and normality
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

        # Add warning for large samples
        if n > 50:
            diagnostics["warning"] = (
                "Shapiro-Wilk may be overly sensitive for large samples (>50). Consider visual checks."
            )

    else:  # unpaired
        n1 = len(data1)
        n2 = len(data2)
        diagnostics["n_group1"] = n1
        diagnostics["n_group2"] = n2

        # Check normality for both groups
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

        # Both groups should be normal for parametric test
        both_normal = is_normal1 and is_normal2
        sufficient_size = (
            n1 >= min_sample_size_parametric and n2 >= min_sample_size_parametric
        )

        # Select test
        if min(n1, n2) < 3:
            method = "mannwhitneyu"
            diagnostics["reason"] = "insufficient_sample_size"
        elif sufficient_size and both_normal:
            # Check variance equality
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

        # Add warning for large samples
        if max(n1, n2) > 50:
            diagnostics["warning"] = (
                "Shapiro-Wilk may be overly sensitive for large samples (>50). Consider visual checks."
            )

    return method, diagnostics


def _auto_select_correction_method(
    num_comparisons: int,
    test_type: Literal["paired", "unpaired"] = "paired",
    alpha: float = 0.05,
) -> Tuple[
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
    Dict[str, any],
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

    # Adjust thresholds based on test_type: paired tests are more conservative
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
    else:  # unpaired
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

    # Selection logic with expanded options
    if num_comparisons <= thresholds["bonferroni"]:
        method = "bonferroni"
        diagnostics["reason"] = (
            "very small number of comparisons; conservative bonferroni correction"
        )
        diagnostics["alternatives"] = {
            "sidak": "similar to bonferroni but slightly less conservative",
            "holm": "step-down procedure for small numbers",
            "none": "no correction (not recommended for multiple comparisons)",
        }
    elif num_comparisons <= thresholds["holm"]:
        method = "holm"
        diagnostics["reason"] = (
            "small number of comparisons; holm provides good balance of power and control"
        )
        diagnostics["alternatives"] = {
            "holm-sidak": "combination of holm and sidak procedures",
            "simes-hochberg": "less conservative than holm for independent tests",
            "bonferroni": "more conservative option",
            "fdr_bh": "FDR control instead of FWER",
        }
    elif num_comparisons <= thresholds["hommel"]:
        method = "hommel"
        diagnostics["reason"] = (
            "moderate number of comparisons; hommel improves on holm for general dependence"
        )
        diagnostics["alternatives"] = {
            "fdr_bh": "FDR control with good power",
            "holm": "more conservative FWER control",
            "fdr_by": "robust FDR for dependencies",
        }
    elif num_comparisons <= thresholds["fdr_bh"]:
        method = "fdr_bh"
        diagnostics["reason"] = "moderate-large number; FDR control with fdr_bh"
        diagnostics["alternatives"] = {
            "fdr_by": "more robust for positive dependencies",
            "hommel": "FWER control with good power",
            "fdr_tsbh": "two-stage FDR procedure",
            "fdr_tsbky": "two-stage FDR with Hochberg step",
        }
    else:
        method = "fdr_by"
        diagnostics["reason"] = (
            "large number of comparisons; robust FDR control with fdr_by"
        )
        diagnostics["alternatives"] = {
            "fdr_tsbh": "two-stage FDR procedure for large numbers",
            "fdr_tsbky": "two-stage FDR with Hochberg step",
            "fdr_bh": "standard FDR (less robust to dependencies)",
            "hommel": "FWER control (more conservative)",
        }

    return method, diagnostics


def mixedlm_learning_curve_analysis(
    df,
    additional_title="",
    subject_col="animal_id",
    outcome_col="auc",
    group_col="condition",
    time_col="task_name",  # renamed for clarity - could be day, session, etc.
    order_timepoints=True,
    plot_predicted=True,
    use_reml=True,
    show_diagnostics=False,
):
    """
    Fits a linear mixed-effects model testing group differences in learning curves
    and visualizes data and statistical significance.

    Parameters:
    - df: pandas DataFrame with columns for subject, outcome, group, time
    - additional_title: additional title for the plot (default "")
    - subject_col: subject/animal ID column name (default 'animal_id')
    - outcome_col: outcome (dependent variable) column name (default 'auc')
    - group_col: group/condition column name (default 'condition')
    - time_col: time/session/task column name (default 'task_name')
    - order_timepoints: assign numeric order to timepoints (default True)
    - plot_predicted: plot model-predicted means (default True)
    - use_reml: use REML estimation (True) vs ML (False) (default True)
    - show_diagnostics: show diagnostic plots (default False)

    Returns:
    - fitted model result object
    """

    # Input validation
    required_cols = [subject_col, outcome_col, group_col, time_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    ddata = df.copy()

    # Remove any missing data
    initial_n = len(ddata)
    ddata = ddata.dropna(subset=required_cols)
    if len(ddata) < initial_n:
        print(f"Removed {initial_n - len(ddata)} rows with missing data")

    # Time/task numbering
    if order_timepoints:
        time_order = sorted(ddata[time_col].unique())
        time_map = {t: i + 1 for i, t in enumerate(time_order)}
        ddata["task_number"] = ddata[time_col].map(time_map)
        print(
            f"Time points ordered as: {dict(zip(time_order, range(1, len(time_order)+1)))}"
        )
    else:
        ddata["task_number"] = ddata[time_col].astype(float)

    # Center task number for better interpretation
    task_mean = ddata["task_number"].mean()
    ddata["task_number_c"] = ddata["task_number"] - task_mean

    # Group coding as categorical
    ddata[group_col] = ddata[group_col].astype("category")
    ddata["condition_num"] = ddata[group_col].cat.codes
    group_labels = ddata[group_col].cat.categories.tolist()

    # Print sample information
    print(f"\nSample Summary:")
    print(f"- Total observations: {len(ddata)}")
    print(f"- Subjects: {ddata[subject_col].nunique()}")
    print(f"- Time points: {ddata['task_number'].nunique()}")
    print(f"- Groups: {group_labels}")
    print(f"- Group sizes: {ddata[group_col].value_counts().to_dict()}")

    # Fit the Linear Mixed-Effects Model
    try:
        model = smf.mixedlm(
            f"{outcome_col} ~ condition_num + task_number_c + condition_num:task_number_c",
            ddata,
            groups=ddata[subject_col],
            re_formula="~task_number_c",  # Random intercepts and slopes
        )
        result = model.fit(reml=use_reml, method="lbfgs")

    except Exception as e:
        print(f"Model fitting failed: {e}")
        print("Trying with random intercepts only...")
        model = smf.mixedlm(
            f"{outcome_col} ~ condition_num + task_number_c + condition_num:task_number_c",
            ddata,
            groups=ddata[subject_col],
            re_formula="~1",  # Random intercepts only
        )
        result = model.fit(reml=use_reml, method="lbfgs")

    # Print model summary
    print(f"\n{'='*60}")
    print("LINEAR MIXED-EFFECTS MODEL RESULTS")
    print(f"{'='*60}")
    print(result.summary())

    # Extract and report key statistics
    params = result.params
    pvalues = result.pvalues
    conf_int = result.conf_int()

    print(f"\n{'='*60}")
    print("KEY RESULTS INTERPRETATION")
    print(f"{'='*60}")

    # Main effect of group (difference in intercepts)
    if "condition_num" in params:
        main_effect_b = params["condition_num"]
        main_effect_p = pvalues["condition_num"]
        main_effect_ci = conf_int.loc["condition_num"]
        print(f"Group main effect (baseline difference):")
        print(
            f"  β = {main_effect_b:.3f}, 95% CI [{main_effect_ci[0]:.3f}, {main_effect_ci[1]:.3f}], p = {main_effect_p:.4f}"
        )
        print(
            f"  Interpretation: Group {group_labels[1]} differs from {group_labels[0]} by {main_effect_b:.3f} units at baseline"
        )

    # Time effect (overall learning)
    if "task_number_c" in params:
        time_effect_b = params["task_number_c"]
        time_effect_p = pvalues["task_number_c"]
        time_effect_ci = conf_int.loc["task_number_c"]
        print(f"\nTime main effect (overall learning rate):")
        print(
            f"  β = {time_effect_b:.3f}, 95% CI [{time_effect_ci[0]:.3f}, {time_effect_ci[1]:.3f}], p = {time_effect_p:.4f}"
        )
        print(
            f"  Interpretation: Overall change of {time_effect_b:.3f} units per time point"
        )

    # Interaction effect (difference in learning rates)
    if "condition_num:task_number_c" in params:
        interaction_b = params["condition_num:task_number_c"]
        interaction_p = pvalues["condition_num:task_number_c"]
        interaction_ci = conf_int.loc["condition_num:task_number_c"]
        print(f"\nGroup × Time interaction (learning curve difference):")
        print(
            f"  β = {interaction_b:.3f}, 95% CI [{interaction_ci[0]:.3f}, {interaction_ci[1]:.3f}], p = {interaction_p:.4f}"
        )
        print(
            f"  Interpretation: Group {group_labels[1]} learning rate differs from {group_labels[0]} by {interaction_b:.3f} units per time point"
        )

        if interaction_p < 0.05:
            print(
                f"  *** SIGNIFICANT: Groups have significantly different learning curves! ***"
            )
        else:
            print(f"  Groups have similar learning curves (no significant difference)")

    # Model fit statistics
    print(f"\nModel Fit:")
    print(f"  AIC: {result.aic:.2f}")
    print(f"  BIC: {result.bic:.2f}")
    print(f"  Log-likelihood: {result.llf:.2f}")

    # Diagnostic plots
    if show_diagnostics:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Residuals vs fitted
        fitted_vals = result.fittedvalues
        residuals = result.resid
        axes[0, 0].scatter(fitted_vals, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color="red", linestyle="--")
        axes[0, 0].set_xlabel("Fitted Values")
        axes[0, 0].set_ylabel("Residuals")
        axes[0, 0].set_title("Residuals vs Fitted")

        # Q-Q plot
        probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title("Q-Q Plot of Residuals")

        # Random effects
        try:
            random_effects = result.random_effects
            subject_intercepts = [
                re[0] if len(re) > 0 else 0 for re in random_effects.values()
            ]
            axes[1, 0].hist(subject_intercepts, bins=15, alpha=0.7)
            axes[1, 0].set_xlabel("Random Intercepts")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].set_title("Distribution of Random Intercepts")

            if len(list(random_effects.values())[0]) > 1:  # Random slopes exist
                subject_slopes = [re[1] for re in random_effects.values()]
                axes[1, 1].hist(subject_slopes, bins=15, alpha=0.7)
                axes[1, 1].set_xlabel("Random Slopes")
                axes[1, 1].set_ylabel("Frequency")
                axes[1, 1].set_title("Distribution of Random Slopes")
            else:
                axes[1, 1].text(
                    0.5,
                    0.5,
                    "No random slopes\nin model",
                    ha="center",
                    va="center",
                    transform=axes[1, 1].transAxes,
                )
                axes[1, 1].set_title("Random Slopes")
        except:
            axes[1, 0].text(
                0.5,
                0.5,
                "Random effects\nnot available",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
            )
            axes[1, 1].text(
                0.5,
                0.5,
                "Random effects\nnot available",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )

        plt.suptitle("Model Diagnostics")
        plt.tight_layout()
        plt.show()

    # Visualization: observed means + error bars
    plot_df = (
        ddata.groupby(["task_number", group_col])
        .agg({outcome_col: ["mean", "sem", "count"]})
        .round(3)
    )
    plot_df.columns = ["mean", "sem", "n"]
    plot_df = plot_df.reset_index()

    # Visualization: observed + predicted in one figure
    plt.figure(figsize=(10, 6))

    # Consistent color map for groups
    colors = {group_name: plt.cm.Set1(i) for i, group_name in enumerate(group_labels)}

    # --- Observed data (means + SEM) ---
    for group_name, group_data in plot_df.groupby(group_col):
        plt.errorbar(
            group_data["task_number"],
            group_data["mean"],
            yerr=group_data["sem"],
            marker="o",
            markersize=8,
            linewidth=2,
            capsize=4,
            label=f"{group_name}",
            color=colors[group_name],
            alpha=0.8,
        )

    # --- Predicted curves ---
    if plot_predicted:
        task_vals = sorted(ddata["task_number"].unique())
        pred_df = pd.DataFrame(
            [
                {
                    "condition_num": cn,
                    "task_number_c": t - task_mean,
                    "task_number": t,
                    group_col: group_name,
                }
                for cn, group_name in enumerate(group_labels)
                for t in task_vals
            ]
        )
        pred_df["predicted"] = result.predict(pred_df)

        for group_name, group_data in pred_df.groupby(group_col):
            plt.plot(
                group_data["task_number"],
                group_data["predicted"],
                marker="",
                markersize=6,
                linewidth=2.5,
                linestyle="--",
                color=colors[group_name],
                label=f"{group_name} (predicted)",
                alpha=0.9,
            )

    # --- Axes, title, legend ---
    plt.xlabel("Time Point", fontsize=12)
    plt.ylabel(f"{outcome_col.upper()}", fontsize=12)
    plt.title(f"Learning Curves by Group {additional_title}", fontsize=14)

    # Add statistics annotations
    y_pos = 0.95
    if "condition_num" in pvalues:
        plt.annotate(
            f"Group effect: p = {pvalues['condition_num']:.3f}"
            + (" *" if pvalues["condition_num"] < 0.05 else ""),
            xy=(0.02, y_pos),
            xycoords="axes fraction",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
        )
        y_pos -= 0.08

    if "condition_num:task_number_c" in pvalues:
        plt.annotate(
            f"Group × Time: p = {pvalues['condition_num:task_number_c']:.3f}"
            + (" *" if pvalues["condition_num:task_number_c"] < 0.05 else ""),
            xy=(0.02, y_pos),
            xycoords="axes fraction",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
        )

    # make legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print(result.summary())
    return result


def check_normality(data: np.ndarray, min_size: int = 3, context: str = "") -> float:
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
        global_logger.warning(f"Warning: Normality violated (p={p:.4f}) {context}.")
    return p


def apply_multiple_correction(
    p_values: List[float], method: str, alpha: float = 0.05
) -> np.ndarray:
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
    values1: np.ndarray,
    values2: np.ndarray,
    method: str,
    n_permutations: int = 5000,
    labels: Optional[Tuple[str, str]] = None,
    group_name: Optional[str] = None,
    col: Optional[str] = None,
) -> Tuple[float, float]:
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
        global_logger.warning(
            f"Insufficient data for paired test (n1={len(values1)}, n2={len(values2)})"
        )
        return np.nan, np.nan


    differences = values1 - values2
    norm_p = check_normality(
        differences,
        context=(
            f"for difference '{labels[0]}' vs '{labels[1]}' in '{group_name}' for '{col}'"
            if labels and group_name and col
            else "for paired difference"
        ),
    )


    current_method = method
    if method == "auto":
        current_method, auto_diagnostics = _auto_select_test_method(
            data1=values1, data2=values2, test_type="paired"
        )
        if labels and group_name and col:
            global_logger.info(
                f"Auto-selected '{current_method}' for '{labels[0]}' vs '{labels[1]}' "
                f"in '{group_name}' for '{col}' (n={auto_diagnostics.get('n_pairs', 'N/A')}, "
                f"norm_p={auto_diagnostics.get('normality_p', 'N/A'):.4f}, "
                f"reason={auto_diagnostics.get('reason', 'N/A')})"
            )


    if current_method == "tukey":
        # Tukey requires multiple groups, but for paired test, we treat as two-group comparison
        data_list = [values1, values2]
        tukey_labels = [labels[0], labels[1]] if labels else ["group1", "group2"]
        p_dict = compute_tukey_pvalues(data_list, tukey_labels)
        comb = tuple(sorted(tukey_labels))
        p_value = p_dict.get(comb) or p_dict.get(comb[::-1], np.nan)
    else:
        # Inline the compute_paired_pvalue logic here
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
                    rvs1 = rvs2 = lambda size: np.random.choice(
                        combined, size, replace=True
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
            global_logger.warning(f"Paired test failed: {e}")
            p_value = np.nan


    return p_value, norm_p


def _compute_unpaired_pvalue(
    values1: np.ndarray,
    values2: np.ndarray,
    method: str,
    n_permutations: int = 5000,
    labels: Optional[Tuple[str, str]] = None,
    group_name: Optional[str] = None,
    col: Optional[str] = None,
    normality: bool = False,
) -> Tuple[float, Optional[float], Optional[float]]:
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
        global_logger.warning(
            f"Insufficient data for unpaired test (n1={len(values1)}, n2={len(values2)})"
        )
        return np.nan, None, None


    # Compute normality p-values if requested
    norm_p1 = norm_p2 = None
    if normality:
        norm_p1 = check_normality(
            values1,
            context=(
                f"for group '{labels[0] if labels else 'group1'}' in '{group_name}' for '{col}'"
                if labels and group_name and col
                else "for first group"
            ),
        )
        norm_p2 = check_normality(
            values2,
            context=(
                f"for group '{labels[1] if labels else 'group2'}' in '{group_name}' for '{col}'"
                if labels and group_name and col
                else "for second group"
            ),
        )


    current_method = method
    if method == "auto":
        current_method, auto_diagnostics = _auto_select_test_method(
            data1=values1, data2=values2, test_type="unpaired"
        )
        if labels and group_name and col:
            global_logger.info(
                f"Auto-selected '{current_method}' for '{group_name}' between groups "
                f"'{labels[0]}' (n={auto_diagnostics.get('n_group1', 'N/A')}) and "
                f"'{labels[1]}' (n={auto_diagnostics.get('n_group2', 'N/A')}) for '{col}' "
                f"(norm_p1={auto_diagnostics.get('normality_p_group1', 'N/A'):.4f}, "
                f"norm_p2={auto_diagnostics.get('normality_p_group2', 'N/A'):.4f}, "
                f"reason={auto_diagnostics.get('reason', 'N/A')})"
            )


    if current_method == "tukey":
        data_list = [values1, values2]
        tukey_labels = [labels[0], labels[1]] if labels else ["group1", "group2"]
        p_dict = compute_tukey_pvalues(data_list, tukey_labels)
        comb = tuple(sorted(tukey_labels))
        p_value = p_dict.get(comb) or p_dict.get(comb[::-1], np.nan)
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
            global_logger.warning(f"Unpaired test failed: {e}")
            p_value = np.nan


    return p_value, norm_p1, norm_p2



def compute_tukey_pvalues(
    data_list: List[np.ndarray], labels: List[str]
) -> Dict[Tuple[str, str], float]:
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
        [[labels[i]] * len(data_list[i]) for i in range(len(labels))]
    )
    res = pairwise_tukeyhsd(all_values, all_groups, alpha=0.05)
    combs = list(combinations(sorted(labels), 2))
    return dict(zip(combs, res.pvalues))


def statistical_comparison(
    df: pd.DataFrame,
    pair_name_col: str,
    compare_by: str,
    value_col: List[str],
    test_type: Literal["paired", "unpaired", "both"] = "both",
    group_by: Optional[str] = None,
    groups: List[str] = None,
    labels: List[str] = None,
    title: str = None,
    additional_title: str = None,
    save_dir: Optional[str] = None,
    plot_heatmaps: bool = True,
    plot_violins: bool = True,
    as_pdf: bool = False,
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
) -> Union[
    Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, Dict[str, pd.DataFrame]]]
]:
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
    title : str, optional
        Custom plot title.
    additional_title : str, optional
        Additional text appended to titles.
    save_dir : str, optional
        Directory to save plots.
    plot_heatmaps : bool, default=True
        Generate p-value heatmaps.
    plot_violins : bool, default=True
        Generate violin plots with significance annotations.
    as_pdf : bool, default=False
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

    if isinstance(test_type, str):
        test_types = [test_type]
    else:
        test_types = list(test_type)

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
        global_logger.info(f"Running paired statistical tests using {method}...")
        paired_method = method if method in paired_methods else "wilcoxon"
        if method not in paired_methods and method != "auto":
            global_logger.warning(
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
            global_logger.warning(
                "Unpaired test requested but group_by is None. Skipping unpaired test."
            )
            unpaired_possible = False
        else:
            unique_groups = df[group_by].unique() if groups is None else groups
            if len(unique_groups) < 2:
                global_logger.warning(
                    f"Unpaired test requested but only {len(unique_groups)} group(s) present. Skipping unpaired test."
                )
                unpaired_possible = False
        if unpaired_possible:
            global_logger.info(f"Running unpaired statistical tests using {method}...")
            unpaired_method = method if method in unpaired_methods else "mannwhitneyu"
            if method not in unpaired_methods and method != "auto":
                global_logger.warning(
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

    if plot_violins and (paired_results is not None or unpaired_results is not None):
        _plot_combined_violins(
            df=df,
            paired_results=paired_results,
            unpaired_results=unpaired_results,
            value_col=value_col,
            compare_by=compare_by,
            group_by=group_by,
            groups=groups,
            labels=labels,
            pair_name_col=pair_name_col,
            title=title,
            additional_title=additional_title,
            save_dir=save_dir,
            normality=normality,
        )

    if plot_heatmaps:
        if paired_results is not None:
            _plot_results(
                df=df,
                results=paired_results,
                test_mode="paired",
                pair_name_col=pair_name_col,
                compare_by=compare_by,
                value_col=value_col,
                group_by=group_by,
                groups=groups,
                labels=labels,
                title=title or f"Paired Test",
                additional_title=additional_title,
                save_dir=save_dir,
                plot_heatmaps=True,
                plot_violins=False,
                as_pdf=as_pdf,
                correction_method=correction_method,
                normality=normality,
            )
        if unpaired_results is not None:
            _plot_results(
                df=df,
                results=unpaired_results,
                test_mode="unpaired",
                pair_name_col=pair_name_col,
                compare_by=compare_by,
                value_col=value_col,
                group_by=group_by,
                groups=groups,
                labels=labels,
                title=title or f"Unpaired {method.title()} Test",
                additional_title=additional_title,
                save_dir=save_dir,
                plot_heatmaps=True,
                plot_violins=False,
                as_pdf=as_pdf,
                correction_method=correction_method,
                normality=normality,
            )

    return results


def _initialize_heatmaps(
    value_col: List[str],
    groups: List[str],
    labels: List[str],
    normality: bool,
    paired: bool,
) -> Tuple[
    Dict[str, Dict[str, pd.DataFrame]], Optional[Dict[str, Dict[str, pd.DataFrame]]]
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
            col: pd.DataFrame(np.nan, index=labels, columns=[col]) for col in value_col
        }
        normality_heatmaps = None
        if normality:
            normality_heatmaps = {
                col: pd.DataFrame(np.nan, index=labels, columns=groups)
                for col in value_col
            }
    return test_heatmaps, normality_heatmaps


def _run_statistical_tests(
    df: pd.DataFrame,
    pair_name_col: str,
    compare_by: str,
    value_col: List[str],
    group_by: Optional[str],
    groups: Optional[List[str]],
    labels: Optional[List[str]],
    method: str,
    correction_method: str,
    n_permutations: int,
    normality: bool,
    paired: bool,
) -> Union[
    Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, Dict[str, pd.DataFrame]]]
]:
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
    labels = df[compare_by].unique() if labels is None else labels
    groups = (
        ["all"]
        if group_by is None
        else (df[group_by].unique() if groups is None else groups)
    )

    if not paired and len(groups) != 2:
        raise ValueError(
            f"Unpaired tests require exactly 2 groups. Got {len(groups)}: {groups}"
        )

    num_value_cols = len(value_col)
    groups_count = len(groups)
    labels_count = len(labels)
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
        global_logger.info(
            f"Auto-selected correction method '{selected_correction_method}' for {'paired' if paired else 'unpaired'} tests "
            f"(estimated {num_comparisons} comparisons): {auto_diagnostics['reason']}"
        )

    test_heatmaps, normality_heatmaps = _initialize_heatmaps(
        value_col, groups, labels, normality, paired
    )

    for group_name in groups:
        group_df = df if group_by is None else df[df[group_by] == group_name]
        if paired:
            group_df = group_df.sort_values(by=pair_name_col)

        all_p_values = []
        all_comparisons = []

        if paired:
            combs = list(combinations(labels, 2))
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
                        values1,
                        values2,
                        method,
                        n_permutations,
                        (label1, label2),
                        group_name,
                        col,
                    )

                    if normality:
                        normality_heatmaps[col][group_name].loc[label1, label2] = norm_p
                        normality_heatmaps[col][group_name].loc[label2, label1] = norm_p

                    all_p_values.append(p_value)
                    all_comparisons.append((col, label1, label2))
        else:
            for col in value_col:
                for label in labels:
                    label_df = group_df[group_df[compare_by] == label]
                    groups_data = [
                        label_df[label_df[group_by] == g][col].dropna().values
                        for g in groups
                    ]

                    if any(len(g) == 0 for g in groups_data):
                        all_p_values.append(np.nan)
                        all_comparisons.append((col, label))
                        continue

                    p_value, norm_p1, norm_p2 = _compute_unpaired_pvalue(
                        groups_data[0],
                        groups_data[1],
                        method,
                        groups,
                        label,
                        col,
                        normality,
                    )

                    if normality:
                        normality_heatmaps[col].loc[label, groups[0]] = norm_p1
                        normality_heatmaps[col].loc[label, groups[1]] = norm_p2

                    all_p_values.append(p_value)
                    all_comparisons.append((col, label))

        corrected_p = apply_multiple_correction(
            all_p_values, selected_correction_method
        )

        if paired:
            for idx, (col, label1, label2) in enumerate(all_comparisons):
                test_heatmaps[col][group_name].loc[label1, label2] = corrected_p[idx]
                test_heatmaps[col][group_name].loc[label2, label1] = corrected_p[idx]
        else:
            for idx, (col, label) in enumerate(all_comparisons):
                test_heatmaps[col].loc[label, col] = corrected_p[idx]

    if normality:
        return {"pvalues": test_heatmaps, "normality": normality_heatmaps}
    return test_heatmaps


def _plot_results(
    df: pd.DataFrame,
    results: Union[Dict, pd.DataFrame],
    test_mode: str,
    pair_name_col: str,
    compare_by: str,
    value_col: List[str],
    group_by: Optional[str],
    groups: List[str],
    labels: List[str],
    title: str,
    additional_title: str,
    save_dir: Optional[str],
    plot_heatmaps: bool,
    plot_violins: bool,
    as_pdf: bool,
    correction_method: str,
    normality: bool,
):
    """Internal function to plot heatmaps and violin plots."""

    # Extract pvalues from results
    if isinstance(results, dict) and "pvalues" in results:
        pvalues_data = results["pvalues"]
        normality_data = results.get("normality", None)
    else:
        pvalues_data = results
        normality_data = None

    if plot_heatmaps:
        if test_mode == "paired":
            # Paired heatmaps - multiple groups
            labels_use = df[compare_by].unique() if labels is None else labels
            groups_use = (
                ["all"]
                if group_by is None
                else (df[group_by].unique() if groups is None else groups)
            )

            heatmap_size_multiplier = len(labels_use) / 10
            mult_by = max(1, heatmap_size_multiplier)
            figsize = (10 * mult_by, 6 * mult_by)

            for col in value_col:
                plot_title = f"{title} - {col} (Correction: {correction_method})"
                data = {
                    group_name: pvalues_data[col][group_name]
                    for group_name in groups_use
                }
                Vizualizer.plot_heatmap_dict_of_dicts(
                    data=data,
                    title=plot_title,
                    additional_title=additional_title,
                    labels=labels_use,
                    colorbar_ticks=[0, 0.025, 0.05],
                    colorbar_ticks_labels=["0", "0.025", "0.05"],
                    sharex=False,
                    sharey=False,
                    vmin=0,
                    vmax=0.05,
                    colorbar_label="p-value",
                    xlabel="Tasks",
                    ylabel="Tasks",
                    cmap="viridis_r",
                    figsize=figsize,
                    save_dir=save_dir,
                    as_pdf=as_pdf,
                    add_line=[(labels_use[0], labels_use[-1])],
                )

            # Plot normality heatmaps if available
            if normality and normality_data:
                for col in value_col:
                    plot_title = f"Normality Test p-values of Differences - {col}"
                    data = {
                        group_name: normality_data[col][group_name]
                        for group_name in groups_use
                    }
                    Vizualizer.plot_heatmap_dict_of_dicts(
                        data=data,
                        title=plot_title,
                        additional_title=additional_title,
                        labels=labels_use,
                        colorbar_ticks=[0, 0.025, 0.05],
                        colorbar_ticks_labels=["0", "0.025", "0.05"],
                        sharex=False,
                        sharey=False,
                        vmin=0,
                        vmax=0.05,
                        colorbar_label="p-value",
                        xlabel="Tasks",
                        ylabel="Tasks",
                        cmap="autumn",
                        figsize=figsize,
                        save_dir=save_dir,
                        as_pdf=as_pdf,
                        add_line=[(labels_use[0], labels_use[-1])],
                    )

        elif test_mode == "unpaired":
            # Unpaired heatmaps - simpler matrix
            labels_use = df[compare_by].unique() if labels is None else labels

            for col in value_col:
                pvalue_matrix = pvalues_data[col].values

                fig, ax = plt.subplots(figsize=(10, 8))
                cmap = plt.colormaps.get_cmap("viridis_r")
                cdark_gray = mcolors.to_rgba("dimgray", alpha=0.3)
                cmap.set_under(cdark_gray)
                cmap.set_over(cdark_gray)
                cax = ax.matshow(pvalue_matrix, cmap=cmap, vmin=0, vmax=0.05)

                for (i, j), val in np.ndenumerate(pvalue_matrix):
                    if not np.isnan(val):
                        ax.text(
                            j, i, f"{val:.3f}", ha="center", va="center", color="white"
                        )

                ax.set_xticks(np.arange(len(pvalues_data[col].columns)))
                ax.set_yticks(np.arange(len(labels_use)))
                ax.set_xticklabels(pvalues_data[col].columns, ha="right")
                ax.set_yticklabels(labels_use)
                ax.set_xlabel("Value Columns")
                ax.set_ylabel(compare_by)

                plot_title = f"{title} - {col} (Correction: {correction_method})"
                if additional_title:
                    plot_title = f"{plot_title} - {additional_title}"
                ax.set_title(plot_title)

                cbar = fig.colorbar(cax)
                cbar.set_label("p-value")
                plt.tight_layout()

                if save_dir:
                    save_path = Path(save_dir).joinpath(
                        f"{clean_filename(plot_title)}.png"
                    )
                    if as_pdf:
                        save_path = save_path.with_suffix(".pdf")
                    plt.savefig(save_path, bbox_inches="tight", dpi=300)
                plt.show()

    if plot_violins:
        for col in value_col:
            if test_mode == "paired":
                # Extract p-values for this column
                pvalues_for_col = pvalues_data[col]

                violin_plot(
                    df=df,
                    pair_name_col=pair_name_col,
                    compare_by=compare_by,
                    value_col=col,
                    group_by=group_by,
                    groups=groups,
                    labels=labels,
                    title=f"{title} - {col}",
                    additional_title=additional_title or "",
                    pvalues=pvalues_for_col,
                    save_dir=save_dir,
                )
            elif test_mode == "unpaired":
                # For unpaired, swap axes to show groups side-by-side
                pvalues_for_col = pvalues_data[col]

                violin_plot(
                    df=df,
                    pair_name_col=pair_name_col,
                    compare_by=group_by,  # Show groups on x-axis
                    value_col=col,
                    group_by=compare_by,  # Group by conditions
                    groups=labels,
                    labels=groups,
                    title=f"{title} - {col}",
                    additional_title=additional_title or "",
                    pvalues=pvalues_for_col,
                    save_dir=save_dir,
                )


def _plot_combined_violins(
    df: pd.DataFrame,
    paired_results: Optional[Dict],
    unpaired_results: Optional[Dict],
    value_col: List[str],
    compare_by: str,
    group_by: Optional[str],
    groups: List[str],
    labels: List[str],
    pair_name_col: str,
    title: Optional[str],
    additional_title: Optional[str],
    save_dir: Optional[str],
    normality: bool,
):
    """
    Create a combined violin plot showing both paired and unpaired test results.

    The plot is organized to show:
    - Left side: Within-group comparisons (paired tests)
    - Right side: Between-group comparisons (unpaired tests)
    - Clear visual separation with vertical line
    - Significance annotations for both test types
    - Individual data points overlaid
    - Colors from dataframe 'color' column
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import itertools
    from matplotlib import colors as mcolors

    # Helper function to convert hex/RGBA to CSS string
    def hex_to_rgba_str(color, alpha: float = 0.5) -> str:
        if color is None or pd.isna(color):
            return f"rgba(100, 100, 100, {alpha})"
        if isinstance(color, str) and color.startswith("#"):
            rgb = mcolors.to_rgb(color)
            return f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, {alpha})"
        elif isinstance(color, (list, tuple)) and len(color) >= 3:
            r, g, b = int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            return f"rgba({r}, {g}, {b}, {alpha})"
        return f"rgba(100, 100, 100, {alpha})"

    # Process groups and labels
    if group_by is not None:
        if groups is None:
            groups = sorted(df[group_by].unique())
    else:
        groups = ["All"]

    if labels is None:
        labels = sorted(df[compare_by].unique())

    # Determine subplot structure
    n_subplots = 0
    subplot_titles = []
    if paired_results is not None:
        n_subplots += 1
        subplot_titles.append("Within-Group (Paired)")
    if unpaired_results is not None:
        n_subplots += 1
        subplot_titles.append("Between-Group (Unpaired)")

    if n_subplots == 0:
        return

    # Create figure with subplots for each value_col
    for col_name in value_col:
        # Compute global min and max for y-axis range
        if group_by is not None and group_by != "All":
            df_all = df[df[group_by].isin(groups)]
        else:
            df_all = df
        global_min = df_all[col_name].min()
        global_max = df_all[col_name].max()
        global_range = global_max - global_min

        # Handle edge case where all values are identical
        if global_range == 0:
            global_range = 1.0

        # Add padding for violin plots (KDE can extend beyond data range)
        violin_padding = global_range * 0.15

        # Space for significance annotations
        bracket_delta = global_range * 0.05

        # Extract p-values
        def extract_pvals(results, col):
            if results is None:
                return None
            if normality and isinstance(results, dict) and "pvalues" in results:
                return results["pvalues"].get(col, None)
            else:
                return results.get(col, None)

        paired_pvals = extract_pvals(paired_results, col_name)
        unpaired_pvals = extract_pvals(unpaired_results, col_name)

        # Pre-calculate maximum number of significance pairs
        max_sig_pairs = 0

        # Count paired significance pairs
        if paired_pvals is not None:
            for group_name in groups:
                pval_df = (
                    paired_pvals.get(group_name, None)
                    if isinstance(paired_pvals, dict)
                    else paired_pvals
                )
                if pval_df is not None and isinstance(pval_df, pd.DataFrame):
                    sig_count = 0
                    for task1, task2 in itertools.combinations(labels, 2):
                        try:
                            if task1 in pval_df.index and task2 in pval_df.columns:
                                p = pval_df.loc[task1, task2]
                                if not pd.isna(p) and p < 0.05:
                                    sig_count += 1
                        except KeyError:
                            continue
                    max_sig_pairs = max(max_sig_pairs, sig_count)

        # Count unpaired significance pairs
        if unpaired_pvals is not None and isinstance(unpaired_pvals, pd.DataFrame):
            sig_count = 0
            for label in labels:
                if len(groups) == 2:
                    try:
                        if (
                            label in unpaired_pvals.index
                            and groups[0] in unpaired_pvals.columns
                        ):
                            p = unpaired_pvals.loc[label, groups[0]]
                            if not pd.isna(p) and p < 0.05:
                                sig_count += 1
                    except KeyError:
                        continue
            max_sig_pairs = max(max_sig_pairs, sig_count)

        # Calculate y-axis limits with proper padding
        y_min_plot = global_min - violin_padding
        if max_sig_pairs > 0:
            annot_space = (
                violin_padding + (max_sig_pairs * bracket_delta) + bracket_delta * 0.5
            )
        else:
            annot_space = violin_padding
        y_max_plot = global_max + annot_space

        # Create figure with vertical layout (rows, not columns)
        fig = make_subplots(
            rows=n_subplots,
            cols=1,
            subplot_titles=subplot_titles,
            shared_xaxes=False,
            shared_yaxes=True,
            vertical_spacing=0.15,
        )

        row_idx = 1

        # Plot paired results (within-group comparisons)
        if paired_results is not None and paired_pvals is not None:
            # Create mapping for unique x-axis positions across all groups
            position_counter = 0
            task_to_position = {}
            all_x_labels = []
            short_labels = {}
            # Generate short labels for each task (first 6 chars or custom logic)
            for label in labels:
                short_labels[label] = str(label)[:6]

            for group_idx, group_name in enumerate(groups):
                if isinstance(paired_pvals, dict) and group_name not in paired_pvals:
                    continue

                # Get filtered dataframe for this group
                if group_by is not None and group_by != "All":
                    df_group = df[df[group_by] == group_name]
                else:
                    df_group = df

                # Sort df_group
                df_group = df_group.sort_values(
                    by=(
                        ["group_key", "condition", "task_name", "task_number"]
                        if "group_key" in df_group.columns
                        else [compare_by]
                    )
                )

                unique_tasks = df_group[compare_by].drop_duplicates().tolist()

                # Map each task in this group to a unique position
                for task in unique_tasks:
                    short_label = short_labels.get(task, str(task))
                    x_label = f"{short_label}_{group_name}"
                    task_to_position[x_label] = position_counter
                    all_x_labels.append(x_label)
                    position_counter += 1

            # Now plot violins with unique positions
            for group_idx, group_name in enumerate(groups):
                if isinstance(paired_pvals, dict) and group_name not in paired_pvals:
                    continue

                # Get filtered dataframe for this group
                if group_by is not None and group_by != "All":
                    df_group = df[df[group_by] == group_name]
                else:
                    df_group = df

                # Sort df_group
                df_group = df_group.sort_values(
                    by=(
                        ["group_key", "condition", "task_name", "task_number"]
                        if "group_key" in df_group.columns
                        else [compare_by]
                    )
                )

                unique_tasks = df_group[compare_by].drop_duplicates().tolist()

                for task in unique_tasks:
                    task_df = df_group[df_group[compare_by] == task]
                    y = task_df[col_name].values.flatten()
                    sample_labels = (
                        task_df[pair_name_col].values.flatten()
                        if pair_name_col in task_df.columns
                        else None
                    )
                    colors = (
                        task_df["color"].values.flatten()
                        if "color" in task_df.columns
                        else None
                    )

                    # Check if all colors are the same
                    if colors is not None and len(set(colors)) == 1:
                        color = hex_to_rgba_str(colors[0], alpha=0.5)
                        violin_line_color = "black"
                    else:
                        color = None
                        violin_line_color = "black"

                    # Use short label for x-axis
                    short_label = short_labels.get(task, str(task))
                    x_label = f"{short_label}_{group_name}"

                    # Add violin plot with improved settings
                    fig.add_trace(
                        go.Violin(
                            y=y,
                            x=[x_label] * len(y),
                            text=sample_labels,
                            name=f"{group_name}",
                            legendgroup=f"{group_name}",
                            showlegend=(
                                False  # task == unique_tasks[0]
                            ),  # Only show legend for first task in group
                            side="positive",
                            hovertemplate=(
                                f"{pair_name_col}: " + "%{text}<br>"
                                f"{col_name}: " + "%{y}<extra></extra>"
                            ),
                            marker=dict(
                                color=color,
                                line=dict(
                                    color="black",
                                    width=1,
                                ),
                            ),
                            width=0.8,
                            points="all",
                            pointpos=-0.1,
                            jitter=0.1,
                            box_visible=True,
                            box=dict(
                                visible=True,
                                width=0.8,
                                fillcolor="rgba(255,255,255,0.8)",
                                line=dict(color="rgba(0,0,0,0.8)", width=2),
                            ),
                            meanline_visible=True,
                            meanline=dict(
                                visible=True,
                                color="red",
                                width=3,
                            ),
                            fillcolor=color,
                            line=dict(color="black", width=1),
                            opacity=0.7,
                        ),
                        row=row_idx,
                        col=1,
                    )

                # Add vertical gray line between groups (only if more than one group)
                if len(groups) == 2 and len(unique_tasks) > 0:
                    # Find the split position between the two groups
                    split_pos = len(unique_tasks) - 0.5
                    fig.add_shape(
                        type="line",
                        x0=split_pos,
                        y0=y_min_plot,
                        x1=split_pos,
                        y1=y_max_plot,
                        line=dict(color="gray", width=2, dash="dot"),
                        xref="x" if row_idx == 1 else f"x{row_idx}",
                        yref="y" if row_idx == 1 else f"y{row_idx}",
                    )
                # Add secondary x label below each paired group center
                if len(groups) == 2:
                    # Find center positions for each group
                    group_centers = []
                    start = 0
                    for group_name in groups:
                        group_tasks = [
                            x for x in all_x_labels if x.endswith(f"_{group_name}")
                        ]
                        if group_tasks:
                            center = start + (len(group_tasks) - 1) / 2
                            group_centers.append(center)
                            start += len(group_tasks)
                    for i, group_name in enumerate(groups):
                        paired_label = f"{group_name}"
                        fig.add_annotation(
                            x=group_centers[i],
                            y=y_min_plot - violin_padding * 1.2,
                            text=paired_label,
                            showarrow=False,
                            font=dict(size=14, color="gray"),
                            xref="x" if row_idx == 1 else f"x{row_idx}",
                            yref="y" if row_idx == 1 else f"y{row_idx}",
                            align="center",
                            valign="bottom",
                        )
                else:
                    # Single group
                    fig.add_annotation(
                        x=(len(all_x_labels) - 1) / 2,
                        y=y_min_plot - violin_padding * 1.2,
                        text=f"{groups[0]}",
                        showarrow=False,
                        font=dict(size=14, color="gray"),
                        xref="x" if row_idx == 1 else f"x{row_idx}",
                        yref="y" if row_idx == 1 else f"y{row_idx}",
                        align="center",
                        valign="bottom",
                    )

                # Add significance annotations for paired tests (within this group only)
                pval_df = (
                    paired_pvals.get(group_name, None)
                    if isinstance(paired_pvals, dict)
                    else paired_pvals
                )
                if pval_df is not None and isinstance(pval_df, pd.DataFrame):
                    sig_pairs = []

                    for task1, task2 in itertools.combinations(unique_tasks, 2):
                        try:
                            if task1 in pval_df.index and task2 in pval_df.columns:
                                p = pval_df.loc[task1, task2]
                                if not pd.isna(p) and p < 0.05:
                                    # Use the task_to_position mapping with group labels
                                    pos1 = task_to_position[f"{task1}_{group_name}"]
                                    pos2 = task_to_position[f"{task2}_{group_name}"]
                                    if pos1 > pos2:
                                        pos1, pos2 = pos2, pos1
                                    if p < 0.001:
                                        sig = "***"
                                    elif p < 0.01:
                                        sig = "**"
                                    else:
                                        sig = "*"
                                    sig_pairs.append((pos1, pos2, sig))
                        except KeyError:
                            continue

                    if sig_pairs:
                        # Sort by span descending for stacking
                        sig_pairs.sort(key=lambda x: x[1] - x[0], reverse=True)
                        current_height = (
                            global_max
                            + violin_padding
                            + (len(sig_pairs) * bracket_delta)
                        )
                        xref = "x" if row_idx == 1 else f"x{row_idx}"
                        yref = "y" if row_idx == 1 else f"y{row_idx}"

                        for pos1, pos2, sig in sig_pairs:
                            height = current_height
                            # Add horizontal line
                            fig.add_shape(
                                type="line",
                                x0=pos1,
                                y0=height,
                                x1=pos2,
                                y1=height,
                                line=dict(color="black", width=1),
                                xref=xref,
                                yref=yref,
                                row=row_idx,
                                col=1,
                            )
                            # Add stars annotation
                            fig.add_annotation(
                                x=(pos1 + pos2) / 2,
                                y=height + bracket_delta * 0.2,
                                text=sig,
                                showarrow=False,
                                font=dict(size=12),
                                xref=xref,
                                yref=yref,
                                row=row_idx,
                                col=1,
                            )
                            current_height -= bracket_delta

            row_idx += 1

        # Plot unpaired results (between-group comparisons)
        if unpaired_results is not None and unpaired_pvals is not None:
            # Create mapping for x-axis positions
            position_counter = 0
            x_label_to_position = {}
            all_x_labels_unpaired = []
            short_labels_unpaired = {}
            for label in labels:
                short_labels_unpaired[label] = str(label)[:6]

            # First pass: create position mapping
            for label in labels:
                for group_name in groups:
                    short_label = short_labels_unpaired.get(label, str(label))
                    x_label = f"{short_label}_{group_name}"
                    x_label_to_position[x_label] = position_counter
                    all_x_labels_unpaired.append(x_label)
                    position_counter += 1

            # Second pass: plot violins
            for label in labels:
                for group_name in groups:
                    # Filter data for this label and group
                    if group_by is not None and group_by != "All":
                        mask = (df[group_by] == group_name) & (df[compare_by] == label)
                    else:
                        mask = df[compare_by] == label

                    task_df = df[mask]
                    y = task_df[col_name].values.flatten()
                    sample_labels = (
                        task_df[pair_name_col].values.flatten()
                        if pair_name_col in task_df.columns
                        else None
                    )
                    colors = (
                        task_df["color"].values.flatten()
                        if "color" in task_df.columns
                        else None
                    )

                    if len(y) == 0:
                        continue

                    # Check if all colors are the same
                    if colors is not None and len(set(colors)) == 1:
                        color = hex_to_rgba_str(colors[0], alpha=0.5)
                        violin_line_color = "black"
                    else:
                        color = None
                        violin_line_color = "black"

                    # Use short label for x-axis
                    short_label = short_labels_unpaired.get(label, str(label))
                    x_label = f"{short_label}_{group_name}"

                    # Add violin plot
                    fig.add_trace(
                        go.Violin(
                            y=y,
                            x=[x_label] * len(y),
                            text=sample_labels,
                            name=f"{group_name}",
                            legendgroup=f"{group_name}",
                            showlegend=False,  # Already shown in paired subplot
                            side="positive",
                            hovertemplate=(
                                f"{pair_name_col}: " + "%{text}<br>"
                                f"{col_name}: " + "%{y}<extra></extra>"
                            ),
                            marker=dict(
                                color=color,
                                line=dict(
                                    color="black",
                                    width=1,
                                ),
                            ),
                            width=0.8,
                            points="all",
                            pointpos=-0.1,
                            jitter=0.1,
                            box_visible=True,
                            box=dict(
                                visible=True,
                                width=0.8,
                                fillcolor="rgba(255,255,255,0.8)",
                                line=dict(color="rgba(0,0,0,0.8)", width=2),
                            ),
                            meanline_visible=True,
                            meanline=dict(
                                visible=True,
                                color="red",
                                width=3,
                            ),
                            fillcolor=color,
                            line=dict(color="black", width=1),
                            opacity=0.7,
                        ),
                        row=row_idx,
                        col=1,
                    )
                # Add secondary x label between each pair of violin plots (per task)
                n_tasks = len(labels)
                if len(groups) == 2:
                    for i, label in enumerate(labels):
                        # Find the positions of the two violins for this task
                        short_label = short_labels_unpaired.get(label, str(label))
                        x_label_0 = f"{short_label}_{groups[0]}"
                        x_label_1 = f"{short_label}_{groups[1]}"
                        pos0 = all_x_labels_unpaired.index(x_label_0)
                        pos1 = all_x_labels_unpaired.index(x_label_1)
                        center = (pos0 + pos1) / 2
                        fig.add_annotation(
                            x=center,
                            y=y_min_plot - violin_padding * 1.2,
                            text=str(label),
                            showarrow=False,
                            font=dict(size=14, color="gray"),
                            xref="x2" if row_idx == 2 else "x",
                            yref="y2" if row_idx == 2 else "y",
                            align="center",
                            valign="bottom",
                        )
                else:
                    # Only one group, so just label each violin
                    for i, label in enumerate(labels):
                        short_label = short_labels_unpaired.get(label, str(label))
                        x_label = f"{short_label}_{groups[0]}"
                        pos = all_x_labels_unpaired.index(x_label)
                        fig.add_annotation(
                            x=pos,
                            y=y_min_plot - violin_padding * 1.2,
                            text=str(label),
                            showarrow=False,
                            font=dict(size=14, color="gray"),
                            xref="x2" if row_idx == 2 else "x",
                            yref="y2" if row_idx == 2 else "y",
                            align="center",
                            valign="bottom",
                        )

            # Add significance annotations for unpaired tests (between groups)
            if isinstance(unpaired_pvals, pd.DataFrame):
                sig_pairs = []
                for label in labels:
                    if len(groups) == 2:
                        try:
                            if label in unpaired_pvals.index:
                                p = unpaired_pvals.loc[label].iloc[0]
                                if not pd.isna(p) and p < 0.05:

                                    # get position of x_label with label insight group name
                                    pos1 = x_label_to_position[f"{label}_{groups[0]}"]
                                    pos2 = x_label_to_position[f"{label}_{groups[1]}"]
                                    if p < 0.001:
                                        sig = "***"
                                    elif p < 0.01:
                                        sig = "**"
                                    else:
                                        sig = "*"
                                    sig_pairs.append((pos1, pos2, sig))
                        except KeyError:
                            continue

                if sig_pairs:
                    # Sort by span descending for stacking
                    sig_pairs.sort(key=lambda x: x[1] - x[0], reverse=True)
                    current_height = (
                        global_max + violin_padding + (len(sig_pairs) * bracket_delta)
                    )
                    # Use categorical axis names for subplot 2
                    xref = "x" if row_idx == 1 else "x2"
                    yref = "y" if row_idx == 1 else "y2"

                    for pos1, pos2, sig in sig_pairs:
                        height = current_height
                        # Add horizontal line
                        fig.add_shape(
                            type="line",
                            x0=pos1,
                            y0=height,
                            x1=pos2,
                            y1=height,
                            line=dict(color="black", width=1),
                            xref=xref,
                            yref=yref,
                        )
                        # Add stars annotation
                        fig.add_annotation(
                            x=(pos1 + pos2) / 2,
                            y=height + bracket_delta * 0.2,
                            text=sig,
                            showarrow=False,
                            font=dict(size=12),
                            xref=xref,
                            yref=yref,
                        )
                        current_height -= bracket_delta

        # Update layout
        num_groups = len(groups) if groups != ["All"] else 1
        plot_title = f"{title or 'Statistical Comparison'} - {col_name}"
        if additional_title:
            plot_title = f"{plot_title} <br>{additional_title}"

        # Adjust figure height and y-axis to fit annotation
        extra_annotation_space = violin_padding * 1.5
        fig.update_layout(
            title_text=plot_title,
            showlegend=False,
            legend=dict(
                title=dict(text="Group"),
                orientation="v",
                yanchor="top",
                y=1.0,
                xanchor="left",
                x=1.02,
            ),
            height=400 * num_groups + 60,
            violinmode="group",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        # Set the same y-axis range for all subplots, with extra space for annotation
        fig.update_yaxes(
            range=[y_min_plot - extra_annotation_space, y_max_plot],
            showgrid=True,
            gridcolor="rgba(200,200,200,0.25)",
            gridwidth=1,
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor="rgba(200,200,200,0.25)",
            gridwidth=1,
        )

        # Save and show
        if save_dir:
            save_path = clean_filename(
                str(Path(save_dir) / f"Violin_{col_name}_{additional_title}.html")
            )
            fig.write_html(save_path)
            global_logger.info(f"Saved combined violin plot to {save_path}")

        fig.show()


def mannwhitneyu_cross_df(
    df: pd.DataFrame,
    group_by: str,
    compare_by: str,
    value_col: List[str],
    groups: List[str] = None,
    labels: List[str] = None,
    title: str = None,
    additional_title: str = None,
    save_dir: Optional[str] = None,
    plot: bool = False,
    as_pdf: bool = False,
) -> Dict[str, pd.DataFrame]:
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
    title : str, optional
        Title for plots. If None, generated automatically.
    additional_title : str, optional
        Additional text for plot titles.
    save_dir : str, optional
        Directory to save plots.
    plot : bool, optional
        Whether to generate heatmaps of p-values.
    as_pdf : bool, optional
        Save plots as PDF if True.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with group names as keys and DataFrames of p-values as values.
        Each DataFrame contains p-values for each label pair.
    """
    value_col = make_list_ifnot(value_col)

    title = (
        title
        or f"Mann-Whitney U Test p-values {value_col.upper()} for {compare_by} across {group_by}"
    )

    if title is None:
        group_by_str = ", ".join(make_list_ifnot(group_by))
        title = (
            f"Wilcoxon Signed-Rank Test across {compare_by} grouped by {group_by_str}"
        )

    labels = df[compare_by].unique() if labels is None else labels
    groups = df[group_by].unique() if groups is None else groups
    if len(groups) != 2:
        do_critical(
            ValueError,
            "Mann-Whitney U test requires exactly two groups. "
            f"Got {len(groups)} groups: {groups}.",
        )

    # create dataframe with compare_by as index and value_col as columns
    pvalues_df = pd.DataFrame(index=labels, columns=value_col, dtype=float)
    for label in labels:
        if label not in df[compare_by].unique():
            do_critical(
                ValueError,
                f"Label '{label}' not found in column '{compare_by}'.",
            )
        label_df = df[df[compare_by] == label]
        for col in value_col:
            if col not in label_df.columns:
                do_critical(
                    ValueError,
                    f"Column '{col}' not found in DataFrame.",
                )

            g1 = label_df[label_df[group_by] == groups[0]][col].values
            g2 = label_df[label_df[group_by] == groups[1]][col].values

            stat, p_value = mannwhitneyu(g1, g2)
            pvalues_df.loc[label, col] = p_value

        # Apply multiple testing correction
        if len(value_col) > 1:
            p_values = pvalues_df.loc[label].values
            valid_mask = ~np.isnan(p_values)
            if valid_mask.any():
                corrected_p = multipletests(
                    p_values[valid_mask],
                    method="holm",
                    alpha=0.05,
                )[1]
                pvalues_df.loc[label, value_col] = np.where(
                    valid_mask, corrected_p, np.nan
                )
            else:
                pvalues_df.loc[label, value_col] = np.nan

    if plot:
        # create a matrix of p-values for heatmap
        pvalue_matrix = pvalues_df.values

        # plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.colormaps.get_cmap("viridis_r")
        cdark_gray = mcolors.to_rgba("dimgray", alpha=0.3)
        cmap.set_under(cdark_gray)
        cmap.set_over(cdark_gray)
        cax = ax.matshow(pvalue_matrix, cmap=cmap, vmin=0, vmax=0.05)
        # add annotation
        for (i, j), val in np.ndenumerate(pvalue_matrix):
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                )
        ax.set_xticks(np.arange(len(value_col)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(value_col, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Value Columns")
        ax.set_ylabel(compare_by)
        if additional_title:
            ax.set_title(f"{title} - {additional_title}")
        ax.set_title(title)
        cbar = fig.colorbar(cax)
        cbar.set_label("p-value")
        plt.tight_layout()
        if save_dir:
            title = clean_filename(title)
            save_path = Path(save_dir).joinpath(f"{title}.png")
            if as_pdf:
                save_path = save_path.with_suffix(".pdf")
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.show()

    return pvalues_df


def sigtest_cross_df(
    df: pd.DataFrame,
    pair_name_col: str,
    compare_by: str,
    value_col: List[str],
    group_by: Optional[str] = None,
    groups: List[str] = None,
    labels: List[str] = None,
    title: str = None,
    additional_title: str = None,
    save_dir: Optional[str] = None,
    plot: bool = False,
    as_pdf: bool = False,
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
) -> Union[
    Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, Dict[str, pd.DataFrame]]]
]:
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
    title : str, optional
        A custom title for the generated plots. If None, a title is
        automatically generated, by default None.
    additional_title : str, optional
        Additional text to append to the plot title, by default None.
    save_dir : str, optional
        The directory path to save the plots. If None, plots are not saved to
        disk, by default None.
    plot : bool, optional
        If True, generates and displays heatmaps of the resulting p-values,
        by default False.
    as_pdf : bool, optional
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
    ...     print(f"\n--- P-values for {value_name} ---")
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
    method_name = {
        "wilcoxon": "Wilcoxon Signed-Rank",
        "ttest": "Paired t",
        "permutation": "Permutation",
        "monte_carlo_test": "Monte Carlo",
        "monte_carlo_test_normal": "Monte Carlo Normal",
        "tukey": "Tukey HSD",
    }[method]
    if title is None:
        if group_by is None:
            title = f"{method_name} Test across {compare_by}"
        else:
            group_by_str = ", ".join(make_list_ifnot(group_by))
            title = f"{method_name} Test p-values across {compare_by} grouped by {group_by_str}"
    labels = df[compare_by].unique() if labels is None else labels
    groups = (
        ["all"]
        if group_by is None
        else (df[group_by].unique() if groups is None else groups)
    )
    test_heatmaps = {
        col: {group_name: None for group_name in groups} for col in value_col
    }
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
        # Initialize p-value DataFrames
        for col in value_col:
            test_df = pd.DataFrame(index=labels, columns=labels, dtype=float)
            test_df = test_df.fillna(np.nan)
            test_heatmaps[col][group_name] = test_df
        # Collect all p-values for this group across all comparisons and value_cols
        all_p_values = []
        all_comparisons = []
        if method == "tukey":
            tukey_results = {}
            if normality:
                for col in value_col:
                    for label in labels:
                        values = (
                            group_df[group_df[compare_by] == label][col].dropna().values
                        )
                        if len(values) >= 3:
                            _, norm_p = shapiro(values)
                            normality_heatmaps[col][group_name].loc[
                                label, label
                            ] = norm_p
                            # warning if normality is violated
                            if norm_p < 0.05:
                                global_logger.warning(
                                    f"Warning: Normality test p-value {norm_p:.4f} < 0.05 for difference between '{label1}' and '{label2}' in group '{group_name}' for column '{col}'. Consider using a non-parametric test."
                                )
            for col in value_col:
                labels_sorted = np.sort(labels)
                groups_data = [
                    group_df[group_df[compare_by] == label][col].dropna().values
                    for label in labels_sorted
                ]
                if len(labels_sorted) < 2 or any(len(g) == 0 for g in groups_data):
                    tukey_results[col] = None
                    continue
                all_values = np.concatenate(groups_data)
                all_groups = np.concatenate(
                    [
                        [labels_sorted[i]] * len(groups_data[i])
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
            # Paired methods
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
                            ValueError,
                            f"Column '{pair_name_col}' has more than one unique value in group '{label1}' for common name '{common_name}'.",
                        )
                    if (
                        group_label2_df[
                            group_label2_df[pair_name_col] == common_name
                        ].shape[0]
                        != 1
                    ):
                        do_critical(
                            ValueError,
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
                            normality_heatmaps[col][group_name].loc[
                                label1, label2
                            ] = norm_p
                            normality_heatmaps[col][group_name].loc[
                                label2, label1
                            ] = norm_p
                            # warning if normality is violated
                            if norm_p < 0.05:
                                global_logger.warning(
                                    f"Warning: Normality test p-value {norm_p:.4f} < 0.05 for difference between '{label1}' and '{label2}' in group '{group_name}' for column '{col}'. Consider using a non-parametric test."
                                )
                        else:
                            normality_heatmaps[col][group_name].loc[
                                label1, label2
                            ] = np.nan
                            normality_heatmaps[col][group_name].loc[
                                label2, label1
                            ] = np.nan
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
                                    lambda size: np.random.normal(mu, sigma, size=size),
                                    lambda size: np.random.normal(mu, sigma, size=size),
                                )
                            else:
                                rvs = (
                                    lambda size: np.random.choice(
                                        combined, size=size, replace=True
                                    ),
                                    lambda size: np.random.choice(
                                        combined, size=size, replace=True
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
        # Apply multiple testing correction across all p-values for this group
        if correction_method != "none" and all_p_values:
            all_p_values = np.array(all_p_values)
            valid_mask = ~np.isnan(all_p_values)
            corrected_p = np.full(all_p_values.shape, np.nan)
            if valid_mask.any():
                corrected_p[valid_mask] = multipletests(
                    all_p_values[valid_mask],
                    method=correction_method,
                    alpha=0.05,
                )[1]
            # Assign corrected p-values back to DataFrames
            for idx, (col, label1, label2) in enumerate(all_comparisons):
                test_heatmaps[col][group_name].loc[label1, label2] = corrected_p[idx]
                test_heatmaps[col][group_name].loc[label2, label1] = corrected_p[idx]
        else:
            # Assign uncorrected p-values
            for idx, (col, label1, label2) in enumerate(all_comparisons):
                test_heatmaps[col][group_name].loc[label1, label2] = all_p_values[idx]
                test_heatmaps[col][group_name].loc[label2, label1] = all_p_values[idx]
    if plot:
        heatmap_size_multiplier = len(labels) / 10
        mult_by = max(1, heatmap_size_multiplier)
        figsize = (10 * mult_by, 6 * mult_by)
        for col in value_col:
            plot_title = f"{title} - {col} (Correction: {correction_method})"
            data = {group_name: test_heatmaps[col][group_name] for group_name in groups}
            Vizualizer.plot_heatmap_dict_of_dicts(
                data=data,
                title=plot_title,
                additional_title=additional_title,
                custom_annotation_label="Number of Animals Used",
                labels=labels,
                colorbar_ticks=[0, 0.025, 0.05],
                colorbar_ticks_labels=["0", "0.025", "0.05"],
                sharex=False,
                sharey=False,
                vmin=0,
                vmax=0.05,
                colorbar_label="p-value",
                xlabel="Tasks",
                ylabel="Tasks",
                cmap="viridis",
                figsize=figsize,
                save_dir=save_dir,
                as_pdf=as_pdf,
            )
        if normality:
            for col in value_col:
                plot_title = f"Normality Test p-values of Differences - {col}"
                data = {
                    group_name: normality_heatmaps[col][group_name]
                    for group_name in groups
                }
                Vizualizer.plot_heatmap_dict_of_dicts(
                    data=data,
                    title=plot_title,
                    additional_title=additional_title,
                    custom_annotation_label="Number of Animals Used",
                    labels=labels,
                    colorbar_ticks=[0, 0.025, 0.05],
                    colorbar_ticks_labels=["0", "0.025", "0.05"],
                    sharex=False,
                    sharey=False,
                    vmin=0,
                    vmax=0.05,
                    colorbar_label="p-value",
                    xlabel="Tasks",
                    ylabel="Tasks",
                    cmap="autumn",
                    figsize=figsize,
                    save_dir=save_dir,
                    as_pdf=as_pdf,
                )
    if normality:
        return {"pvalues": test_heatmaps, "normality": normality_heatmaps}
    return test_heatmaps


def statistical_comparison(
    df: pd.DataFrame,
    pair_name_col: str,
    compare_by: str,
    value_col: List[str],
    test_type: Literal["paired", "unpaired", "both"] = "both",
    group_by: Optional[str] = None,
    groups: List[str] = None,
    labels: List[str] = None,
    title: str = None,
    additional_title: str = None,
    save_dir: Optional[str] = None,
    plot_heatmaps: bool = True,
    plot_violins: bool = True,
    as_pdf: bool = False,
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
) -> Union[
    Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, Dict[str, pd.DataFrame]]]
]:
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
    title : str, optional
        Custom plot title.
    additional_title : str, optional
        Additional text appended to titles.
    save_dir : str, optional
        Directory to save plots.
    plot_heatmaps : bool, default=True
        Generate p-value heatmaps.
    plot_violins : bool, default=True
        Generate violin plots with significance annotations.
    as_pdf : bool, default=False
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

    if isinstance(test_type, str):
        test_types = [test_type]
    else:
        test_types = list(test_type)

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
        global_logger.info(f"Running paired statistical tests using {method}...")
        paired_method = method if method in paired_methods else "wilcoxon"
        if method not in paired_methods and method != "auto":
            global_logger.warning(
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
            global_logger.warning(
                "Unpaired test requested but group_by is None. Skipping unpaired test."
            )
            unpaired_possible = False
        else:
            unique_groups = df[group_by].unique() if groups is None else groups
            if len(unique_groups) < 2:
                global_logger.warning(
                    f"Unpaired test requested but only {len(unique_groups)} group(s) present. Skipping unpaired test."
                )
                unpaired_possible = False
        if unpaired_possible:
            global_logger.info(f"Running unpaired statistical tests using {method}...")
            unpaired_method = method if method in unpaired_methods else "mannwhitneyu"
            if method not in unpaired_methods and method != "auto":
                global_logger.warning(
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

    if plot_violins and (paired_results is not None or unpaired_results is not None):
        _plot_combined_violins(
            df=df,
            paired_results=paired_results,
            unpaired_results=unpaired_results,
            value_col=value_col,
            compare_by=compare_by,
            group_by=group_by,
            groups=groups,
            labels=labels,
            pair_name_col=pair_name_col,
            title=title,
            additional_title=additional_title,
            save_dir=save_dir,
            normality=normality,
        )

    if plot_heatmaps:
        if paired_results is not None:
            _plot_results(
                df=df,
                results=paired_results,
                test_mode="paired",
                pair_name_col=pair_name_col,
                compare_by=compare_by,
                value_col=value_col,
                group_by=group_by,
                groups=groups,
                labels=labels,
                title=title or f"Paired Test",
                additional_title=additional_title,
                save_dir=save_dir,
                plot_heatmaps=True,
                plot_violins=False,
                as_pdf=as_pdf,
                correction_method=correction_method,
                normality=normality,
            )
        if unpaired_results is not None:
            _plot_results(
                df=df,
                results=unpaired_results,
                test_mode="unpaired",
                pair_name_col=pair_name_col,
                compare_by=compare_by,
                value_col=value_col,
                group_by=group_by,
                groups=groups,
                labels=labels,
                title=title or f"Unpaired {method.title()} Test",
                additional_title=additional_title,
                save_dir=save_dir,
                plot_heatmaps=True,
                plot_violins=False,
                as_pdf=as_pdf,
                correction_method=correction_method,
                normality=normality,
            )

    return results


#################################  Fitting Lines ########################################
FunctionType = Callable[
    [np.ndarray, float, Optional[float], Optional[float]], np.ndarray
]
FunctionConfig = Dict[str, Union[bool, float]]


class FunctionModel:
    """Base class for function models with shift and inverse options."""

    @staticmethod
    def shift_x(x: np.ndarray, x_shift: float) -> np.ndarray:
        """Apply horizontal shift to x values."""
        return x - x_shift

    @staticmethod
    def shift_y(y: np.ndarray, y_shift: float) -> np.ndarray:
        """Apply vertical shift to y values."""
        return y + y_shift

    @classmethod
    def apply_inverse(cls, y: np.ndarray, inverse: bool) -> np.ndarray:
        """Apply inverse transformation if requested."""
        return 1.0 / y if inverse else y

    @classmethod
    def linear(
        cls, x: np.ndarray, m: float, c: float, config: FunctionConfig
    ) -> np.ndarray:
        """Linear function: y = m*(x - x_shift) + c or its inverse."""
        x_shifted = cls.shift_x(x, config.get("x_shift", 0.0))
        y = m * x_shifted + c
        y_shifted = cls.shift_y(y, config.get("y_shift", 0.0))
        return cls.apply_inverse(y_shifted, config.get("inverse", False))

    @classmethod
    def exponential(
        cls, x: np.ndarray, k: float, a: float, config: FunctionConfig
    ) -> np.ndarray:
        """Exponential function: y = a*exp(-k*(x - x_shift)) or its inverse."""
        x_shifted = cls.shift_x(x, config.get("x_shift", 0.0))
        y = a * np.exp(-k * x_shifted)
        y_shifted = cls.shift_y(y, config.get("y_shift", 0.0))
        return cls.apply_inverse(y_shifted, config.get("inverse", False))

    @classmethod
    def gaussian(
        cls, x: np.ndarray, k: float, a: float, config: FunctionConfig
    ) -> np.ndarray:
        """Gaussian function: y = a*exp(-k*(x - x_shift)^2) or its inverse."""
        x_shifted = cls.shift_x(x, config.get("x_shift", 0.0))
        y = a * np.exp(-k * x_shifted**2)
        y_shifted = cls.shift_y(y, config.get("y_shift", 0.0))
        return cls.apply_inverse(y_shifted, config.get("inverse", False))

    @classmethod
    def hyperbolic(
        cls, x: np.ndarray, k: float, a: float, config: FunctionConfig
    ) -> np.ndarray:
        """Hyperbolic function: y = a/(1 + k*(x - x_shift)) or its inverse."""
        x_shifted = cls.shift_x(x, config.get("x_shift", 0.0))
        y = a / (1 + k * x_shifted)
        y_shifted = cls.shift_y(y, config.get("y_shift", 0.0))
        return cls.apply_inverse(y_shifted, config.get("inverse", False))

    @classmethod
    def power(
        cls, x: np.ndarray, k: float, n: float, a: float, config: FunctionConfig
    ) -> np.ndarray:
        """Power function: y = a/(1 + k*(x - x_shift))^n or its inverse."""
        x_shifted = cls.shift_x(x, config.get("x_shift", 0.0))
        y = a / ((1 + k * x_shifted) ** n)
        y_shifted = cls.shift_y(y, config.get("y_shift", 0.0))
        return cls.apply_inverse(y_shifted, config.get("inverse", False))


def get_auc(
    x: List[float],
    y: List[float],
    functions: Union[
        str,
        List[
            Literal["auto", "linear", "exponential", "gaussian", "hyperbolic", "power"]
        ],
    ] = "auto",
    plot: bool = False,
    method: Literal["best", "fast"] = "best",
    return_fit: bool = False,
) -> float:
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
    plot : bool, default=False
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
    if method == "fast":
        # Use trapezoidal rule for quick AUC calculation
        auc = np.trapz(y, x)
    else:
        fit_result = get_best_fit(x, y, functions=functions, plot=plot)
        auc = fit_result.get("auc", np.nan)
        if auc is np.nan:
            global_logger.error(
                "Failed to compute AUC from best fit; falling back to trapezoidal rule."
            )
            auc = np.trapz(y, x)
        if return_fit:
            return auc, fit_result
    return auc


def get_best_fit(
    x: List[float],
    y: List[float],
    functions: Union[
        str,
        List[
            Literal["auto", "linear", "exponential", "gaussian", "hyperbolic", "power"]
        ],
    ] = "auto",
    save_dir: Optional[str] = None,
    plot: bool = True,
    x_shift: float = 0.0,
    y_shift: float = 0.0,
    inverse: bool = False,
    maxfev: int = 10000,
) -> Dict[str, Union[float, List[float], np.ndarray, str]]:
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
    save_dir : str, optional
        Directory to save the plot of the best-fitting model. If None, plot is not saved.
    plot : bool, default=True
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
    # Initialize variables
    x = np.array(x)
    y = np.array(y)
    x_dense = np.linspace(min(x), max(x), 100)
    fits: Dict[str, Dict] = {}
    implemented_functions = ["linear", "exponential", "gaussian", "hyperbolic", "power"]
    function_map: Dict[str, FunctionType] = {
        "linear": FunctionModel.linear,
        "exponential": FunctionModel.exponential,
        "gaussian": FunctionModel.gaussian,
        "hyperbolic": FunctionModel.hyperbolic,
        "power": FunctionModel.power,
    }

    if functions is None or (isinstance(functions, str) and functions == "auto"):
        functions = implemented_functions
    else:
        functions = make_list_ifnot(functions)
        for func in functions:
            if func not in implemented_functions:
                raise ValueError(
                    f"Function '{func}' is not implemented. Choose from {implemented_functions} or use 'auto'."
                )

    config: FunctionConfig = {
        "x_shift": x_shift,
        "y_shift": y_shift,
        "inverse": inverse,
    }
    initial_guesses = {
        "positive": [0.1, 2.0, 1.0],  # k, n, a
        "negative": [-0.1, -2.0, 1.0],
    }
    for func_name in functions:
        func = function_map[func_name]
        try:
            try:
                if func_name == "power":
                    p0 = initial_guesses["positive"]  # Initial guess for k, n, a
                    bounds = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])
                    popt, _ = curve_fit(
                        lambda x, k, n, a: func(x, k, n, a, config),
                        x,
                        y,
                        p0=p0,
                        bounds=bounds,
                        maxfev=maxfev,  # Increase max function evaluations
                    )
                    y_pred = func(x, *popt, config)
                    y_dense = func(x_dense, *popt, config)
                    # create a dictionary for parameters
                    param = {"k": popt[0], "n": popt[1], "a": popt[2]}
                else:
                    p0 = [0.1, 1.0]  # Initial guess for k, a (or m, c for linear)
                    bounds = ([0, -np.inf], [np.inf, np.inf])
                    popt, _ = curve_fit(
                        lambda x, k, a: func(x, k, a, config),
                        x,
                        y,
                        p0=p0,
                        bounds=bounds,
                        maxfev=maxfev,  # Increase max function evaluations
                    )
                    y_pred = func(x, *popt, config)
                    y_dense = func(x_dense, *popt, config)
                    if func_name == "linear":
                        # For linear, popt[0] is slope (m) and popt[1] is intercept (c)
                        param = {"m": popt[0], "c": popt[1]}
                    else:
                        param = {"k": popt[0], "a": popt[1]}
            except:
                # Try with negative initial guesses
                if func_name == "power":
                    p0 = initial_guesses["negative"]  # Initial guess for k, n, a
                    bounds = ([-np.inf, -np.inf, -np.inf], [0, 0, np.inf])
                    popt, _ = curve_fit(
                        lambda x, k, n, a: func(x, k, n, a, config),
                        x,
                        y,
                        p0=p0,
                        bounds=bounds,
                        maxfev=maxfev,  # Increase max function evaluations
                    )
                    y_pred = func(x, *popt, config)
                    y_dense = func(x_dense, *popt, config)
                    # create a dictionary for parameters
                    param = {"k": popt[0], "n": popt[1], "a": popt[2]}
                else:
                    p0 = [0.1, -1.0]  # Initial guess for k, a (or m, c for linear)
                    bounds = ([-np.inf, -np.inf], [0, np.inf])
                    popt, _ = curve_fit(
                        lambda x, k, a: func(x, k, a, config),
                        x,
                        y,
                        p0=p0,
                        bounds=bounds,
                        maxfev=maxfev,  # Increase max function evaluations
                    )
                    y_pred = func(x, *popt, config)
                    y_dense = func(x_dense, *popt, config)
                    if func_name == "linear":
                        # For linear, popt[0] is slope (m) and popt[1] is intercept (c)
                        param = {"m": popt[0], "c": popt[1]}
                    else:
                        param = {"k": popt[0], "a": popt[1]}
            mse = np.mean((y_pred - y) ** 2)
            auc = np.trapz(y_dense, x=x_dense)

            # Generate legend description
            shift_str = f"(x - {x_shift:.2f})" if x_shift != 0 else "x"
            y_shift_str = f" + {y_shift:.2f}" if y_shift != 0 else ""
            inv_str = "1/" if inverse else ""
            if func_name == "linear":
                legend_desc = f"Function: linear\nFit: y = {inv_str}({popt[0]:.4f}{shift_str} + {popt[1]:.4f}){y_shift_str}"
                if not inverse:
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    ss_res = np.sum((y - y_pred) ** 2)
                    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                    legend_desc += f"\nR²: {r_squared:.4f}"
            elif func_name == "exponential":
                legend_desc = f"Function: exponential\nFit: y = {inv_str}({popt[1]:.4f}exp(-{popt[0]:.4f}{shift_str})){y_shift_str}"
            elif func_name == "gaussian":
                legend_desc = f"Function: gaussian\nFit: y = {inv_str}({popt[1]:.4f}exp(-{popt[0]:.4f}{shift_str}^2)){y_shift_str}"
            elif func_name == "hyperbolic":
                legend_desc = f"Function: hyperbolic\nFit: y = {inv_str}({popt[1]:.4f}/(1 + {popt[0]:.4f}{shift_str})){y_shift_str}"
            elif func_name == "power":
                legend_desc = f"Function: power\nFit: y = {inv_str}({popt[2]:.4f}/(1 + {popt[0]:.4f}{shift_str})^{popt[1]:.4f}){y_shift_str}"

            fits[func_name] = {
                "param": param,
                "mse": mse,
                "y_dense": y_dense,
                "auc": auc,
                "legend_desc": legend_desc,
            }
        except RuntimeError:
            print(f"Warning: Curve fitting failed for {func_name} function.")

    # Select best fit based on lowest MSE
    if not fits:
        raise ValueError("No functions could be fitted successfully.")
    best_fit_name = min(fits, key=lambda x: fits[x]["mse"])
    best_fit_dict = fits[best_fit_name]
    best_fit_dict["function_name"] = best_fit_name

    # Plotting
    if plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, color="blue", label="Data Points", alpha=0.6)
        auc = best_fit_dict.pop("auc", np.nan)
        plt.plot(
            x_dense,
            best_fit_dict["y_dense"],
            color="red",
            label=best_fit_dict["legend_desc"],
            linewidth=2,
        )
        # Shade the area under the curve
        plt.fill_between(
            x_dense,
            best_fit_dict["y_dense"],
            0,  # baseline
            label=f"AUC: {auc:.4f}",
            color="red",
            alpha=0.15,
        )

        plt.legend()
        plt.title(f"Best fit: {best_fit_name} function")
        plt.xlabel("Parameter Range")
        plt.ylabel("Values")
        plt.grid(True, alpha=0.2)

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(
                Path(save_dir).joinpath(f"best_fit_{best_fit_name}.png"),
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()
        plt.close()

    return best_fit_dict
