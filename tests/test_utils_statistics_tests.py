import numpy as np
import pandas as pd

from neural_analysis.utils.statistics.tests import (
    _auto_select_test_method,
    apply_multiple_correction,
    check_normality,
    statistical_comparison,
)


def test_check_normality():
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 100)
    p_val = check_normality(normal_data)
    assert p_val > 0.05

    non_normal = np.random.exponential(1, 100)
    p_val2 = check_normality(non_normal)
    assert p_val2 < 0.05


def test_auto_select_test_method():
    np.random.seed(42)
    data1 = np.random.normal(0, 1, 30)
    data2 = np.random.normal(0.5, 1, 30)

    method, info = _auto_select_test_method(data1, data2, test_type="paired")
    assert method == "ttest"
    assert "p_normality_diff" in info or "normality_p" in info


def test_apply_multiple_correction():
    pvals = np.array([0.01, 0.02, 0.04, 0.05])
    corrected = apply_multiple_correction(pvals, method="fdr_bh")
    assert np.all(corrected >= pvals)


def test_statistical_comparison_basic():
    df = pd.DataFrame(
        {
            "subject": ["A", "A", "B", "B"],
            "condition": ["pre", "post", "pre", "post"],
            "value": [1.0, 2.0, 1.5, 2.5],
        }
    )

    # We disable plotting to just test stat evaluation
    results = statistical_comparison(
        df=df,
        pair_name_col="subject",
        compare_by="condition",
        value_col="value",  # Ensure this works with single string as original implementation did
        test_type="paired",
        correction_method="none",
    )

    assert "paired" in results
    assert "pvalues" in results["paired"]
    assert "value" in results["paired"]["pvalues"]
    assert len(results["paired"]["pvalues"]["value"]) > 0


def test_mannwhitneyu_cross_df():
    from neural_analysis.utils.statistics.tests import mannwhitneyu_cross_df

    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "condition": ["pre", "post", "pre", "post"],
            "val1": [1.0, 2.0, 1.5, 2.5],
            "val2": [0.1, 0.2, 0.15, 0.25],
        }
    )
    results = mannwhitneyu_cross_df(
        df=df,
        group_by="group",
        compare_by="condition",
        value_col=["val1", "val2"],
    )
    assert len(results) > 0
    assert "val1" in results


def test_sigtest_cross_df():
    from neural_analysis.utils.statistics.tests import sigtest_cross_df

    df = pd.DataFrame(
        {
            "subject": ["A", "A", "B", "B"],
            "condition": ["pre", "post", "pre", "post"],
            "val1": [1.0, 2.0, 1.5, 2.5],
        }
    )
    results = sigtest_cross_df(
        df=df,
        pair_name_col="subject",
        compare_by="condition",
        value_col=["val1"],
        method="wilcoxon",
    )
    assert len(results) > 0
    assert "pvalues" in results
    assert "val1" in results["pvalues"]
