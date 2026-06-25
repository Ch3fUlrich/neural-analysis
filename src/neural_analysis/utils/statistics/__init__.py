from .tests import (
    check_normality,
    _auto_select_test_method,
    _compute_paired_pvalue,
    _compute_unpaired_pvalue,
    apply_multiple_correction,
    compute_tukey_pvalues,
    mannwhitneyu_cross_df,
    sigtest_cross_df,
)
from .models import FunctionModel, get_auc, get_best_fit
