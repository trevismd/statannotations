import numpy as np
from scipy import stats

from stats.ComparisonsCorrection import ComparisonsCorrection, get_correction_method
from stats.StatResult import StatResult
from utils import assert_valid_correction_name


IMPLEMENTED_TESTS = ['t-test_ind', 't-test_welch', 't-test_paired',
                     'Mann-Whitney', 'Mann-Whitney-gt', 'Mann-Whitney-ls',
                     'Levene', 'Wilcoxon', 'Kruskal']


def stat_test(
        box_data1,
        box_data2,
        test_name,
        comparisons_correction=None,
        num_comparisons=1,
        verbose=1,
        **stats_params
):
    """Get formatted result of two sample statistical test.

    Arguments
    ---------
    box_data1, box_data2
    test_name: str
        Statistical test to run. Must be one of:
        - `Levene`
        - `Mann-Whitney`
        - `Mann-Whitney-gt`
        - `Mann-Whitney-ls`
        - `t-test_ind`
        - `t-test_welch`
        - `t-test_paired`
        - `Wilcoxon`
        - `Kruskal`
    comparisons_correction: str or or ComparisonCorrection or None, default None
        Method to use for multiple comparisons correction. Currently only the
        Bonferroni correction is implemented to be applied on per-test basis.
    num_comparisons: int, default 1
        Number of comparisons to use for multiple comparisons correction.
    verbose: int
        Verbosity parameter, see add_stat_annotation
    stats_params
        Additional keyword arguments to pass to scipy stats functions.

    Returns
    -------
    StatResult object with formatted result of test.

    """
    # Check arguments.
    if isinstance(comparisons_correction, ComparisonsCorrection):
        pass
    else:
        assert_valid_correction_name(comparisons_correction)
        comparisons_correction = get_correction_method(comparisons_correction)

    # Switch to run scipy.stats hypothesis test.
    if test_name == 'Levene':
        stat, pval = stats.levene(box_data1, box_data2, **stats_params)
        result = StatResult(
            'Levene test of variance', 'levene', 'stat_value', stat, pval
        )
    elif test_name == 'Mann-Whitney':
        u_stat, pval = stats.mannwhitneyu(
            box_data1, box_data2, alternative='two-sided', **stats_params
        )
        result = StatResult(
            'Mann-Whitney-Wilcoxon test two-sided',
            'M.W.W.',
            'U_stat',
            u_stat,
            pval,
        )
    elif test_name == 'Mann-Whitney-gt':
        u_stat, pval = stats.mannwhitneyu(
            box_data1, box_data2, alternative='greater', **stats_params
        )
        result = StatResult(
            'Mann-Whitney-Wilcoxon test greater',
            'M.W.W.',
            'U_stat',
            u_stat,
            pval,
        )
    elif test_name == 'Mann-Whitney-ls':
        u_stat, pval = stats.mannwhitneyu(
            box_data1, box_data2, alternative='less', **stats_params
        )
        result = StatResult(
            'Mann-Whitney-Wilcoxon test smaller',
            'M.W.W.',
            'U_stat',
            u_stat,
            pval,
        )
    elif test_name == 't-test_ind':
        stat, pval = stats.ttest_ind(a=box_data1, b=box_data2, **stats_params)
        result = StatResult(
            't-test independent samples', 't-test_ind', 'stat_value', stat, pval
        )
    elif test_name == 't-test_welch':
        stat, pval = stats.ttest_ind(
            a=box_data1, b=box_data2, equal_var=False, **stats_params
        )
        result = StatResult(
            'Welch\'s t-test independent samples',
            't-test_welch',
            'stat_value',
            stat,
            pval,
        )
    elif test_name == 't-test_paired':
        stat, pval = stats.ttest_rel(a=box_data1, b=box_data2, **stats_params)
        result = StatResult(
            't-test paired samples', 't-test_rel', 'stat_value', stat, pval
        )
    elif test_name == 'Wilcoxon':
        zero_method_default = len(box_data1) <= 20 and "pratt" or "wilcox"
        zero_method = stats_params.get('zero_method', zero_method_default)
        if verbose >= 1:
            print("Using zero_method ", zero_method)
        stat, pval = stats.wilcoxon(
            box_data1, box_data2, zero_method=zero_method, **stats_params
        )
        result = StatResult(
            'Wilcoxon test (paired samples)', 'Wilcoxon', 'stat_value', stat, pval
        )
    elif test_name == 'Kruskal':
        stat, pval = stats.kruskal(box_data1, box_data2, **stats_params)
        result = StatResult(
            'Kruskal-Wallis paired samples', 'Kruskal', 'stat_value', stat, pval
        )
    else:
        result = StatResult(None, '', None, None, np.nan)

    # Optionally, run multiple comparisons correction.
    if comparisons_correction.type == 0:  # Correction can separately be applied to each pval
        result.pval = comparisons_correction(result.pval, num_comparisons)
        result.correction_method = comparisons_correction.name

    return result
