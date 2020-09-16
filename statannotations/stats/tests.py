import numpy as np
from scipy import stats

from statannotations.stats.ComparisonsCorrection import ComparisonsCorrection
from statannotations.stats.StatResult import StatResult
from statannotations.stats.utils import assert_valid_correction_name


IMPLEMENTED_TESTS = ['t-test_ind', 't-test_welch', 't-test_paired',
                     'Mann-Whitney', 'Mann-Whitney-gt', 'Mann-Whitney-ls',
                     'Levene', 'Wilcoxon', 'Kruskal', 'stat_func']


def stat_test(
        box_data1,
        box_data2,
        test_name,
        comparisons_correction=None,
        num_comparisons=1,
        alpha=0.05,
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
        - `stat_func` - Which means a function is passed in stats_params
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
    alpha: Used for pvalue interpretation in case of comparisons_correction.
    verbose: int
        Verbosity parameter, see add_stat_annotation
    stats_params
        Additional keyword arguments to pass to scipy stats functions.
        A keyword "stat_func" is expected to be a callable with interface
         (X1, X2, **stats_params) and return a stat and a pvalue,
          a keyword "test_long_name" and "test_short_name" are also expected.

    Returns
    -------
    StatResult object with formatted result of test.

    """
    # Check arguments.
    if (isinstance(comparisons_correction, ComparisonsCorrection)
            or comparisons_correction is None):
        pass
    else:
        assert_valid_correction_name(comparisons_correction)
        comparisons_correction = ComparisonsCorrection(comparisons_correction)

    # Switch to run scipy.stats hypothesis test.
    if test_name == 'Levene':
        stat, pval = stats.levene(box_data1, box_data2, **stats_params)
        result = StatResult('Levene test of variance', 'levene',
                            'stat_value', stat, pval, alpha=alpha)
    elif test_name == 'Mann-Whitney':
        u_stat, pval = stats.mannwhitneyu(box_data1, box_data2,
                                          alternative='two-sided',
                                          **stats_params)

        result = StatResult('Mann-Whitney-Wilcoxon test two-sided', 'M.W.W.',
                            'U_stat', u_stat, pval, alpha=alpha)

    elif test_name == 'Mann-Whitney-gt':
        u_stat, pval = stats.mannwhitneyu(
            box_data1, box_data2, alternative='greater', **stats_params)
        result = StatResult('Mann-Whitney-Wilcoxon test greater', 'M.W.W.',
                            'U_stat', u_stat, pval, alpha=alpha)

    elif test_name == 'Mann-Whitney-ls':
        u_stat, pval = stats.mannwhitneyu(box_data1, box_data2,
                                          alternative='less', **stats_params)
        result = StatResult('Mann-Whitney-Wilcoxon test smaller', 'M.W.W.',
                            'U_stat', u_stat, pval, alpha=alpha)

    elif test_name == 't-test_ind':
        stat, pval = stats.ttest_ind(a=box_data1, b=box_data2, **stats_params)
        result = StatResult('t-test independent samples', 't-test_ind',
                            'stat_value', stat, pval, alpha=alpha)

    elif test_name == 't-test_welch':
        stat, pval = stats.ttest_ind(
            a=box_data1, b=box_data2, equal_var=False, **stats_params)
        result = StatResult('Welch\'s t-test independent samples',
                            't-test_welch',
                            'stat_value', stat, pval, alpha=alpha)

    elif test_name == 't-test_paired':
        stat, pval = stats.ttest_rel(a=box_data1, b=box_data2, **stats_params)
        result = StatResult('t-test paired samples', 't-test_rel',
                            'stat_value', stat, pval, alpha=alpha)

    elif test_name == 'Wilcoxon':
        zero_method = stats_params.pop('zero_method', None)
        if zero_method is None:
            zero_method = len(box_data1) <= 20 and "pratt" or "wilcox"

        if verbose >= 1:
            print("Using zero_method ", zero_method)

        stat, pval = stats.wilcoxon(box_data1, box_data2,
                                    zero_method=zero_method, **stats_params)
        result = StatResult('Wilcoxon test (paired samples)', 'Wilcoxon',
                            'stat_value', stat, pval, alpha=alpha)

    elif test_name == 'Kruskal':
        stat, pval = stats.kruskal(box_data1, box_data2, **stats_params)
        result = StatResult('Kruskal-Wallis paired samples', 'Kruskal',
                            'stat_value', stat, pval, alpha=alpha)

    else:
        if test_name == "stat_func":
            try:
                test_short_name = stats_params.pop("test_short_name")
                test_long_name = stats_params.pop("test_long_name")
                stat_func = stats_params.pop("stat_func")

            except KeyError:
                raise KeyError("All expected values were not found in "
                               "stats_params")

            stat, pval = stat_func(box_data1, box_data2, **stats_params)
            result = StatResult(test_long_name, test_short_name,
                                'stat_value', stat, pval, alpha=alpha)
        else:
            result = StatResult(None, '', None, None, np.nan)

    # Optionally, run multiple comparisons correction that can independently be
    # applied to each pval
    if comparisons_correction is not None and comparisons_correction.type == 0:
        result.pval = comparisons_correction(result.pval, num_comparisons)
        result.correction_method = comparisons_correction.name

    return result
