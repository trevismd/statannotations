import numpy as np
from scipy import stats

from statannotations.stats.ComparisonsCorrection import ComparisonsCorrection
from statannotations.stats.StatResult import StatResult
from statannotations.stats.utils import assert_valid_correction_name


IMPLEMENTED_TESTS = ['t-test_ind', 't-test_welch', 't-test_paired',
                     'Mann-Whitney', 'Mann-Whitney-gt', 'Mann-Whitney-ls',
                     'Levene', 'Wilcoxon', 'Kruskal', 'bootstrap', 'paired_bootstrap', 
                     'stat_func']


def stat_test(
        box_data1,
        box_data2,
        test_name,
        comparisons_correction=None,
        num_comparisons=1,
        alpha=0.05,
        verbose=1,
        n_bootstraps=None,
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
        - `bootstrap`
        - `paired_bootstrap`
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

    if 'boot' in test_name and n_bootstraps is None:
        print ('No value specified for the number of bootstraps to perform. Please provide a value for the \'n_boostraps\' variable.')
        raise ValueError

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

    elif test_name == 'bootstrap':
        pval = bootstrap(box_data1, box_data2, n_bootstraps)
        # 'stat' doesn't really apply for a bootstrapped comparison,
        # but it is useful to record the number of bootstraps used.
        # For a bootstrapped p-value, the statistic will be the number of bootstraps.

        result = StatResult('Non-parametric bootstrapped two-sided comparison', 'bootstrap', 
                            'stat_value', n_bootstraps, pval, alpha=alpha)

    elif test_name == 'paired_bootstrap':
        assert n_bootstraps is not None, 'No value specified for the number of bootstraps to perform. Please provide a value for the \'n_boostraps\' variable.'
        pval = paired_bootstrap(box_data1, box_data2, n_bootstraps)
        test_short_name = 'paired_bootstrap'
        result = StatResult('Non-parametric paired bootstrap two-sided comparison', 'paired_bootstrap',
                            'stat_value', n_bootstraps, pval, alpha=alpha)

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

def bootstrap(dataset_a, dataset_b, n):
    """Determine via bootstrapping if the mean of dataset a is 
       significantly different than mean of dataset b.

    Arguments
    ---------
    dataset_a: numpy array of data points
    dataset_b: numpy array of data points
    n: Number of bootstrapped samples to collect.

    Returns
    -------
    Two-sided likelihood of samples being drawn from the same population of data points.

    """
    boot_a = np.array([np.mean(np.random.choice(dataset_a, size=len(dataset_a), replace=True)) for _ in range(n)])
    boot_b = np.array([np.mean(np.random.choice(dataset_b, size=len(dataset_b), replace=True)) for _ in range(n)])

    return min(np.mean(boot_a > boot_b), np.mean(boot_a < boot_b))*2  # two-sided

def paired_bootstrap(dataset_a, dataset_b, n):
    """determine via bootstrapping method if items from one dataset are higher or lower
       than items from another paired dataset a significantly distinct proportion of the time.

    Arguments
    ---------
    dataset_a: numpy array of data points
    dataset_b: numpy array of data points. Must be the same size as dataset_a.
               Test is valid only if dataset_a[n] and dataset_b[n] are paired.
    n: Number of bootstrapped samples to collect.

    Returns
    -------
    Two-sided likelihood of datapoints from dataset_a being higher or lower 
    than paired datapoints from dataset_b.

    """
    if len(dataset_a) != len(dataset_b):
        print ('Paired bootstrap tests can only be performed between two variables of equal size.')
        raise ValueError

    diffs = dataset_a.values - dataset_b.values
    samples = np.array([np.sum(np.random.choice(diffs, size=len(diffs), replace=True)) for _ in range(n)])
    return min(np.mean(samples > 0), np.mean(samples < 0))*2  # two-sided