from typing import Union, Callable

import numpy as np

from statannotations.stats.ComparisonsCorrection import ComparisonsCorrection
from statannotations.stats.StatResult import StatResult
from statannotations.stats.StatTest import StatTest
from statannotations.stats.utils import check_valid_correction_name

IMPLEMENTED_TESTS = ['t-test_ind', 't-test_welch', 't-test_paired',
                     'Mann-Whitney', 'Mann-Whitney-gt', 'Mann-Whitney-ls',
                     'Levene', 'Wilcoxon', 'Kruskal']


def stat_test(
        box_data1,
        box_data2,
        test: Union[StatTest, str] = None,
        comparisons_correction: Union[ComparisonsCorrection, str] = None,
        num_comparisons: int = 1,
        alpha: float = 0.05,
        **stats_params
):
    """Get formatted result of two-sample statistical test.

    Arguments
    ---------
    box_data1, box_data2:
        Data
    test:
        Statistical test to run. Either a `StatTest` instance or one of:
        - `Levene`
        - `Mann-Whitney`
        - `Mann-Whitney-gt`
        - `Mann-Whitney-ls`
        - `t-test_ind`
        - `t-test_welch`
        - `t-test_paired`
        - `Wilcoxon`
        - `Kruskal`
    comparisons_correction:
        Method to use for multiple comparisons correction. Either a
        `ComparisonsCorrection` instance or one of (interfacing statsmodels):
        - Bonferroni
        - Holm-Bonferroni
        - Benjamini-Hochberg
        - Benjamini-Yekutieli
    num_comparisons:
        Number of comparisons to use for multiple comparisons correction.
    alpha: Used for pvalue interpretation in case of comparisons_correction.
    stats_params
        Additional keyword arguments to pass to the test function

    Returns
    -------
    StatResult object with formatted result of test.

    """
    # Check arguments.
    if (isinstance(comparisons_correction, ComparisonsCorrection)
            or comparisons_correction is None):
        pass
    else:
        check_valid_correction_name(comparisons_correction)
        comparisons_correction = ComparisonsCorrection(comparisons_correction)

    if test is None:
        result = StatResult(None, '', None, None, np.nan)

    else:
        if isinstance(test, StatTest):
            get_stat_result = test
        else:
            get_stat_result = StatTest.from_library(test)

        result = get_stat_result(
            box_data1, box_data2, alpha=alpha, **stats_params)

    # Optionally, run multiple comparisons correction that can independently be
    # applied to each pval
    if comparisons_correction is not None and comparisons_correction.type == 0:
        result.pval = comparisons_correction(result.pval, num_comparisons)
        result.correction_method = comparisons_correction.name

    return result
