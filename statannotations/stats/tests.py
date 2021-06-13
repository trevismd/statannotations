from typing import Union, Callable

import numpy as np

from statannotations.stats.ComparisonsCorrection import ComparisonsCorrection
from statannotations.stats.StatTest import StatTest
from statannotations.stats.StatResult import StatResult
from statannotations.stats.utils import assert_valid_correction_name


IMPLEMENTED_TESTS = ['t-test_ind', 't-test_welch', 't-test_paired',
                     'Mann-Whitney', 'Mann-Whitney-gt', 'Mann-Whitney-ls',
                     'Levene', 'Wilcoxon', 'Kruskal', 'stat_func']


def stat_test(
        box_data1,
        box_data2,
        test_name: str = None,
        comparisons_correction: Union[ComparisonsCorrection, str] = None,
        num_comparisons: int = 1,
        alpha: float = 0.05,
        verbose: int = 1,
        stat_func: Union[StatTest, Callable] = None,
        **stats_params
):
    """Get formatted result of two sample statistical test.

    Arguments
    ---------
    box_data1, box_data2
    test_name:
        Statistical test to run. Not considered if stat_func is provided.
        Otherwise, must be one of:
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
        Method to use for multiple comparisons correction. Currently only the
        Bonferroni correction is implemented to be applied on per-test basis.
    num_comparisons:
        Number of comparisons to use for multiple comparisons correction.
    alpha: Used for pvalue interpretation in case of comparisons_correction.
    verbose:
        Verbosity parameter, see add_stat_annotation
    stat_func:
        StatTest instance or callable with interface (X1, X2, **stats_params)
         and returning a stat and a pvalue,
    stats_params
        Additional keyword arguments to pass to scipy stats function or to
        stat_func.
        If test_name is stat_func, the keyword "test_long_name" and
        "test_short_name" are also expected.

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

    get_stat_result = None
    result = StatResult(None, '', None, None, np.nan)

    if stat_func is not None:
        if test_name is not None:
            print("Warning: `using stat_test` over `test_name` argument. "
                  "To provide the name arguments for `stat_test`, use "
                  "`test_long_name` and `test_short_name`")

        if isinstance(stat_func, StatTest):
            get_stat_result = stat_func

        else:
            if callable(stat_func):
                try:
                    test_long_name = stats_params.pop("test_long_name")
                    test_short_name = stats_params.pop("test_short_name")
                except KeyError:
                    raise KeyError("All expected values were not found in "
                                   "stats_params")

                get_stat_result = StatTest(
                    stat_func, test_long_name, test_short_name)
            else:
                raise ValueError("Incorrect parameter {stat_func} for "
                                 "`stat_test`")

    elif test_name is not None:
        get_stat_result = StatTest.from_library(test_name)

    if get_stat_result is not None:
        result = get_stat_result(box_data1, box_data2,
                                 alpha=alpha, **stats_params)

    # Optionally, run multiple comparisons correction that can independently be
    # applied to each pval
    if comparisons_correction is not None and comparisons_correction.type == 0:
        result.pval = comparisons_correction(result.pval, num_comparisons)
        result.correction_method = comparisons_correction.name

    return result
