from typing import Callable

from scipy import stats

from statannotations.stats.StatResult import StatResult


# Also example for how to add other functions
def wilcoxon(box_data1, box_data2, verbose=1, **stats_params):
    """
    This function provides the equivalent behavior from earlier versions of
    statannot/statannotations.
    """
    zero_method = stats_params.pop('zero_method', None)
    if zero_method is None:
        return stats.wilcoxon(box_data1, box_data2, **stats_params)
    if zero_method == "AUTO":
        zero_method = len(box_data1) <= 20 and "pratt" or "wilcox"
        if verbose >= 1:
            print("Using zero_method ", zero_method)

    return stats.wilcoxon(box_data1, box_data2,
                          zero_method=zero_method, **stats_params)


class StatTest:
    @staticmethod
    def from_library(test_name: str) -> 'StatTest':

        # Switch to run scipy.stats hypothesis test.
        if test_name == 'Levene':
            return StatTest(stats.levene, 'Levene test of variance', 'levene')

        elif test_name == 'Mann-Whitney':
            return StatTest(stats.mannwhitneyu,
                            'Mann-Whitney-Wilcoxon test two-sided', 'M.W.W.',
                            'U_stat', alternative='two-sided')

        elif test_name == 'Mann-Whitney-gt':
            return StatTest(stats.mannwhitneyu,
                            'Mann-Whitney-Wilcoxon test greater', 'M.W.W.',
                            'U_stat', alternative='greater')

        elif test_name == 'Mann-Whitney-ls':
            return StatTest(stats.mannwhitneyu,
                            'Mann-Whitney-Wilcoxon test smaller', 'M.W.W.',
                            'U_stat', alternative='less')

        elif test_name == 't-test_ind':
            return StatTest(stats.ttest_ind,
                            't-test independent samples', 't-test_ind',
                            't')

        elif test_name == 't-test_welch':
            return StatTest(stats.ttest_ind,
                            'Welch\'s t-test independent samples',
                            't-test_welch', 't', equal_var=False)

        elif test_name == 't-test_paired':
            return StatTest(stats.ttest_rel,
                            't-test paired samples', 't-test_rel',
                            't')

        elif test_name == 'Wilcoxon':
            return StatTest(stats.wilcoxon,
                            'Wilcoxon test (paired samples)', 'Wilcoxon')

        elif test_name == 'Wilcoxon-legacy':
            return StatTest(wilcoxon,
                            'Wilcoxon test (paired samples)', 'Wilcoxon')

        elif test_name == 'Kruskal':
            return StatTest(stats.kruskal,
                            'Kruskal-Wallis paired samples', 'Kruskal')

        raise NotImplementedError(
            f"Test named {test_name} is not implemented, or wrongly spelled")

    def __init__(self, func: Callable, test_long_name: str,
                 test_short_name: str, stat_name: str = "Stat",
                 alpha: float = 0.05, *args, **kwargs):
        """
        Wrapper for any statistical function to use with statannotations.
        The function must be callable with two 1D collections of data points as
        first and second arguments, can have any number of additional arguments
        , and must return a) a statistical value, and b) a pvalue.

        :param func:
            Function to be called
        :param test_long_name:
            Description to be used in results logging
        :param test_short_name:
            Text to be used on graph annotation to describe the test
        :param stat_name:
            Name of the statistic in results logging (t, W, F, etc)
        :param args
            Positional arguments to provide `func` at each call
        :param kwargs:
            Keyword arguments to provide `func` at each call
        """
        self._func = func
        self._test_long_name = test_long_name
        self._test_short_name = test_short_name
        self._stat_name = stat_name
        self._alpha = alpha
        self. args = args
        self.kwargs = kwargs
        self.__doc__ = (
            f"Wrapper for the function "
            f"'{func.__name__}', providing the attributes\n"
            f"    `test_long_name`: {test_long_name}\n"
            f"    `test_short_name`: {test_short_name} \n"
            f"------ Original function documentation ------ \n"
            f"{func.__doc__}")

    def __call__(self, box_data1, box_data2, alpha=0.05,
                 **stat_params):

        stat, pval = self._func(box_data1, box_data2, *self.args,
                                **{**self.kwargs, **stat_params})

        return StatResult(self._test_long_name, self._test_short_name,
                          self._stat_name, stat, pval, alpha=alpha)
