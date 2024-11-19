from functools import partial
from typing import Union, List

import numpy as np

from statannotations.utils import check_is_in

try:
    from statsmodels.stats.multitest import multipletests

except ImportError:
    multipletests = None

# For user convenience
methods_names = {'bonferroni': 'Bonferroni',
                 'bonf': 'Bonferroni',
                 'Bonferroni': 'Bonferroni',
                 'holm-bonferroni': 'Holm-Bonferroni',
                 'HB': 'Holm-Bonferroni',
                 'Holm-Bonferroni': 'Holm-Bonferroni',
                 'holm': 'Holm-Bonferroni',
                 'benjamini-hochberg': 'Benjamini-Hochberg',
                 'BH': 'Benjamini-Hochberg',
                 'fdr_bh': 'Benjamini-Hochberg',
                 'Benjamini-Hochberg': 'Benjamini-Hochberg',
                 'fdr_by': 'Benjamini-Yekutieli',
                 'Benjamini-Yekutieli': 'Benjamini-Yekutieli',
                 'BY': 'Benjamini-Yekutieli'
                 }

# methods_data represents function and method type:
# Type 0 means per-pvalue correction with modification
# Type 1 means correction of interpretation of set of pvalues

methods_data = {'Bonferroni': ["bonferroni", 0],
                'Holm-Bonferroni': ["holm", 1],
                'Benjamini-Hochberg': ["fdr_bh", 1],
                'Benjamini-Yekutieli': ["fdr_by", 1]}

IMPLEMENTED_METHODS = [name for name in methods_names.keys()]


def get_validated_comparisons_correction(comparisons_correction):
    if comparisons_correction is None:
        return

    if isinstance(comparisons_correction, str):
        check_valid_correction_name(comparisons_correction)
        try:
            comparisons_correction = ComparisonsCorrection(
                comparisons_correction)
        except ImportError as e:
            raise ImportError(f"{e} Please install statsmodels or pass "
                              f"`comparisons_correction=None`.")

    if not (isinstance(comparisons_correction, ComparisonsCorrection)):
        raise ValueError("comparisons_correction must be a statmodels "
                         "method name or a ComparisonCorrection instance")

    return comparisons_correction


def get_correction_parameters(name):
    if name is None:
        return None

    if name not in IMPLEMENTED_METHODS:
        raise NotImplementedError(f"Correction method {str(name)} is not "
                                  f"implemented.")

    correction_method = methods_names[name]

    return methods_data[correction_method]


def check_valid_correction_name(name):
    check_is_in(name, IMPLEMENTED_METHODS + [None],
                label='argument `comparisons_correction`')


def _get_correction_attributes(method: Union[str, callable], alpha, name,
                               method_type, corr_kwargs):
    if isinstance(method, str):
        if multipletests is None:
            raise ImportError("The statsmodels package is required to use "
                              "one of the multiple comparisons correction "
                              "methods proposed in statannotations.")

        name = name if name is not None else method
        method, method_type = get_correction_parameters(method)
        if corr_kwargs is None:
            corr_kwargs = {}
        func = partial(multipletests, alpha=alpha, method=method,
                       **corr_kwargs)

    else:
        if name is None:
            raise ValueError("A method name must be provided if passing a "
                             "function")
        if method_type is None:
            raise ValueError("A method type must be provided if passing a "
                             "function")
        func = method
        method = None
        if corr_kwargs is not None:
            func = partial(func, **corr_kwargs)

    return name, method, method_type, func


class ComparisonsCorrection(object):
    def __init__(self, method: Union[str, callable], alpha: float = 0.05,
                 name: str = None, method_type: int = None,
                 statsmodels_api: bool = True, corr_kwargs: dict = None):
        """
        Wrapper for a multiple testing correction method. All arguments must be
        provided here so that it can be called only with the pvalues.

        :param method:
            str that matches one of statsmodel multipletests methods or a
            callable
        :param alpha:
            If using statsmodel: FWER, family-wise error rate, else use
            `corr_kwargs`
        :param name: Necessary if `method` is a function, optional otherwise
        :param method_type:
            type 0: Per-pvalue correction with modification
            type 1 function: Correction of interpretation of set of
            (ordered) pvalues
        :param statsmodels_api:
            statsmodels multipletests returns reject and pvals_corrected as
            first and second values, which correspond to our type 1 and 0
            method_type. Set to False if the used method only returns
            the correct array
        :param corr_kwargs: additional parameters for correction method
        """

        self.name, self.method, self.type, self.func = \
            _get_correction_attributes(method, alpha, name, method_type,
                                       corr_kwargs)
        self.alpha = alpha
        self.statsmodels_api = statsmodels_api
        self.name = methods_names.get(self.name, self.name)

        func = self.func
        while isinstance(func, partial):
            func = func.func

        self.document(func)

    def document(self, func):
        self.__doc__ = (
            f"ComparisonCorrection wrapper for the function "
            f"'{func.__name__}', providing the attributes\n"
            f"    `name`: {self.name}\n"
            f"    `type`: {self.type} \n"
            f"    `alpha (FWER)`: {self.alpha} \n")

        if self.method is not None:
            self.__doc__ += f"    `method`: {self.method} \n"

        self.__doc__ += (
            f"With\n"
            f"    type 0 function: Per-pvalue correction with modification\n"
            f"    type 1 function: Correction of interpretation of set of "
            f"pvalues\n"
            f"------ Original function documentation ------ \n"
            f"{func.__doc__}")

    def __call__(self, pvalues: Union[List[float], float],
                 num_comparisons=None):
        try:
            # Will raise TypeError if scalar value
            n_vals = len(pvalues)
            got_pvalues_in_array = True

        except TypeError:
            n_vals = 1
            pvalues = np.array([pvalues])
            got_pvalues_in_array = False

        if num_comparisons is not None:
            if num_comparisons < n_vals:
                raise ValueError("Number of comparisons must be at least "
                                 "as long as the passed number of pvalues")

            pvalues = np.pad(pvalues, (0, int(num_comparisons) - n_vals),
                             mode="constant", constant_values=1)

        result = self.func(pvalues)

        # See multipletests return values for reject and pvals_corrected
        if self.statsmodels_api:
            result = result[int(not self.type)]

        if num_comparisons is not None and num_comparisons > n_vals:
            result = result[:n_vals]

        return result if got_pvalues_in_array else result[0]

    def _apply_type1_comparisons_correction(self, test_result_list):
        original_pvalues = [result.pvalue for result in test_result_list]

        significant_pvalues = self(original_pvalues)
        for is_significant, result in zip(significant_pvalues,
                                          test_result_list):
            result.correction_method = self.name
            result.corrected_significance = is_significant

    def _apply_type0_comparisons_correction(self, test_result_list):
        for result in test_result_list:
            result.correction_method = self.name
            result.corrected_significance = (
                    result.pvalue < self.alpha
                    or np.isclose(result.pvalue, self.alpha))

    def apply(self, test_result_list):

        if self.type == 1:
            self._apply_type1_comparisons_correction(test_result_list)

        else:
            self._apply_type0_comparisons_correction(test_result_list)
