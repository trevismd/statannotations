from functools import partial
from typing import Union

import numpy as np

try:
    from statsmodels.stats.multitest import multipletests

except ImportError as statsmodel_import_error:
    multipletests = None

else:
    statsmodel_import_error = None

# For user convenience
methods_names = {'bonferroni':          'Bonferroni',
                 'bonf':                'Bonferroni',
                 'Bonferroni':          'Bonferroni',
                 'holm-bonferroni':     'Holm-Bonferroni',
                 'HB':                  'Holm-Bonferroni',
                 'Holm-Bonferroni':     'Holm-Bonferroni',
                 'holm':                'Holm-Bonferroni',
                 'benjamini-hochberg':  'Benjamini-Hochberg',
                 'BH':                  'Benjamini-Hochberg',
                 'fdr_bh':              'Benjamini-Hochberg',
                 'Benjamini-Hochberg':  'Benjamini-Hochberg',
                 'fdr_by':              'Benjamini-Yekutieli',
                 'Benjamini-Yekutieli': 'Benjamini-Yekutieli',
                 'BY':                  'Benjamini-Yekutieli'
                 }

# methods_data represents function and method type:
# Type 0 means per-pvalue correction with modification
# Type 1 means correction of interpretation of set of pvalues

methods_data = {'Bonferroni':        ["bonferroni", 0],
                'Holm-Bonferroni':   ["holm", 1],
                'Benjamini-Hochberg': ["fdr_bh", 1],
                'Benjamini-Yekutieli': ["fdr_by", 1]}

IMPLEMENTED_METHODS = [name for name in methods_names.keys()]


def get_correction_parameters(name):
    if name is None:
        return None

    if name not in IMPLEMENTED_METHODS:
        raise NotImplementedError(f"Correction method {str(name)} is not "
                                  f"implemented.")

    correction_method = methods_names[name]

    return methods_data[correction_method]


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

        if isinstance(method, str):
            if multipletests is None:
                raise ImportError(statsmodel_import_error)

            self.name = name if name is not None else method
            self.method, self.type = get_correction_parameters(method)
            if corr_kwargs is None:
                corr_kwargs = {}
            self.func = partial(multipletests, alpha=alpha, method=self.method,
                                **corr_kwargs)

        if callable(method):
            if name is None:
                raise ValueError("A method name must be provided if passing a "
                                 "function")
            if method_type is None:
                raise ValueError("A method type must be provided if passing a "
                                 "function")
            self.method = None
            self.name = name
            self.type = method_type
            self.func = method
            if corr_kwargs is not None:
                self.func = partial(self.func, **corr_kwargs)

        self.alpha = alpha
        self.statsmodels_api = statsmodels_api
        self.name = methods_names.get(self.name, self.name)
        func = self.func
        while isinstance(func, partial):
            func = func.func

        self.__doc__ = (
            f"ComparisonCorrection wrapper for the function "
            f"'{func.__name__}', providing the attributes\n"
            f"    `name`: {self.name}\n"
            f"    `type`: {self.type} \n"
            f"    `alpha (FWER)`: {self.alpha} \n")

        if self.method is not None:
            # Used as multipletests parameter
            self.__doc__ += f"    `method`: {self.method} \n"

        self.__doc__ += (
            f"With\n"
            f"    type 0 function: Per-pvalue correction with modification\n"
            f"    type 1 function: Correction of interpretation of set of "
            f"pvalues\n"
            f"------ Original function documentation ------ \n"
            f"{func.__doc__}")

    def __call__(self, pvalues, m=None):
        try:
            # Will raise TypeError if scalar value
            n_vals = len(pvalues)
            is_array = True

        except TypeError:
            n_vals = 1
            pvalues = np.array([pvalues])
            is_array = False

        # m is number of comparisons
        # If specified, as many "1" as needed will be added to the pvalues list
        if m is not None and m != n_vals:
            if isinstance(m, float):
                m = int(m)
            if m < n_vals:
                raise ValueError("Number of comparisons must be at least "
                                 "as long as the passed number of pvalues")
            else:
                n_pvalues = np.ones(m)
                n_pvalues[:n_vals] = pvalues
                result = self.func(n_pvalues)
        else:
            result = self.func(pvalues)

        # See multipletests return values for reject and pvals_corrected
        if self.statsmodels_api:
            result = result[1] if self.type == 0 else result[0]

        if m is not None and m > n_vals:
            result = result[:n_vals]

        return result if is_array else result[0]
