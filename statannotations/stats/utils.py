import warnings

import numpy as np

import statannotations.stats.ComparisonsCorrection as ComparisonsCorrection
from statannotations.utils import raise_expected_got, check_is_in


def check_valid_correction_name(name):
    check_is_in(name, ComparisonsCorrection.IMPLEMENTED_METHODS + [None],
                label='argument `comparisons_correction`')


def check_pval_correction_input_values(p_values, num_comparisons):
    if np.ndim(p_values) > 1:
        raise_expected_got(
            'Scalar or list-like', 'argument `p_values`', p_values
        )

    if num_comparisons != 'auto':
        try:
            # Raise a TypeError if num_comparisons is not numeric, and raise
            # an AssertionError if it isn't int-like.
            assert np.ceil(num_comparisons) == num_comparisons
        except (AssertionError, TypeError):
            raise_expected_got(
                'Int or `auto`', 'argument `num_comparisons`', num_comparisons
            )


def check_alpha(alpha):
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be strictly larger than zero and "
                         "smaller than one.")


def get_num_comparisons(p_values_array, num_comparisons):
    if num_comparisons == 'auto':
        # Infer number of comparisons
        num_comparisons = len(p_values_array)

    elif 1 < len(p_values_array) != num_comparisons:
        # Warn if multiple p_values have been passed and num_comparisons is
        # set manually.
        warnings.warn(
            'Manually-specified `num_comparisons={}` differs from number of '
            'p_values to correct ({}).'.format(
                num_comparisons, len(p_values_array)
            )
        )
    return num_comparisons


def return_results(results_array):
    if len(results_array) == 1:
        # Return a scalar if input was a scalar.
        return results_array[0]
    else:
        return results_array
