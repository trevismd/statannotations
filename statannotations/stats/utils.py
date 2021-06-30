import warnings

import numpy as np

from statannotations.utils import raise_expected_got


def check_pvalues(p_values):
    if np.ndim(p_values) > 1:
        raise_expected_got(
            'Scalar or list-like', 'argument `p_values`', p_values
        )


def check_num_comparisons(num_comparisons):
    if num_comparisons != 'auto':
        try:
            # Raise a TypeError if num_comparisons is not numeric, and raise
            # an AssertionError if it isn't int-like.
            assert np.ceil(num_comparisons) == num_comparisons
            if not num_comparisons:
                raise ValueError
        except (AssertionError, TypeError):
            raise_expected_got(
                'Int or `auto`', '`num_comparisons`', num_comparisons
            )
        except ValueError:
            raise_expected_got(
                'a positive value', '`num_comparisons`', num_comparisons
            )


def check_alpha(alpha):
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be strictly larger than zero and "
                         "smaller than one.")


def get_num_comparisons(p_values, num_comparisons):
    if num_comparisons == 'auto':
        # Infer number of comparisons
        num_comparisons = len(p_values)

    elif num_comparisons < len(p_values):
        # Warn if multiple p_values have been passed and num_comparisons is
        # set manually.
        warnings.warn(
            'Manually-specified `num_comparisons={}` to a smaller value than '
            'the number of p_values to correct ({}).'.format(
                num_comparisons, len(p_values)
            )
        )
    return num_comparisons


def return_results(results_array):
    if len(results_array) == 1:
        # Return a scalar if input was a scalar.
        return results_array[0]
    else:
        return results_array
