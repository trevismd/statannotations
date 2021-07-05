import warnings
from numbers import Number
import numpy as np

from statannotations.utils import raise_expected_got


def check_pvalues(p_values):
    def is_number(value):
        return isinstance(value, Number)

    if np.ndim(p_values) > 1 or not np.all(list(map(is_number, p_values))):
        raise_expected_got(
            'Scalar or list-like', 'argument `p_values`', p_values
        )


def check_num_comparisons(num_comparisons):
    if num_comparisons != 'auto':
        try:
            assert num_comparisons == int(num_comparisons)
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
        num_comparisons = len(p_values)

    elif num_comparisons < len(p_values):
        warnings.warn(
            f'Manually-specified `num_comparisons={num_comparisons}` to a '
            f'smaller value than the number of p_values to correct '
            f'({len(p_values)})')
    return num_comparisons


def return_results(results_array):
    if len(results_array) == 1:
        # Return a scalar if input was a scalar.
        return results_array[0]
    else:
        return results_array
