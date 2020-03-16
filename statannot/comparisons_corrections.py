import warnings

import numpy as np

from .utils import raise_expected_got


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


def bonferroni(p_values, num_comparisons='auto'):
    """Apply Bonferroni correction for multiple comparisons.

    The Bonferroni correction is defined as
        p_corrected = min(num_comparisons * p, 1.0).

    Arguments
    ---------
    p_values: scalar or list-like
        One or more p_values to correct.
    num_comparisons: int or `auto`
        Number of comparisons. Use `auto` to infer the number of comparisons
        from the length of the `p_values` list.

    Returns
    -------
    Scalar or numpy array of corrected p-values.

    """
    # Input checks.
    check_pval_correction_input_values(p_values, num_comparisons)

    # Coerce p_values to numpy array.
    p_values_array = np.atleast_1d(p_values)

    num_comparisons = get_num_comparisons(p_values_array, num_comparisons)

    # Apply correction by multiplying p_values and thresholding at p=1.0
    p_values_array *= num_comparisons
    p_values_array = np.min(
        [p_values_array, np.ones_like(p_values_array)], axis=0
    )

    return return_results(p_values_array)


def holm_bonferroni(p_values, num_comparisons='auto', alpha=0.05):
    """Apply Holm-Bonferroni correction for multiple comparisons.

    The Holm-Bonferroni correction is implemented to fit definition from
        https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method

    Arguments
    ---------
    p_values: scalar or list-like
        One or more p_values to correct.
    num_comparisons: int or `auto`
        Number of comparisons. Use `auto` to infer the number of comparisons
        from the length of the `p_values` list.
    alpha: float
        Significance level (accepted type I error rate)
    Returns
    -------
    Scalar or numpy array of boolean representing significance of p-values

    """
    # Input checks.
    check_pval_correction_input_values(p_values, num_comparisons)

    # Coerce p_values to numpy array.
    p_values_array = np.atleast_1d(p_values)

    num_comparisons = get_num_comparisons(p_values_array, num_comparisons)

    # Order p_values
    ordered_p_val_idx = np.argsort(p_values_array)
    last_index_to_reject = 0
    output_p_values = np.zeros_like(p_values_array)

    # Set to True rejected hypotheses (pval is sufficiently low)
    while (p_values_array[ordered_p_val_idx[last_index_to_reject]] <=
           alpha / (num_comparisons - last_index_to_reject)):
        output_p_values[ordered_p_val_idx[last_index_to_reject]] = True
        last_index_to_reject += 1
        if last_index_to_reject == len(p_values_array):
            break

    # Set to False non-rejected hypotheses (pval is to high even if below alpha)
    else:
        counter = last_index_to_reject
        while counter < len(p_values_array):
            output_p_values[ordered_p_val_idx[counter]] = False
            counter += 1

    return return_results(output_p_values)


def benjamin_hochberg(p_values, num_comparisons='auto', alpha=0.05):
    """Apply Benjamin-Hochberg correction for multiple comparisons.

    The  Benjamin-Hochberg correction is implemented to fit definition from
        https://en.wikipedia.org/wiki/False_discovery_rate
    Arguments
    ---------
    p_values: scalar or list-like
        One or more p_values to correct.
    num_comparisons: int or `auto`
        Number of comparisons. Use `auto` to infer the number of comparisons
        from the length of the `p_values` list.
    alpha: float
        Significance level (accepted type I error rate)
    Returns
    -------
    Scalar or numpy array of boolean representing significance of p-values

    """
    # Input checks.
    check_pval_correction_input_values(p_values, num_comparisons)

    # Coerce p_values to numpy array.
    p_values_array = np.atleast_1d(p_values)

    num_comparisons = get_num_comparisons(p_values_array, num_comparisons)

    # Order p_values
    ordered_p_val_idx = np.argsort(p_values_array)

    last_index_to_reject = 0
    output_p_values = np.zeros_like(p_values_array)

    # Set to True rejected hypotheses (pval is sufficiently low)
    while (p_values_array[ordered_p_val_idx[last_index_to_reject]] <=
           (last_index_to_reject + 1) * alpha / num_comparisons):
        output_p_values[ordered_p_val_idx[last_index_to_reject]] = True
        last_index_to_reject += 1
        if last_index_to_reject == len(p_values_array):
            break

    # Set to False non-rejected hypotheses (pval is to high even if below alpha)
    else:
        counter = last_index_to_reject
        while counter < len(p_values_array):
            output_p_values[ordered_p_val_idx[counter]] = False
            counter += 1

    return return_results(output_p_values)
