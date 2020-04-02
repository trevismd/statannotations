import numpy as np

from statannot.comparisons_corrections.utils import check_pval_correction_input_values, return_results, \
    get_num_comparisons


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

    # Apply correction by multiplying p_values and using max p=1.0
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
    significant_comparisons = np.zeros_like(p_values_array, dtype=np.bool)

    # Set to True rejected hypotheses (pval is sufficiently low)
    while (p_values_array[ordered_p_val_idx[last_index_to_reject]] <=
           alpha / (num_comparisons - last_index_to_reject)):
        significant_comparisons[ordered_p_val_idx[last_index_to_reject]] = True
        last_index_to_reject += 1
        if last_index_to_reject == len(p_values_array):
            break

    # Set to False non-rejected hypotheses (pval is to high even if below alpha)
    else:
        for idx in ordered_p_val_idx[last_index_to_reject:]:
            significant_comparisons[idx] = False

    return return_results(significant_comparisons)


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
        from the length of the `p_values` list. If this number is specified,
        the missing p-values will be considered higher, and the specified
        p-values compared to the remaining smallest BH critical values.
    alpha: float
        False discovery rate (Q)
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

    last_index_to_reject = len(p_values_array) - 1
    significant_comparisons = np.zeros_like(p_values_array, dtype=np.bool)

    # Set to False non-rejected hypotheses (pval is too high vs critical value)
    while (p_values_array[ordered_p_val_idx[last_index_to_reject]] >
           (last_index_to_reject + 1) * alpha / num_comparisons):
        significant_comparisons[ordered_p_val_idx[last_index_to_reject]] = False
        last_index_to_reject -= 1
        if last_index_to_reject < 0:
            break

    # Set to True to reject the rest of the hypotheses
    else:
        for idx in ordered_p_val_idx[:last_index_to_reject + 1]:
            significant_comparisons[idx] = True

    return return_results(significant_comparisons)
