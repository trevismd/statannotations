from statannotations.stats.StatResult import StatResult
from typing import Union, List
import numpy as np
import pandas as pd
from operator import itemgetter


def pval_annotation_text(result: Union[List[StatResult], StatResult],
                         pvalue_thresholds: List) -> Union[List[str], str]:
    """

    :param result: StatResult instance or list thereof
    :param pvalue_thresholds: thresholds to use for formatter
    :returns: A List of rendered annotations if a list of StatResults was
        provided, a string otherwise.
    """
    was_list = True

    if not isinstance(result, list):
        was_list = False
        result = [result]

    x1_pval = np.array([res.pvalue for res in result])

    pvalue_thresholds = sorted(pvalue_thresholds, key=itemgetter(0),
                               reverse=True)

    x_annot = pd.Series(["" for _ in x1_pval])

    for i, threshold in enumerate(pvalue_thresholds[:-1]):
        condition = ((x1_pval <= threshold[0])
                     & (pvalue_thresholds[i + 1][0] < x1_pval))
        x_annot[condition] = threshold[1]

    x_annot[x1_pval <= pvalue_thresholds[-1][0]] = pvalue_thresholds[-1][1]

    x_annot = [f"{star}{res.significance_suffix}"
               for star, res in zip(x_annot, result)]

    return x_annot if was_list else x_annot[0]


def simple_text(result: StatResult, pvalue_format, pvalue_thresholds) -> str:
    """
    Generates simple text for test name and pvalue.

    :param result: StatResult instance
    :param pvalue_format: format string for pvalue
    :param pvalue_thresholds: String to display per pvalue range
    :param result: StatResult
    :returns: simple annotation

    """
    # Sort thresholds
    thresholds = sorted(pvalue_thresholds, key=lambda x: x[0])

    text = result.test_short_name and f"{result.test_short_name} " or ""

    for threshold in thresholds:
        if result.pvalue < threshold[0]:
            pval_text = "p ≤ {}".format(threshold[1])
            break
    else:
        pval_text = "p = {}".format(pvalue_format).format(result.pvalue)

    return text + pval_text + result.significance_suffix
