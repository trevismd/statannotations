from statannotations.stats.StatResult import StatResult
from typing import Union, List
import numpy as np
import pandas as pd


def pval_annotation_text(result: Union[List[StatResult], StatResult], pvalue_thresholds):
    single_value = False

    if isinstance(result, list):
        x1_pval = np.array([res.pval for res in result])
        x1_signif_suff = [res.significance_suffix for res in result]
    else:
        x1_pval = np.array([result.pval])
        x1_signif_suff = [result.significance_suffix]
        single_value = True

    # Sort the threshold array
    pvalue_thresholds = pd.DataFrame(pvalue_thresholds).sort_values(by=0, ascending=False).values
    x_annot = pd.Series(["" for _ in range(len(x1_pval))])

    for i in range(0, len(pvalue_thresholds)):

        if i < len(pvalue_thresholds) - 1:
            condition = (x1_pval <= pvalue_thresholds[i][0]) & (pvalue_thresholds[i + 1][0] < x1_pval)
            x_annot[condition] = pvalue_thresholds[i][1]

        else:
            condition = x1_pval < pvalue_thresholds[i][0]
            x_annot[condition] = pvalue_thresholds[i][1]

    x_annot = pd.Series([f"{star}{signif}" for star, signif in zip(x_annot, x1_signif_suff)])

    return x_annot if not single_value else x_annot.iloc[0]


def simple_text(result: StatResult, pvalue_format, pvalue_thresholds, test_short_name=None):
    """
    Generates simple text for test name and pvalue
    :param result: StatResult instance
    :param pvalue_format: format string for pvalue
    :param test_short_name: Short name of test to show
    :param pvalue_thresholds: String to display per pvalue range
    :return: simple annotation
    """
    # Sort thresholds
    thresholds = sorted(pvalue_thresholds, key=lambda x: x[0])

    # Test name if passed
    text = test_short_name and test_short_name + " " or ""

    for threshold in thresholds:
        if result.pval < threshold[0]:
            pval_text = "p ≤ {}".format(threshold[1])
            break
    else:
        pval_text = "p = {}".format(pvalue_format).format(result.pval)

    return text + pval_text + result.significance_suffix
