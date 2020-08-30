from typing import Union, List

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import lines
from matplotlib.font_manager import FontProperties
from scipy import stats

from statannot.StatResult import StatResult
from statannot.comparisons_corrections.ComparisonsCorrection import ComparisonsCorrection
from statannot.comparisons_corrections.utils import assert_valid_correction_name
from statannot.utils import assert_is_in, remove_null

DEFAULT = object()


def stat_test(
        box_data1,
        box_data2,
        test_name,
        comparisons_correction=None,
        num_comparisons=1,
        verbose=1,
        **stats_params
):
    """Get formatted result of two sample statistical test.

    Arguments
    ---------
    box_data1, box_data2
    test_name: str
        Statistical test to run. Must be one of:
        - `Levene`
        - `Mann-Whitney`
        - `Mann-Whitney-gt`
        - `Mann-Whitney-ls`
        - `t-test_ind`
        - `t-test_welch`
        - `t-test_paired`
        - `Wilcoxon`
        - `Kruskal`
    comparisons_correction: ComparisonCorrection (type 0) or None, default None
        Method to use for multiple comparisons correction.
    num_comparisons: int, default 1
        Number of comparisons to use for multiple comparisons correction.
    verbose: int
        Verbosity parameter, see add_stat_annotation
    stats_params
        Additional keyword arguments to pass to scipy stats functions.

    Returns
    -------
    StatResult object with formatted result of test.

    """
    # Check arguments.
    if comparisons_correction is not None:
        if not isinstance(comparisons_correction, ComparisonsCorrection):
            raise ValueError("comparisons_correction must be a "
                             "ComparisonsCorrection instance")

    # Switch to run scipy.stats hypothesis test.
    if test_name == 'Levene':
        stat, pval = stats.levene(box_data1, box_data2, **stats_params)
        result = StatResult(
            'Levene test of variance', 'levene', 'stat_value', stat, pval
        )
    elif test_name == 'Mann-Whitney':
        u_stat, pval = stats.mannwhitneyu(
            box_data1, box_data2, alternative='two-sided', **stats_params
        )
        result = StatResult(
            'Mann-Whitney-Wilcoxon test two-sided',
            'M.W.W.',
            'U_stat',
            u_stat,
            pval,
        )
    elif test_name == 'Mann-Whitney-gt':
        u_stat, pval = stats.mannwhitneyu(
            box_data1, box_data2, alternative='greater', **stats_params
        )
        result = StatResult(
            'Mann-Whitney-Wilcoxon test greater',
            'M.W.W.',
            'U_stat',
            u_stat,
            pval,
        )
    elif test_name == 'Mann-Whitney-ls':
        u_stat, pval = stats.mannwhitneyu(
            box_data1, box_data2, alternative='less', **stats_params
        )
        result = StatResult(
            'Mann-Whitney-Wilcoxon test smaller',
            'M.W.W.',
            'U_stat',
            u_stat,
            pval,
        )
    elif test_name == 't-test_ind':
        stat, pval = stats.ttest_ind(a=box_data1, b=box_data2, **stats_params)
        result = StatResult('t-test independent samples', 't-test_ind',
                            'stat_value', stat, pval)

    elif test_name == 't-test_welch':
        stat, pval = stats.ttest_ind(
            a=box_data1, b=box_data2, equal_var=False, **stats_params
        )
        result = StatResult(
            'Welch\'s t-test independent samples',
            't-test_welch',
            'stat_value',
            stat,
            pval,
        )
    elif test_name == 't-test_paired':
        stat, pval = stats.ttest_rel(a=box_data1, b=box_data2, **stats_params)
        result = StatResult('t-test paired samples', 't-test_rel',
                            'stat_value', stat, pval)
    elif test_name == 'Wilcoxon':
        zero_method_default = len(box_data1) <= 20 and "pratt" or "wilcox"
        zero_method = stats_params.get('zero_method', zero_method_default)
        if verbose >= 1:
            print("Using zero_method ", zero_method)
        stat, pval = stats.wilcoxon(box_data1, box_data2,
                                    zero_method=zero_method, **stats_params)
        result = StatResult('Wilcoxon test (paired samples)', 'Wilcoxon',
                            'stat_value', stat, pval)
    elif test_name == 'Kruskal':
        stat, pval = stats.kruskal(box_data1, box_data2, **stats_params)
        result = StatResult('Kruskal-Wallis paired samples', 'Kruskal',
                            'stat_value', stat, pval
        )
    else:
        result = StatResult(None, '', None, None, np.nan)

    # Optionally, run multiple comparisons correction
    if comparisons_correction is not None and comparisons_correction.type == 0:
        result.pval = comparisons_correction(result.pval, num_comparisons)
        result.correction_method = comparisons_correction.name

    return result


def pval_annotation_text(result: Union[List[StatResult], StatResult], pvalue_thresholds):
    single_value = False

    if isinstance(result, list):
        x1_pval = np.array([res.pval for res in result])
        x1_corr_signif = [res.corrected_significance for res in result]
    else:
        x1_pval = np.array([result.pval])
        x1_corr_signif = [result.corrected_significance]
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

    x_annot = pd.Series(["ns" if corr_signif is False else star
                         for star, corr_signif in zip(x_annot, x1_corr_signif)])

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


def add_stat_annotation(ax, plot='boxplot', data=None, x=None, y=None,
                        hue=None, units=None, order=None,
                        hue_order=None, box_pairs=None, width=0.8,
                        perform_stat_test=True, pvalues=None,
                        test_short_name=None, test=None, text_format='star',
                        pvalue_format_string=DEFAULT, text_annot_custom=None,
                        loc='inside', show_test_name=True,
                        pvalue_thresholds=DEFAULT, stats_params: dict = None,
                        comparisons_correction='bonferroni',
                        num_comparisons='auto', use_fixed_offset=False,
                        line_offset_to_box=None, line_offset=None,
                        line_height=0.02, text_offset=1, color='0.2',
                        linewidth=1.5, fontsize='medium', verbose=1):
    """
    Optionally computes statistical test between pairs of data series, and add statistical annotation on top
    of the boxes/bars. The same exact arguments `data`, `x`, `y`, `hue`, `order`, `width`,
    `hue_order` (and `units`) as in the seaborn boxplot/barplot function must be passed to this function.

    This function works in one of the two following modes:
    a) `perform_stat_test` is True: statistical test as given by argument `test` is performed.
       The `test_short_name` argument can be used to customize what appears before the pvalues
    b) `perform_stat_test` is False: no statistical test is performed, list of custom p-values `pvalues` are
       used for each pair of boxes. The `test_short_name` argument is then used as the name of the
       custom statistical test.

    :param plot: type of the plot, one of 'boxplot' or 'barplot'.
    :param line_height: in axes fraction coordinates
    :param text_offset: in points
    :param box_pairs: can be of either form: For non-grouped boxplot: `[(cat1, cat2), (cat3, cat4)]`.
        For boxplot grouped by hue: `[((cat1, hue1), (cat2, hue2)), ((cat3, hue3), (cat4, hue4))]`
    :param pvalue_format_string: defaults to `"{.3e}"`
    :param pvalue_thresholds: list of lists, or tuples. Default is:
        For "star" text_format: `[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, "ns"]]`.
        For "simple" text_format : `[[1e-5, "1e-5"], [1e-4, "1e-4"], [1e-3, "0.001"], [1e-2, "0.01"]]`
    :param pvalues: list or array of p-values for each box pair comparison.
    :param stats_params: Parameters for statistical test functions.
    :param comparisons_correction: Method for multiple comparisons correction.
        One of `statsmodel` `multipletests` methods (w/ default FWER), or a ComparisonCorrection instance.
    :param num_comparisons: Override number of comparisons otherwise calculated with number of box_pairs
    """

    def find_x_position_box(b_plotter, boxName):
        """
        boxName can be either a name "cat" or a tuple ("cat", "hue")
        """
        if b_plotter.plot_hues is None:
            cat = boxName
            hue_offset = 0
        else:
            cat = boxName[0]
            hue_level = boxName[1]
            hue_offset = b_plotter.hue_offsets[
                b_plotter.hue_names.index(hue_level)]

        group_pos = b_plotter.group_names.index(cat)
        box_pos = group_pos + hue_offset
        return box_pos

    def get_box_data(b_plotter, boxName):
        """
        boxName can be either a name "cat" or a tuple ("cat", "hue")

        Here we really have to duplicate seaborn code, because there is not
        direct access to the box_data in the BoxPlotter class.
        """
        cat = b_plotter.plot_hues is None and boxName or boxName[0]

        index = b_plotter.group_names.index(cat)
        group_data = b_plotter.plot_data[index]

        if b_plotter.plot_hues is None:
            # Draw a single box or a set of boxes
            # with a single level of grouping
            box_data = remove_null(group_data)
        else:
            hue_level = boxName[1]
            hue_mask = b_plotter.plot_hues[index] == hue_level
            box_data = remove_null(group_data[hue_mask])

        return box_data

    # Set default values if necessary
    if pvalue_format_string is DEFAULT:
        pvalue_format_string = '{:.3e}'
        simple_format_string = '{:.2f}'
    else:
        simple_format_string = pvalue_format_string

    if pvalue_thresholds is DEFAULT:
        if text_format == "star":
            pvalue_thresholds = [[1e-4, "****"], [1e-3, "***"],
                                 [1e-2, "**"], [0.05, "*"], [1, "ns"]]
        else:
            pvalue_thresholds = [[1e-5, "1e-5"], [1e-4, "1e-4"],
                                 [1e-3, "0.001"], [1e-2, "0.01"]]

    if stats_params is None:
        stats_params = dict()

    fig = plt.gcf()

    # Validate arguments
    if perform_stat_test:
        if test is None:
            raise ValueError("If `perform_stat_test` is True, `test` must be specified.")
        if pvalues is not None or test_short_name is not None:
            raise ValueError("If `perform_stat_test` is True, custom `pvalues` "
                             "or `test_short_name` must be `None`.")
        valid_list = ['t-test_ind', 't-test_welch', 't-test_paired',
                      'Mann-Whitney', 'Mann-Whitney-gt', 'Mann-Whitney-ls',
                      'Levene', 'Wilcoxon', 'Kruskal']
        if test not in valid_list:
            raise ValueError("test value should be one of the following: {}."
                             .format(', '.join(valid_list)))
    else:
        if pvalues is None:
            raise ValueError("If `perform_stat_test` is False, custom `pvalues` must be specified.")
        if test is not None:
            raise ValueError("If `perform_stat_test` is False, `test` must be None.")
        if len(pvalues) != len(box_pairs):
            raise ValueError("`pvalues` should be of the same length as `box_pairs`.")

    if text_annot_custom is not None and len(text_annot_custom) != len(box_pairs):
        raise ValueError("`text_annot_custom` should be of same length as `box_pairs`.")

    assert_is_in(
        loc, ['inside', 'outside'], label='argument `loc`'
    )
    assert_is_in(
        text_format,
        ['full', 'simple', 'star'],
        label='argument `text_format`'
    )

    # Comparisons correction
    if comparisons_correction is None:
        pass
    elif isinstance(comparisons_correction, str):
        assert_valid_correction_name(comparisons_correction)
        comparisons_correction = ComparisonsCorrection(comparisons_correction)
    elif not(isinstance(comparisons_correction, ComparisonsCorrection)):
            raise ValueError("comparisons_correction must be a statmodels "
                             "method name or a ComparisonCorrection instance")

    if verbose >= 1 and text_format == 'star':
        print("p-value annotation legend:")
        pvalue_thresholds = pd.DataFrame(pvalue_thresholds).sort_values(by=0, ascending=False).values
        for i in range(0, len(pvalue_thresholds)):
            if i < len(pvalue_thresholds) - 1:
                print('{}: {:.2e} < p <= {:.2e}'.format(pvalue_thresholds[i][1],
                                                        pvalue_thresholds[i + 1][0],
                                                        pvalue_thresholds[i][0]))
            else:
                print('{}: p <= {:.2e}'.format(pvalue_thresholds[i][1], pvalue_thresholds[i][0]))
        print()

    ylim = ax.get_ylim()
    yrange = ylim[1] - ylim[0]

    if line_offset is None:
        if loc == 'inside':
            line_offset = 0.05
            if line_offset_to_box is None:
                line_offset_to_box = 0.06
        # 'outside', see valid_list
        else:
            line_offset = 0.03
            if line_offset_to_box is None:
                line_offset_to_box = line_offset
    else:
        if loc == 'inside':
            if line_offset_to_box is None:
                line_offset_to_box = 0.06
        elif loc == 'outside':
            line_offset_to_box = line_offset
    y_offset = line_offset * yrange
    y_offset_to_box = line_offset_to_box * yrange

    if plot == 'boxplot':
        # Create the same plotter object as seaborn's boxplot
        box_plotter = sns.categorical._BoxPlotter(
            x, y, hue, data, order, hue_order, orient=None, width=width, color=None,
            palette=None, saturation=.75, dodge=True, fliersize=5, linewidth=None)

    elif plot == 'barplot':
        # Create the same plotter object as seaborn's barplot
        box_plotter = sns.categorical._BarPlotter(
            x, y, hue, data, order, hue_order,
            estimator=np.mean, ci=95, n_boot=1000, units=None,
            orient=None, color=None, palette=None, saturation=.75, seed=None,
            errcolor=".26", errwidth=None, capsize=None, dodge=True)

    else:
        raise NotImplementedError("Only boxplots and barplots are supported.")

    # Build the list of box data structures with the x and ymax positions
    group_names = box_plotter.group_names
    hue_names = box_plotter.hue_names
    if box_plotter.plot_hues is None:
        box_names = group_names
        labels = box_names
    else:
        box_names = [(group_name, hue_name) for group_name in group_names for hue_name in hue_names]
        labels = ['{}_{}'.format(group_name, hue_name) for (group_name, hue_name) in box_names]

    box_structs = [
        {
            'box':      box_names[i],
            'label':    labels[i],
            'x':        find_x_position_box(box_plotter, box_names[i]),
            'box_data': get_box_data(box_plotter, box_names[i]),
            'ymax':     (np.amax(get_box_data(box_plotter, box_names[i]))
                         if len(get_box_data(box_plotter, box_names[i])) > 0
                         else np.nan)
        } for i in range(len(box_names))]

    # Sort the box data structures by position along the x axis
    box_structs = sorted(box_structs, key=lambda a: a['x'])
    # Add the index position in the list of boxes along the x axis
    box_structs = [dict(box_struct, xi=i) for i, box_struct in enumerate(box_structs)]
    # Same data structure list with access key by box name
    box_structs_dic = {box_struct['box']: box_struct for box_struct in box_structs}

    # Build the list of box data structure pairs
    box_struct_pairs = []
    for i_box_pair, (box1, box2) in enumerate(box_pairs):
        valid = box1 in box_names and box2 in box_names
        if not valid:
            raise ValueError("box_pairs contains an invalid box pair.")
            pass
        # i_box_pair will keep track of the original order of the box pairs.
        box_struct1 = dict(box_structs_dic[box1], i_box_pair=i_box_pair)
        box_struct2 = dict(box_structs_dic[box2], i_box_pair=i_box_pair)
        if box_struct1['x'] <= box_struct2['x']:
            pair = (box_struct1, box_struct2)
        else:
            pair = (box_struct2, box_struct1)
        box_struct_pairs.append(pair)

    # Draw first the annotations with the shortest between-boxes distance, in order to reduce
    # overlapping between annotations.
    box_struct_pairs = sorted(box_struct_pairs, key=lambda a: abs(a[1]['x'] - a[0]['x']))

    # Build array that contains the x and y_max position of the highest annotation or box data at
    # a given x position, and also keeps track of the number of stacked annotations.
    # This array will be updated when a new annotation is drawn.
    y_stack_arr = np.array([[box_struct['x'] for box_struct in box_structs],
                            [box_struct['ymax'] for box_struct in box_structs],
                            [0 for _ in range(len(box_structs))]])
    if loc == 'outside':
        y_stack_arr[1, :] = ylim[1]
    ann_list = []
    test_result_list = []
    ymaxs = []
    y_stack = []

    # fetch results of all tests
    for box_struct1, box_struct2 in box_struct_pairs:
        box1 = box_struct1['box']
        box2 = box_struct2['box']
        box_data1 = box_struct1['box_data']
        box_data2 = box_struct2['box_data']
        i_box_pair = box_struct1['i_box_pair']

        if perform_stat_test:
            result = stat_test(
                box_data1,
                box_data2,
                test,
                comparisons_correction,
                num_comparisons != "auto" and num_comparisons or len(box_struct_pairs),
                verbose,
                **stats_params
            )
        else:
            test_short_name = test_short_name if test_short_name is not None else ''
            result = StatResult(
                'Custom statistical test',
                test_short_name,
                None,
                None,
                pvalues[i_box_pair]
            )
        result.box1 = box1
        result.box2 = box2

        test_result_list.append(result)

    # Perform other types of correction methods for multiple testing
    if comparisons_correction is not None:
        corr_name = comparisons_correction.name

        # If correction is applied to a set of pvalues
        if comparisons_correction.type == 1:
            original_pvalues = [result.pval for result in test_result_list]

            significant_pvalues = comparisons_correction(original_pvalues)

            for is_significant, result in zip(significant_pvalues, test_result_list):
                result.correction_method = corr_name
                result.corrected_significance = is_significant

        # If correction is applied per pvalue, just compare with alpha
        else:
            alpha = comparisons_correction.alpha
            for result in test_result_list:
                result.correction_method = corr_name
                result.corrected_significance = result.pval < alpha

    # Then annotate
    for box_structs, result in zip(box_struct_pairs, test_result_list):
        x1 = box_structs[0]['x']
        x2 = box_structs[1]['x']
        xi1 = box_structs[0]['xi']
        xi2 = box_structs[1]['xi']
        label1 = box_structs[0]['label']
        label2 = box_structs[1]['label']
        # ymax1 = box_structs[0]['ymax']  # Not used, so do not assign them
        # ymax2 = box_structs[1]['ymax']
        i_box_pair = box_structs[0]['i_box_pair']
        if verbose >= 1:
            print("{} v.s. {}: {}".format(label1, label2, result.formatted_output))

        if text_annot_custom is not None:
            text = text_annot_custom[i_box_pair]
        else:
            if text_format == 'full':
                text = "{} p = {}{}".format('{}', pvalue_format_string, '{}').format(
                    result.test_short_name, result.pval, result.significance_suffix)
            elif text_format is 'star':
                text = pval_annotation_text(result, pvalue_thresholds)
            elif text_format is 'simple':
                test_short_name = result.test_short_name if show_test_name else None
                text = simple_text(result, simple_format_string,
                                   pvalue_thresholds, test_short_name)
            else:  # None:
                text = None

        # Find y maximum for all the y_stacks *in between* the box1 and the box2
        i_ymax_in_range_x1_x2 = xi1 + np.nanargmax(
            y_stack_arr[1, np.where((x1 <= y_stack_arr[0, :])
                                    & (y_stack_arr[0, :] <= x2))])
        ymax_in_range_x1_x2 = y_stack_arr[1, i_ymax_in_range_x1_x2]

        yref = ymax_in_range_x1_x2
        yref2 = yref

        # Choose the best offset depending on whether there is an annotation below
        # at the x position in the range [x1, x2] where the stack is the highest
        if y_stack_arr[2, i_ymax_in_range_x1_x2] == 0:
            # there is only a box below
            offset = y_offset_to_box
        else:
            # there is an annotation below
            offset = y_offset
        y = yref2 + offset
        h = line_height * yrange
        line_x, line_y = [x1, x1, x2, x2], [y, y + h, y + h, y]
        if loc == 'inside':
            ax.plot(line_x, line_y, lw=linewidth, c=color)
        elif loc == 'outside':
            line = lines.Line2D(line_x, line_y, lw=linewidth, c=color, transform=ax.transData)
            line.set_clip_on(False)
            ax.add_line(line)

        # why should we change here the ylim if at the very end we set it to the correct range????
        # ax.set_ylim((ylim[0], 1.1*(y + h)))

        if text is not None:
            ann = ax.annotate(
                text, xy=(np.mean([x1, x2]), y + h),
                xytext=(0, text_offset), textcoords='offset points',
                xycoords='data', ha='center', va='bottom',
                fontsize=fontsize, clip_on=False, annotation_clip=False)
            ann_list.append(ann)

            plt.draw()
            y_top_annot = None
            got_mpl_error = False
            if not use_fixed_offset:
                try:
                    bbox = ann.get_window_extent()
                    bbox_data = bbox.transformed(ax.transData.inverted())
                    y_top_annot = bbox_data.ymax
                except RuntimeError:
                    got_mpl_error = True

            if use_fixed_offset or got_mpl_error:
                if verbose >= 1:
                    print("Warning: cannot get the text bounding box. Falling back to a fixed"
                          " y offset. Layout may be not optimal.")
                # We will apply a fixed offset in points,
                # based on the font size of the annotation.
                fontsize_points = FontProperties(size='medium').get_size_in_points()
                offset_trans = mtransforms.offset_copy(
                    ax.transData, fig=fig, x=0,
                    y=1.0 * fontsize_points + text_offset, units='points')
                y_top_display = offset_trans.transform((0, y + h))
                y_top_annot = ax.transData.inverted().transform(y_top_display)[1]
        else:
            y_top_annot = y + h

        y_stack.append(y_top_annot)  # remark: y_stack is not really necessary if we have the stack_array
        ymaxs.append(max(y_stack))
        # Fill the highest y position of the annotation into the y_stack array
        # for all positions in the range x1 to x2
        y_stack_arr[1, (x1 <= y_stack_arr[0, :]) & (y_stack_arr[0, :] <= x2)] = y_top_annot
        # Increment the counter of annotations in the y_stack array
        y_stack_arr[2, xi1:xi2 + 1] = y_stack_arr[2, xi1:xi2 + 1] + 1

    y_stack_max = max(ymaxs)
    if loc == 'inside':
        ax.set_ylim((ylim[0], max(1.03 * y_stack_max, ylim[1])))
    elif loc == 'outside':
        ax.set_ylim((ylim[0], ylim[1]))

    return ax, test_result_list
