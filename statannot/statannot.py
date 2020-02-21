import warnings

import matplotlib.pyplot as plt
from matplotlib import lines
import matplotlib.transforms as mtransforms
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn.utils import remove_na

from .utils import raise_expected_got, assert_is_in
from .StatResult import StatResult

from scipy import stats

DEFAULT = object()


def stat_test(
    box_data1,
    box_data2,
    test,
    comparisons_correction=None,
    num_comparisons=1,
    **stats_params
):
    """Get formatted result of two sample statistical test.

    Arguments
    ---------
    bbox_data1, bbox_data2
    test: str
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
    comparisons_correction: str or None, default None
        Method to use for multiple comparisons correction. Currently only the
        Bonferroni correction is implemented.
    num_comparisons: int, default 1
        Number of comparisons to use for multiple comparisons correction.
    stats_params
        Additional keyword arguments to pass to scipy stats functions.

    Returns
    -------
    StatResult object with formatted result of test.

    """
    # Check arguments.
    assert_is_in(
        comparisons_correction,
        ['bonferroni', None],
        label='argument `comparisons_correction`',
    )

    # Switch to run scipy.stats hypothesis test.
    if test == 'Levene':
        stat, pval = stats.levene(box_data1, box_data2, **stats_params)
        result = StatResult(
            'Levene test of variance', 'levene', 'stat', stat, pval
        )
    elif test == 'Mann-Whitney':
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
    elif test == 'Mann-Whitney-gt':
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
    elif test == 'Mann-Whitney-ls':
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
    elif test == 't-test_ind':
        stat, pval = stats.ttest_ind(a=box_data1, b=box_data2, **stats_params)
        result = StatResult(
            't-test independent samples', 't-test_ind', 'stat', stat, pval
        )
    elif test == 't-test_welch':
        stat, pval = stats.ttest_ind(
            a=box_data1, b=box_data2, equal_var=False, **stats_params
        )
        result = StatResult(
            'Welch\'s t-test independent samples',
            't-test_welch',
            'stat',
            stat,
            pval,
        )
    elif test == 't-test_paired':
        stat, pval = stats.ttest_rel(a=box_data1, b=box_data2, **stats_params)
        result = StatResult(
            't-test paired samples', 't-test_rel', 'stat', stat, pval
        )
    elif test == 'Wilcoxon':
        zero_method_default = len(box_data1) <= 20 and "pratt" or "wilcox"
        zero_method = stats_params.get('zero_method', zero_method_default)
        print("Using zero_method ", zero_method)
        stat, pval = stats.wilcoxon(
            box_data1, box_data2, zero_method=zero_method, **stats_params
        )
        result = StatResult(
            'Wilcoxon test (paired samples)', 'Wilcoxon', 'stat', stat, pval
        )
    elif test == 'Kruskal':
        stat, pval = stats.kruskal(box_data1, box_data2, **stats_params)
        test_short_name = 'Kruskal'
        result = StatResult(
            'Kruskal-Wallis paired samples', 'Kruskal', 'stat', stat, pval
        )
    else:
        result = StatResult(None, '', None, None, np.nan)

    # Optionally, run multiple comparisons correction.
    if comparisons_correction == 'bonferroni':
        result.pval = bonferroni(result.pval, num_comparisons)
        result.test_str = result.test_str + ' with Bonferroni correction'
    elif comparisons_correction is None:
        pass
    else:
        # This should never be reached because `comparisons_correction` must
        # be a valid correction method or None.
        raise RuntimeError('Unexpectedly reached end of switch.')

    return result


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
    if np.ndim(p_values) > 1:
        raise_expected_got(
            'Scalar or list-like', 'argument `p_values`', p_values
        )
    if num_comparisons != 'auto':
        try:
            # Raise a TypeError if num_comparisons is not numeric, and raise
            # an AssertionError if it isn't int-like.
            assert np.ceil(num_comparisons) == num_comparisons
        except (AssertionError, TypeError) as e:
            raise_expected_got(
                'Int or `auto`', 'argument `num_comparisons`', num_comparisons
            )

    # Coerce p_values to numpy array.
    p_values_array = np.atleast_1d(p_values)

    if num_comparisons == 'auto':
        # Infer number of comparisons
        num_comparisons = len(p_values_array)
    elif len(p_values_array) > 1 and num_comparisons != len(p_values_array):
        # Warn if multiple p_values have been passed and num_comparisons is
        # set manually.
        warnings.warn(
            'Manually-specified `num_comparisons={}` differs from number of '
            'p_values to correct ({}).'.format(
                num_comparisons, len(p_values_array)
            )
        )

    # Apply correction by multiplying p_values and thresholding at p=1.0
    p_values_array *= num_comparisons
    p_values_array = np.min(
        [p_values_array, np.ones_like(p_values_array)], axis=0
    )

    if len(p_values_array) == 1:
        # Return a scalar if input was a scalar.
        return p_values_array[0]
    else:
        return p_values_array



def pval_annotation_text(x, pvalue_thresholds):
    single_value = False
    if type(x) is np.array:
        x1 = x
    else:
        x1 = np.array([x])
        single_value = True
    # Sort the threshold array
    pvalue_thresholds = pd.DataFrame(pvalue_thresholds).sort_values(by=0, ascending=False).values
    x_annot = pd.Series(["" for _ in range(len(x1))])
    for i in range(0, len(pvalue_thresholds)):
        if i < len(pvalue_thresholds)-1:
            condition = (x1 <= pvalue_thresholds[i][0]) & (pvalue_thresholds[i+1][0] < x1)
            x_annot[condition] = pvalue_thresholds[i][1]
        else:
            condition = x1 < pvalue_thresholds[i][0]
            x_annot[condition] = pvalue_thresholds[i][1]

    return x_annot if not single_value else x_annot.iloc[0]


def simple_text(pval, pvalue_format, pvalue_thresholds, test_short_name=None):
    """
    Generates simple text for test name and pvalue
    :param pval: pvalue
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
        if pval < threshold[0]:
            pval_text = "p ≤ {}".format(threshold[1])
            break
    else:
        pval_text = "p = {}".format(pvalue_format).format(pval)

    return text + pval_text


def add_stat_annotation(ax, plot='boxplot',
                        data=None, x=None, y=None, hue=None, units=None, order=None,
                        hue_order=None, box_pairs=None, width=0.8,
                        perform_stat_test=True,
                        pvalues=None, test_short_name=None,
                        test=None, text_format='star', pvalue_format_string=DEFAULT,
                        text_annot_custom=None,
                        loc='inside', show_test_name=True,
                        pvalue_thresholds=DEFAULT, stats_params=dict(),
                        comparisons_correction='bonferroni',
                        use_fixed_offset=False, line_offset_to_box=None,
                        line_offset=None, line_height=0.02, text_offset=1,
                        color='0.2', linewidth=1.5,
                        fontsize='medium', verbose=1):
    """
    Optionally computes statistical test between pairs of data series, and add statistical annotation on top
    of the boxes/bars. The same exact arguments `data`, `x`, `y`, `hue`, `order`, `width`,
    `hue_order` (and `units`) as in the seaborn boxplot/barplot function must be passed to this function.

    This function works in one of the two following modes:
    a) `perform_stat_test` is True: statistical test as given by argument `test` is performed.
    b) `perform_stat_test` is False: no statistical test is performed, list of custom p-values `pvalues` are
       used for each pair of boxes. The `test_short_name` argument is then used as the name of the
       custom statistical test.

    :param plot: type of the plot, one of 'boxplot' or 'barplot'.
    :param line_height: in axes fraction coordinates
    :param text_offset: in points
    :param box_pairs: can be of either form: For non-grouped boxplot: `[(cat1, cat2), (cat3, cat4)]`. For boxplot grouped by hue: `[((cat1, hue1), (cat2, hue2)), ((cat3, hue3), (cat4, hue4))]`
    :param pvalue_format_string: defaults to `"{.3e}"`
    :param pvalue_thresholds: list of lists, or tuples. Default is: For "star" text_format: `[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, "ns"]]`. For "simple" text_format : `[[1e-5, "1e-5"], [1e-4, "1e-4"], [1e-3, "0.001"], [1e-2, "0.01"]]`
    :param pvalues: list or array of p-values for each box pair comparison.
    :param comparisons_correction: Method for multiple comparisons correction. `bonferroni` or None.
    """

    def find_x_position_box(box_plotter, boxName):
        """
        boxName can be either a name "cat" or a tuple ("cat", "hue")
        """
        if box_plotter.plot_hues is None:
            cat = boxName
            hue_offset = 0
        else:
            cat = boxName[0]
            hue = boxName[1]
            hue_offset = box_plotter.hue_offsets[
                box_plotter.hue_names.index(hue)]

        group_pos = box_plotter.group_names.index(cat)
        box_pos = group_pos + hue_offset
        return box_pos

    def get_box_data(box_plotter, boxName):
        """
        boxName can be either a name "cat" or a tuple ("cat", "hue")

        Here we really have to duplicate seaborn code, because there is not
        direct access to the box_data in the BoxPlotter class.
        """
        cat = box_plotter.plot_hues is None and boxName or boxName[0]

        index = box_plotter.group_names.index(cat)
        group_data = box_plotter.plot_data[index]

        if box_plotter.plot_hues is None:
            # Draw a single box or a set of boxes
            # with a single level of grouping
            box_data = remove_na(group_data)
        else:
            hue_level = boxName[1]
            hue_mask = box_plotter.plot_hues[index] == hue_level
            box_data = remove_na(group_data[hue_mask])

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
    assert_is_in(
        comparisons_correction,
        ['bonferroni', None],
        label='argument `comparisons_correction`'
    )

    if verbose >= 1 and text_format == 'star':
        print("p-value annotation legend:")
        pvalue_thresholds = pd.DataFrame(pvalue_thresholds).sort_values(by=0, ascending=False).values
        for i in range(0, len(pvalue_thresholds)):
            if i < len(pvalue_thresholds)-1:
                print('{}: {:.2e} < p <= {:.2e}'.format(pvalue_thresholds[i][1],
                                                        pvalue_thresholds[i+1][0],
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
    y_offset = line_offset*yrange
    y_offset_to_box = line_offset_to_box*yrange

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
            orient=None, color=None, palette=None, saturation=.75,
            errcolor=".26", errwidth=None, capsize=None, dodge=True)

    # Build the list of box data structures with the x and ymax positions
    group_names = box_plotter.group_names
    hue_names = box_plotter.hue_names
    if box_plotter.plot_hues is None:
        box_names = group_names
        labels = box_names
    else:
        box_names = [(group_name, hue_name) for group_name in group_names for hue_name in hue_names]
        labels = ['{}_{}'.format(group_name, hue_name) for (group_name, hue_name) in box_names]

    box_structs = [{'box':box_names[i],
                    'label':labels[i],
                    'x':find_x_position_box(box_plotter, box_names[i]),
                    'box_data':get_box_data(box_plotter, box_names[i]),
                    'ymax':np.amax(get_box_data(box_plotter, box_names[i])) if
                           len(get_box_data(box_plotter, box_names[i])) > 0 else np.nan}
                   for i in range(len(box_names))]
    # Sort the box data structures by position along the x axis
    box_structs = sorted(box_structs, key=lambda x: x['x'])
    # Add the index position in the list of boxes along the x axis
    box_structs = [dict(box_struct, xi=i) for i, box_struct in enumerate(box_structs)]
    # Same data structure list with access key by box name
    box_structs_dic = {box_struct['box']:box_struct for box_struct in box_structs}

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
    box_struct_pairs = sorted(box_struct_pairs, key=lambda x: abs(x[1]['x'] - x[0]['x']))

    # Build array that contains the x and y_max position of the highest annotation or box data at
    # a given x position, and also keeps track of the number of stacked annotations.
    # This array will be updated when a new annotation is drawn.
    y_stack_arr = np.array([[box_struct['x'] for box_struct in box_structs],
                            [box_struct['ymax'] for box_struct in box_structs],
                            [0 for i in range(len(box_structs))]])
    if loc == 'outside':
        y_stack_arr[1, :] = ylim[1]
    ann_list = []
    test_result_list = []
    ymaxs = []
    y_stack = []

    for box_struct1, box_struct2 in box_struct_pairs:

        box1 = box_struct1['box']
        box2 = box_struct2['box']
        label1 = box_struct1['label']
        label2 = box_struct2['label']
        box_data1 = box_struct1['box_data']
        box_data2 = box_struct2['box_data']
        x1 = box_struct1['x']
        x2 = box_struct2['x']
        xi1 = box_struct1['xi']
        xi2 = box_struct2['xi']
        ymax1 = box_struct1['ymax']
        ymax2 = box_struct2['ymax']
        i_box_pair = box_struct1['i_box_pair']

        # Find y maximum for all the y_stacks *in between* the box1 and the box2
        i_ymax_in_range_x1_x2 = xi1 + np.nanargmax(y_stack_arr[1, np.where((x1 <= y_stack_arr[0, :]) &
                                                                           (y_stack_arr[0, :] <= x2))])
        ymax_in_range_x1_x2 = y_stack_arr[1, i_ymax_in_range_x1_x2]

        if perform_stat_test:
            result = stat_test(
                box_data1,
                box_data2,
                test,
                comparisons_correction,
                len(box_struct_pairs),
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

        if verbose >= 1:
            print("{} v.s. {}: {}".format(label1, label2, result.formatted_output))

        if text_annot_custom is not None:
            text = text_annot_custom[i_box_pair]
        else:
            if text_format == 'full':
                text = "{} p = {}".format('{}', pvalue_format_string).format(result.test_short_name, result.pval)
            elif text_format is None:
                text = None
            elif text_format is 'star':
                text = pval_annotation_text(result.pval, pvalue_thresholds)
            elif text_format is 'simple':
                test_short_name = show_test_name and test_short_name or ""
                text = simple_text(result.pval, simple_format_string, pvalue_thresholds, test_short_name)

        yref = ymax_in_range_x1_x2
        yref2 = yref

        # Choose the best offset depending on wether there is an annotation below
        # at the x position in the range [x1, x2] where the stack is the highest
        if y_stack_arr[2, i_ymax_in_range_x1_x2] == 0:
            # there is only a box below
            offset = y_offset_to_box
        else:
            # there is an annotation below
            offset = y_offset
        y = yref2 + offset
        h = line_height*yrange
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
                    y=1.0*fontsize_points + text_offset, units='points')
                y_top_display = offset_trans.transform((0, y + h))
                y_top_annot = ax.transData.inverted().transform(y_top_display)[1]
        else:
            y_top_annot = y + h

        y_stack.append(y_top_annot)    # remark: y_stack is not really necessary if we have the stack_array
        ymaxs.append(max(y_stack))
        # Fill the highest y position of the annotation into the y_stack array
        # for all positions in the range x1 to x2
        y_stack_arr[1, (x1 <= y_stack_arr[0, :]) & (y_stack_arr[0, :] <= x2)] = y_top_annot
        # Increment the counter of annotations in the y_stack array
        y_stack_arr[2, xi1:xi2 + 1] = y_stack_arr[2, xi1:xi2 + 1] + 1

    y_stack_max = max(ymaxs)
    if loc == 'inside':
        ax.set_ylim((ylim[0], max(1.03*y_stack_max, ylim[1])))
    elif loc == 'outside':
        ax.set_ylim((ylim[0], ylim[1]))

    return ax, test_result_list
