import warnings

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import seaborn as sns
from matplotlib import lines
from matplotlib.collections import PathCollection
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

from statannotations.format_annotations import pval_annotation_text, simple_text
from statannotations.stats.ComparisonsCorrection import ComparisonsCorrection
from statannotations.stats.StatResult import StatResult
from statannotations.stats.StatTest import StatTest
from statannotations.stats.tests import stat_test, IMPLEMENTED_TESTS
from statannotations.stats.utils import check_valid_correction_name
from statannotations.utils import check_is_in, remove_null, \
    check_order_box_pairs_in_data, check_not_none, check_valid_text_format

DEFAULT = object()
# temp: only for annotate : line_height, use_fixed_offset,
# fontsize, color

CONFIGURABLE_PARAMETERS = [
    'comparisons_correction',
    'line_offset',
    'line_offset_to_box',
    'loc',
    'pvalue_format_string',
    'pvalue_thresholds',
    'test',
    'test_short_name',
    'text_format',
    'verbose'
]


def _get_box_x_position(b_plotter, box_name):
    """
    box_name can be either a name "cat" or a tuple ("cat", "hue")
    """
    if b_plotter.plot_hues is None:
        cat = box_name
        hue_offset = 0
    else:
        cat = box_name[0]
        hue_level = box_name[1]
        hue_offset = b_plotter.hue_offsets[
            b_plotter.hue_names.index(hue_level)]

    group_pos = b_plotter.group_names.index(cat)
    box_pos = group_pos + hue_offset
    return box_pos


def _get_xpos_location(pos, xranges):
    """
    Finds the x-axis location of a categorical variable
    """
    for xrange in xranges:
        if (pos >= xrange[0]) & (pos <= xrange[1]):
            return xrange[2]


def _generate_ymaxes(ax, fig, box_plotter, box_names):
    """
    given box plotter and the names of two categorical variables,
    returns highest y point drawn between those two variables before
    annotations.

    The highest data point is often not the highest item drawn
    (eg, error bars and/or bar charts).
    """
    xpositions = {
        np.round(_get_box_x_position(box_plotter, box_name), 1): box_name
        for box_name in box_names}
    ymaxes = {name: 0 for name in box_names}
    data_to_ax = _get_transform_func(ax, 'data_to_ax')
    for child in ax.get_children():
        if ((type(child) == PathCollection)
                and len(child.properties()['offsets'])):

            ymax = child.properties()['offsets'][:, 1].max()
            xpos = float(np.round(np.nanmean(
                child.properties()['offsets'][:, 0]), 1))

            xname = xpositions[xpos]
            ypos = data_to_ax.transform((0, ymax))[1]
            if ypos > ymaxes[xname]:
                ymaxes[xname] = ypos

        elif (type(child) == lines.Line2D) or (type(child) == Rectangle):
            xunits = (max(list(xpositions.keys())) + 1) / len(xpositions)
            xranges = {(pos - xunits / 2, pos + xunits / 2, pos): box_name
                       for pos, box_name in xpositions.items()}
            box = ax.transData.inverted().transform(
                child.get_window_extent(fig.canvas.get_renderer()))

            if (box[:, 0].max() - box[:, 0].min()) > 1.1 * xunits:
                continue
            raw_xpos = np.round(box[:, 0].mean(), 1)
            xpos = _get_xpos_location(raw_xpos, xranges)
            if xpos not in xpositions:
                continue
            xname = xpositions[xpos]
            ypos = box[:, 1].max()
            ypos = data_to_ax.transform((0, ypos))[1]
            if ypos > ymaxes[xname]:
                ymaxes[xname] = ypos

    return ymaxes


def _get_group_data(b_plotter, cat):
    index = b_plotter.group_names.index(cat)

    return b_plotter.plot_data[index], index


def _get_box_data_with_hue(b_plotter, box_name):
    cat = box_name[0]
    group_data, index = _get_group_data(b_plotter, cat)

    hue_level = box_name[1]
    hue_mask = b_plotter.plot_hues[index] == hue_level

    return group_data[hue_mask]


def _get_box_data_without_hue(b_plotter, box_name):
    cat = box_name
    group_data, _ = _get_group_data(b_plotter, cat)

    return group_data


def _get_box_data(b_plotter, box_name):
    """
    box_name can be either a name "cat" or a tuple ("cat", "hue")

    Almost a duplicate of seaborn's code, because there is not
    direct access to the box_data in the BoxPlotter class.
    """
    if b_plotter.plot_hues is None:
        data = _get_box_data_without_hue(b_plotter, box_name)
    else:
        data = _get_box_data_with_hue(b_plotter, box_name)

    box_data = remove_null(data)

    return box_data


def _get_transform_func(ax, kind):
    """
    Given an axis object, returns one of three possible transformation
    functions to move between coordinate systems, depending on the value of
    kind:
    'data_to_ax': converts data coordinates to axes coordinates
    'ax_to_data': converts axes coordinates to data coordinates
    'pix_to_ax': converts pixel coordinates to axes coordinates
    'all': return tuple of all three

    This function should be called whenever axes limits are altered.
    """

    check_is_in(kind, ['data_to_ax', 'ax_to_data', 'pix_to_ax', 'all'], 'kind')

    if kind == 'pix_to_ax':
        return ax.transAxes.inverted()

    data_to_ax = ax.transData + ax.get_xaxis_transform().inverted()

    if kind == 'data_to_ax':
        return data_to_ax

    elif kind == 'ax_to_data':
        return data_to_ax.inverted()

    else:
        return data_to_ax, data_to_ax.inverted(), ax.transAxes.inverted()


def _check_test_pvalues_perform(test, test_short_name):
    if test is None:
        raise ValueError("If `perform_stat_test` is True, `test` must be specified.")
    if test_short_name is not None:
        raise ValueError("If `perform_stat_test` is True"
                         "and `test_short_name` must be `None`.")

    if test not in IMPLEMENTED_TESTS and not isinstance(test, StatTest):
        raise ValueError("test value should be a StatTest instance "
                         "or one of the following strings: {}."
                         .format(', '.join(IMPLEMENTED_TESTS)))


def _check_test_pvalues_no_perform(test, pvalues, box_pairs):
    if pvalues is None:
        raise ValueError("If `perform_stat_test` is False, custom `pvalues` must be specified.")
    if test is not None:
        raise ValueError("If `perform_stat_test` is False, `test` must be None.")
    if len(pvalues) != len(box_pairs):
        raise ValueError("`pvalues` should be of the same length as `box_pairs`.")


def get_validated_comparisons_correction(comparisons_correction):
    if comparisons_correction is None:
        return

    if isinstance(comparisons_correction, str):
        check_valid_correction_name(comparisons_correction)
        try:
            comparisons_correction = ComparisonsCorrection(
                comparisons_correction)
        except ImportError as e:
            raise ImportError(f"{e} Please install statsmodels or pass "
                              f"`comparisons_correction=None`.")

        except NotImplementedError as e:
            raise NotImplementedError(e)

        except ValueError as e:
            raise NotImplementedError(e)

    if not (isinstance(comparisons_correction, ComparisonsCorrection)):
        raise ValueError("comparisons_correction must be a statmodels "
                         "method name or a ComparisonCorrection instance")

    return comparisons_correction


def sort_pvalue_thresholds(pvalue_thresholds):
    return sorted(pvalue_thresholds,
                  key=lambda threshold_notation: threshold_notation[0])


def check_correct_number_custom_annotations(text_annot_custom, box_struct_pairs):
    if text_annot_custom is None:
        return

    if len(text_annot_custom) != len(box_struct_pairs):
        raise ValueError(
            "`text_annot_custom` should be of same length as `box_pairs`.")


def _apply_type1_comparisons_correction(comparisons_correction, test_result_list):

    original_pvalues = [result.pval for result in test_result_list]

    significant_pvalues = comparisons_correction(original_pvalues)

    for is_significant, result in zip(significant_pvalues,
                                      test_result_list):
        result.correction_method = comparisons_correction.name
        result.corrected_significance = is_significant


def _apply_type0_comparisons_correction(comparisons_correction, test_result_list):
    alpha = comparisons_correction.alpha
    for result in test_result_list:
        result.correction_method = comparisons_correction.name
        result.corrected_significance = (
                result.pval < alpha
                or np.isclose(result.pval, alpha))


def _apply_comparisons_correction(comparisons_correction, test_result_list):
    if comparisons_correction is None:
        return

    if comparisons_correction.type == 1:
        _apply_type1_comparisons_correction(comparisons_correction,
                                            test_result_list)

    else:
        _apply_type0_comparisons_correction(comparisons_correction,
                                            test_result_list)


def _get_pvalue_and_simple_formats(pvalue_format_string):
    if pvalue_format_string is DEFAULT:
        pvalue_format_string = '{:.3e}'
        simple_format_string = '{:.2f}'
    else:
        simple_format_string = pvalue_format_string

    return pvalue_format_string, simple_format_string


def _get_stat_result_from_test(box_struct1, box_struct2, test,
                               comparisons_correction, num_comparisons,
                               pvalue_thresholds, **stats_params) -> StatResult:

    box_data1 = box_struct1['box_data']
    box_data2 = box_struct2['box_data']

    result = stat_test(
        box_data1,
        box_data2,
        test,
        comparisons_correction=comparisons_correction,
        num_comparisons=num_comparisons,
        alpha=pvalue_thresholds[-2][0],
        **stats_params
    )

    return result


def _get_custom_results(box_struct1, test_short_name, pvalue_thresholds,
                        pvalues) -> StatResult:

    i_box_pair = box_struct1['i_box_pair']

    result = StatResult(
        'Custom statistical test',
        test_short_name,
        None,
        None,
        pval=pvalues[i_box_pair],
        alpha=pvalue_thresholds[-2][0]
    )

    return result


def _get_box_struct_pairs(box_pairs, box_structs, box_names):
    box_structs_dic = {box_struct['box']: box_struct
                       for box_struct in box_structs}

    box_struct_pairs = []
    for i_box_pair, (box1, box2) in enumerate(box_pairs):
        valid = box1 in box_names and box2 in box_names
        if not valid:
            raise ValueError("box_pairs contains an invalid box pair.")

        # i_box_pair will keep track of the original order of the box pairs.
        box_struct1 = dict(box_structs_dic[box1], i_box_pair=i_box_pair)
        box_struct2 = dict(box_structs_dic[box2], i_box_pair=i_box_pair)
        if box_struct1['x'] <= box_struct2['x']:
            pair = (box_struct1, box_struct2)
        else:
            pair = (box_struct2, box_struct1)
        box_struct_pairs.append(pair)

    # Draw first the annotations with the shortest between-boxes distance, in
    # order to reduce overlapping between annotations.
    box_struct_pairs = sorted(box_struct_pairs,
                              key=lambda a: abs(a[1]['x'] - a[0]['x']))

    return box_struct_pairs


def _get_pvalue_thresholds(pvalue_thresholds, text_format):
    if pvalue_thresholds is DEFAULT:
        if text_format == "star":
            return [[1e-4, "****"], [1e-3, "***"],
                    [1e-2, "**"], [0.05, "*"], [1, "ns"]]
        else:
            return [[1e-5, "1e-5"], [1e-4, "1e-4"],
                    [1e-3, "0.001"], [1e-2, "0.01"],
                    [5e-2, "0.05"]]

    return pvalue_thresholds


def _get_box_plotter(plot, x, y, hue, data, order, hue_order,
                     **plot_params):

    if plot_params.pop("dodge", None) is False:
        raise ValueError(f"`dodge` must be true in statannotations")

    if plot == 'boxplot':
        # noinspection PyProtectedMember
        box_plotter = sns.categorical._BoxPlotter(
            x, y, hue, data, order, hue_order,
            orient=plot_params.get("orient"),
            width=plot_params.get("width", 0.8),
            dodge=True,
            fliersize=plot_params.get("fliersize", 5),
            linewidth=plot_params.get("linewidth"),
            saturation=.75, color=None, palette=None)

    elif plot == 'barplot':
        # noinspection PyProtectedMember
        box_plotter = sns.categorical._BarPlotter(
            x, y, hue, data, order, hue_order,
            estimator=plot_params.get("estimator", np.mean),
            ci=plot_params.get("ci", 95),
            n_boot=plot_params.get("nboot", 1000),
            units=plot_params.get("units"),
            orient=plot_params.get("orient"),
            seed=plot_params.get("seed"),
            color=None, palette=None, saturation=.75,
            errcolor=".26", errwidth=plot_params.get("errwidth"),
            capsize=None,
            dodge=True)

    else:
        raise NotImplementedError("Only boxplots and barplots are supported.")

    return box_plotter


def _get_offsets(loc, line_offset, line_offset_to_box):
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

    y_offset = line_offset
    return y_offset, line_offset_to_box


def _get_box_structs(box_plotter, box_names, labels, ymaxes):
    box_structs = [
        {
            'box': box_names[i],
            'label': labels[i],
            'x': _get_box_x_position(box_plotter, box_names[i]),
            'box_data': _get_box_data(box_plotter, box_names[i]),
            'ymax': ymaxes[box_names[i]]
        } for i in range(len(box_names))]

    # Sort the box data structures by position along the x axis
    box_structs = sorted(box_structs, key=lambda a: a['x'])

    # Add the index position in the list of boxes along the x axis
    box_structs = [dict(box_struct, xi=i)
                   for i, box_struct in enumerate(box_structs)]

    return box_structs


def _get_box_names_and_labels(box_plotter):
    group_names = box_plotter.group_names
    hue_names = box_plotter.hue_names

    if box_plotter.plot_hues is None:
        box_names = group_names
        labels = box_names
    else:
        box_names = [(group_name, hue_name) for group_name in group_names
                     for hue_name in hue_names]
        labels = ['{}_{}'.format(group_name, hue_name)
                  for (group_name, hue_name) in box_names]
    return box_names, labels


class Annotator:
    """
    Optionally computes statistical test between pairs of data series, and add
    statistical annotation on top of the boxes/bars. The same exact arguments
    `data`, `x`, `y`, `hue`, `order`, `width`, `hue_order` (and `units`) as in
    the seaborn boxplot/barplot function must be passed to this function.

    This function works in one of the two following modes:
    a) `perform_stat_test` is True (default):
        statistical test as given by argument `test` is performed.
        The `test_short_name` argument can be used to customize what appears
        before the pvalues if test is a string.
    b) `perform_stat_test` is False: no statistical test is performed, list of
        custom p-values `pvalues` are used for each pair of boxes.
    """
    def __init__(self, ax, box_pairs, plot='boxplot', data=None, x=None,
                 y=None, hue=None, order=None, hue_order=None, **plot_params):
        """
        :param ax: Ax of existing plot
        :param box_pairs: can be of either form:
            For non-grouped boxplot: `[(cat1, cat2), (cat3, cat4)]`.
            For boxplot grouped by hue: `[((cat1, hue1), (cat2, hue2)), ((cat3, hue3), (cat4, hue4))]`
        :param plot: type of the plot, one of 'boxplot' or 'barplot'.
        :param data: seaborn  plot's data
        :param x: seaborn plot's x
        :param y: seaborn plot's y
        :param hue: seaborn plot's hue
        :param order: seaborn plot's order
        :param hue_order: seaborn plot's hue_order
        :param plot_params: Other parameters for seaborn plotter
        """
        basic_plot = self.get_basic_plot(ax, box_pairs, plot, data, x, y, hue,
                                         order, hue_order, **plot_params)

        self.y_stack_arr = basic_plot[0]
        self.box_pairs = basic_plot[1]
        self.box_struct_pairs = basic_plot[2]
        self.fig = basic_plot[3]
        self.ax = ax

        self.test = None
        self.perform_stat_test = None
        self.test_results = None
        self.test_short_name = None
        self._pvalue_thresholds = _get_pvalue_thresholds(DEFAULT, "star")
        self.annotations = None
        self._text_format = None
        self._comparisons_correction = None
        self._pvalue_format_string, self._simple_format_string = (
            _get_pvalue_and_simple_formats(DEFAULT))
        self._loc = "inside"
        self.verbose = 1
        self._just_configured = False

    @staticmethod
    def get_basic_plot(
            ax, box_pairs, plot='boxplot', data=None, x=None, y=None, hue=None,
            order=None, hue_order=None, **plot_params):

        check_not_none("box_pairs", box_pairs)
        check_order_box_pairs_in_data(box_pairs, data, x, order, hue, hue_order)

        box_plotter = _get_box_plotter(plot, x, y, hue, data, order, hue_order,
                                       **plot_params)

        box_names, labels = _get_box_names_and_labels(box_plotter)

        fig = plt.gcf()
        ymaxes = _generate_ymaxes(ax, fig, box_plotter, box_names)
        box_structs = _get_box_structs(box_plotter, box_names, labels, ymaxes)

        box_struct_pairs = _get_box_struct_pairs(
            box_pairs, box_structs, box_names)

        y_stack_arr = np.array(
            [[box_struct['x'], box_struct['ymax'], 0]
             for box_struct in box_structs]
        ).T

        return y_stack_arr, box_pairs, box_struct_pairs, fig, ax

    def new_plot(self, ax, box_pairs=None, plot='boxplot', data=None, x=None,
                 y=None, hue=None, order=None, hue_order=None, **plot_params):

        if box_pairs is None:
            box_pairs = self.box_pairs

        basic_plot = self.get_basic_plot(ax, box_pairs, plot, data, x, y, hue,
                                         order, hue_order, **plot_params)

        self.y_stack_arr = basic_plot[0]
        self.box_pairs = basic_plot[1]
        self.box_struct_pairs = basic_plot[2]
        self.fig = basic_plot[3]
        self.ax = ax

    def reset_configuration(self):
        self.test = None
        self.perform_stat_test = None
        self.test_results = None
        self.test_short_name = None
        self._pvalue_thresholds = _get_pvalue_thresholds(DEFAULT, "star")
        self.annotations = None
        self._text_format = None
        self._comparisons_correction = None
        self._pvalue_format_string, self._simple_format_string = (
            _get_pvalue_and_simple_formats(DEFAULT))
        self._loc = "inside"
        self.verbose = 1
        self._just_configured = False

    @property
    def comparisons_correction(self):
        return self._comparisons_correction

    @comparisons_correction.setter
    def comparisons_correction(self, comparisons_correction):
        """
        :param comparisons_correction: Method for multiple comparisons correction.
            One of `statsmodels` `multipletests` methods (w/ default FWER), or a
            `ComparisonsCorrection` instance.
        """
        self._comparisons_correction = get_validated_comparisons_correction(
            comparisons_correction)

    @property
    def loc(self):
        return self._loc

    @loc.setter
    def loc(self, loc):
        check_is_in(
            loc, ['inside', 'outside'], label='argument `loc`'
        )
        self._loc = loc

    @property
    def text_format(self):
        return self._text_format

    @text_format.setter
    def text_format(self, text_format):
        """
        :param text_format: `star`, `simple` or `full`
        """
        check_valid_text_format(text_format)
        self._text_format = text_format

    @property
    def pvalue_format_string(self):
        return self._pvalue_format_string

    @property
    def simple_format_string(self):
        return self._simple_format_string

    @pvalue_format_string.setter
    def pvalue_format_string(self, pvalue_format_string):
        """
        :param pvalue_format_string: By default is `"{:.3e}"`, or `"{:.2f}"`
            for `"simple"` format
        """
        self._pvalue_format_string, self._simple_format_string = (
            _get_pvalue_and_simple_formats(pvalue_format_string)
        )

    @property
    def pvalue_thresholds(self):
        return self._pvalue_thresholds

    @pvalue_thresholds.setter
    def pvalue_thresholds(self, pvalue_thresholds):
        """
        :param pvalue_thresholds: list of lists, or tuples.
            Default is:
            For "star" text_format: `[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, "ns"]]`.
            For "simple" text_format : `[[1e-5, "1e-5"], [1e-4, "1e-4"], [1e-3, "0.001"], [1e-2, "0.01"], [5e-2, "0.05"]]`
        """
        self._pvalue_thresholds = _get_pvalue_thresholds(
            pvalue_thresholds, text_format=self.text_format)

    def configure(self, **parameters):
        for parameter in parameters:
            if parameter not in CONFIGURABLE_PARAMETERS:
                raise ValueError(f"Parameter `{parameter}` is invalid to "
                                 f"configure annotator.")

        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        self._just_configured = True

        return self

    def apply_test(self, num_comparisons='auto', **stats_params):
        """
        :param stats_params: Parameters for statistical test functions.

        :param num_comparisons: Override number of comparisons otherwise
            calculated with number of box_pairs
        """

        _check_test_pvalues_perform(self.test, self.test_short_name)

        if stats_params is None:
            stats_params = dict()

        self.perform_stat_test = True

        self.annotations = self._get_results(num_comparisons=num_comparisons,
                                             **stats_params)
        self._just_configured = False

        return self

    def set_custom_annotation(self, pvalues, test_short_name=None,
                              num_comparisons='auto'):
        """
        :param pvalues: list or array of p-values for each box pair comparison.
        :param num_comparisons: Override number of comparisons otherwise
            calculated with number of box_pairs
        :param test_short_name: Name to use for labelling the custom
            statistical test results.
        """

        _check_test_pvalues_no_perform(self.test, pvalues, self.box_pairs)

        self.perform_stat_test = False

        self.annotations = self._get_results(test_short_name=test_short_name,
                                             num_comparisons=num_comparisons,
                                             pvalues=pvalues)
        self._just_configured = False

        return self

    def add_stat_annotation(self,
                            text_annot_custom=None,
                            show_test_name=True,
                            use_fixed_offset=False,
                            line_offset_to_box=None, line_offset=None,
                            line_height=0.02, text_offset=1, color='0.2',
                            line_width=1.5, fontsize='medium', verbose=1):

        """
        :param line_height: in axes fraction coordinates
        :param text_offset: in points
        """
        if self._just_configured and text_annot_custom is None:
            warnings.warn("Annotator was reconfigured without applying the "
                          "test (again) which will probably lead to unexpected "
                          "results")
        check_correct_number_custom_annotations(text_annot_custom,
                                                self.box_struct_pairs)
        ax_to_data = _get_transform_func(self.ax, 'ax_to_data')
        ann_list = []
        ymaxs = []
        y_stack = []
        orig_ylim = self.ax.get_ylim()

        ylim = (0, 1)

        if self.loc == 'outside':
            self.y_stack_arr[1, :] = ylim[1]

        if verbose >= 1 and self.text_format == 'star':
            self.print_pvalue_legend()

        y_offset, line_offset_to_box = _get_offsets(
            self.loc, line_offset, line_offset_to_box)

        for box_structs, result in zip(self.box_struct_pairs, self.annotations):

            x1 = box_structs[0]['x']
            x2 = box_structs[1]['x']
            xi1 = box_structs[0]['xi']
            xi2 = box_structs[1]['xi']

            if text_annot_custom is not None:
                i_box_pair = box_structs[0]['i_box_pair']
                text = text_annot_custom[i_box_pair]

            else:
                text = self._get_text(box_structs, result, self.test_short_name,
                                      show_test_name)

            # Find y maximum for all the y_stacks *in between* the box1 and the box2
            i_ymax_in_range_x1_x2 = xi1 + np.nanargmax(
                self.y_stack_arr[1, np.where((x1 <= self.y_stack_arr[0, :])
                                             & (self.y_stack_arr[0, :] <= x2))])
            ymax_in_range_x1_x2 = self.y_stack_arr[1, i_ymax_in_range_x1_x2]

            yref = ymax_in_range_x1_x2
            yref2 = yref

            # Choose the best offset depending on whether there is an annotation below
            # at the x position in the range [x1, x2] where the stack is the highest
            if self.y_stack_arr[2, i_ymax_in_range_x1_x2] == 0:
                # there is only a box below
                offset = line_offset_to_box
            else:
                # there is an annotation below
                offset = y_offset

            y = yref2 + offset

            # Determine lines in axes coordinates
            ax_line_x, ax_line_y = [x1, x1, x2, x2], [y, y + line_height, y + line_height, y]
            # Then transform the resulting points from axes coordinates to data coordinates
            points = [ax_to_data.transform((x, y)) for x, y in zip(ax_line_x, ax_line_y)]
            line_x, line_y = [x for x, y in points], [y for x, y in points]

            if self.loc == 'inside':
                self.ax.plot(line_x, line_y, lw=line_width, c=color)
            else:
                line = lines.Line2D(line_x, line_y, lw=line_width, c=color,
                                    transform=self.ax.transData)
                line.set_clip_on(False)
                self.ax.add_line(line)

            if text is not None:
                ann = self.ax.annotate(
                    text, xy=(np.mean([x1, x2]), line_y[2]),
                    xytext=(0, text_offset), textcoords='offset points',
                    xycoords='data', ha='center', va='bottom',
                    fontsize=fontsize, clip_on=False, annotation_clip=False)
                ann_list.append(ann)

                plt.draw()
                self.ax.set_ylim(orig_ylim)
                data_to_ax, ax_to_data, pix_to_ax = _get_transform_func(
                    self.ax, 'all')

                y_top_annot = None
                got_mpl_error = False
                if not use_fixed_offset:
                    try:
                        bbox = ann.get_window_extent()
                        bbox_ax = bbox.transformed(pix_to_ax)
                        y_top_annot = bbox_ax.ymax
                    except RuntimeError:
                        got_mpl_error = True

                if use_fixed_offset or got_mpl_error:
                    if verbose >= 1:
                        print("Warning: cannot get the text bounding box. Falling "
                              "back to a fixed y offset. Layout may be not optimal.")

                    # We will apply a fixed offset in points,
                    # based on the font size of the annotation.
                    fontsize_points = FontProperties(size='medium').get_size_in_points()
                    offset_trans = mtransforms.offset_copy(
                        self.ax.transAxes, fig=self.fig, x=0,
                        y=1.0 * fontsize_points + text_offset, units='points')
                    y_top_display = offset_trans.transform((0, y + line_height))
                    y_top_annot = self.ax.transAxes.inverted().transform(y_top_display)[1]
            else:
                y_top_annot = y + line_height

            # remark: y_stack is not really necessary if we have the stack_array
            y_stack.append(y_top_annot)
            ymaxs.append(max(y_stack))

            # Fill the highest y position of the annotation into the y_stack array
            # for all positions in the range x1 to x2
            self.y_stack_arr[1,
                             (x1 <= self.y_stack_arr[0, :])
                             & (self.y_stack_arr[0, :] <= x2)] = y_top_annot
            # Increment the counter of annotations in the y_stack array
            self.y_stack_arr[2, xi1:xi2 + 1] = self.y_stack_arr[2, xi1:xi2 + 1] + 1

        y_stack_max = max(self.y_stack_arr[1, :])

        # reset transformation
        ax_to_data = _get_transform_func(self.ax, 'ax_to_data')
        ylims = ([(0, 0), (0, max(1.03 * y_stack_max, 1))]
                 if self.loc == 'inside'
                 else [(0, 0), (0, 1)])

        self.ax.set_ylim(ax_to_data.transform(ylims)[:, 1])

        return self.ax, self.annotations

    def print_pvalue_legend(self):
        print("p-value annotation legend:")

        pvalue_thresholds = sort_pvalue_thresholds(self.pvalue_thresholds)

        for i in range(0, len(pvalue_thresholds)):
            if i < len(pvalue_thresholds) - 1:
                print('{}: {:.2e} < p <= {:.2e}'
                      .format(pvalue_thresholds[i][1],
                              pvalue_thresholds[i + 1][0],
                              pvalue_thresholds[i][0]))
            else:
                print('{}: p <= {:.2e}'.format(pvalue_thresholds[i][1],
                                               pvalue_thresholds[i][0]))
        print()

    def _get_results(self, num_comparisons='auto', test_short_name=None,
                     pvalues=None, **stats_params):

        test_result_list = []

        if num_comparisons == "auto":
            num_comparisons = len(self.box_struct_pairs)

        for box_struct1, box_struct2 in self.box_struct_pairs:
            box1 = box_struct1['box']
            box2 = box_struct2['box']

            if self.perform_stat_test:
                result = _get_stat_result_from_test(
                    box_struct1, box_struct2, self.test,
                    self.comparisons_correction,
                    num_comparisons, self.pvalue_thresholds,
                    **stats_params)
            else:
                result = _get_custom_results(box_struct1, test_short_name,
                                             self.pvalue_thresholds, pvalues)

            result.box1 = box1
            result.box2 = box2

            test_result_list.append(result)

        # Perform other types of correction methods for multiple testing
        _apply_comparisons_correction(self.comparisons_correction,
                                      test_result_list)

        return test_result_list

    def _get_text(self, box_structs, result, test_short_name,
                  show_test_name):

        label1 = box_structs[0]['label']
        label2 = box_structs[1]['label']

        if self.verbose >= 1:
            print(f"{label1} v.s. {label2}: {result.formatted_output}")

        if self.text_format == 'full':
            text = ("{} p = {}{}"
                    .format('{}', self.pvalue_format_string, '{}')
                    .format(result.test_short_name, result.pval,
                            result.significance_suffix))

        elif self.text_format == 'star':
            text = pval_annotation_text(result, self.pvalue_thresholds)

        elif self.text_format == 'simple':
            if show_test_name:
                if test_short_name is None:
                    test_short_name = (self.test
                                       if isinstance(self.test, str)
                                       else self.test.short_name)
            else:
                test_short_name = ""
            text = simple_text(result, self.simple_format_string,
                               self.pvalue_thresholds, test_short_name)
        else:  # None:
            text = None

        return text