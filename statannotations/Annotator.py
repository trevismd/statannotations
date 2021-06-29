import warnings
from typing import List

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import seaborn as sns
from matplotlib import lines
from matplotlib.collections import PathCollection
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

from statannotations.format_annotations import pval_annotation_text, \
    simple_text
from statannotations.stats.ComparisonsCorrection import \
    get_validated_comparisons_correction
from statannotations.stats.StatResult import StatResult
from statannotations.stats.StatTest import StatTest
from statannotations.stats.tests import stat_test, IMPLEMENTED_TESTS
from statannotations.stats.utils import check_alpha
from statannotations.utils import check_is_in, remove_null, \
    check_box_pairs_in_data, check_not_none, check_valid_text_format, \
    check_order_in_data, render_collection

DEFAULT = object()

CONFIGURABLE_PARAMETERS = [
    'comparisons_correction',
    'loc',
    'pvalue_format_string',
    'pvalue_thresholds',
    'test',
    'test_short_name',
    'text_format',
    'verbose',
    'show_test_name',
    'use_fixed_offset',
    'line_offset_to_box',
    'line_offset',
    'line_height',
    'text_offset',
    'color',
    'line_width',
    'fontsize'
]

IMPLEMENTED_PLOTTERS = ['boxplot', 'stripplot', 'barplot', 'swarmplot']


def check_implemented_plotters(plot):
    if plot not in IMPLEMENTED_PLOTTERS:
        raise NotImplementedError(
            f"Only {render_collection(IMPLEMENTED_PLOTTERS)} are supported.")


class Annotator:
    """
    Optionally computes statistical test between pairs of data series, and add
    statistical annotation on top of the boxes/bars. The same exact arguments
    `data`, `x`, `y`, `hue`, `order`, `width`, `hue_order` (and `units`) as in
    the seaborn boxplot/barplot function must be passed to this function.

    This function works in one of the two following modes:
    a) `perform_stat_test` is True (default):
    * statistical test as given by argument `test` is performed.
    * The `test_short_name` argument can be used to customize what appears
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
            For boxplot grouped by hue: `[
                ((cat1, hue1), (cat2, hue2)), ((cat3, hue3), (cat4, hue4))
            ]`
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
        self.box_pairs = box_pairs

        self.y_stack_arr = basic_plot[0]
        self.box_struct_pairs = basic_plot[1]
        self.fig = basic_plot[2]
        self.reordering = self._sort_box_struct_pairs()
        self.ax = ax

        self._test = None
        self.perform_stat_test = None
        self.test_short_name = None
        self._text_format = "star"
        self._default_pvalue_thresholds = True
        self._pvalue_format_string = '{:.3e}'
        self._simple_format_string = '{:.2f}'
        self._alpha = 0.05
        self._pvalue_thresholds = self._get_pvalue_thresholds(DEFAULT)

        self.annotations = None
        self._comparisons_correction = None

        self._loc = "inside"
        self.verbose = 1
        self._just_configured = True
        self.show_test_name = True
        self.use_fixed_offset = False
        self.line_offset_to_box = None
        self.line_offset = None
        self.line_height = 0.02
        self.text_offset = 1
        self.color = '0.2'
        self.line_width = 1.5
        self.fontsize = 'medium'
        self.y_offset = None
        self.custom_annotations = None

        self.OFFSET_FUNCS = {"inside": Annotator._get_offsets_inside,
                             "outside": Annotator._get_offsets_outside}

    @staticmethod
    def get_basic_plot(
            ax, box_pairs, plot='boxplot', data=None, x=None, y=None, hue=None,
            order=None, hue_order=None, **plot_params):
        check_implemented_plotters(plot)
        check_not_none("box_pairs", box_pairs)
        check_order_in_data(data, x, order)
        check_box_pairs_in_data(box_pairs, data, x, hue, hue_order)

        box_plotter = Annotator._get_plotter(plot, x, y, hue, data, order,
                                             hue_order, **plot_params)

        box_names, labels = Annotator._get_box_names_and_labels(box_plotter)

        fig = plt.gcf()
        ymaxes = Annotator._generate_ymaxes(ax, fig, box_plotter, box_names)
        box_structs = Annotator._get_box_structs(
            box_plotter, box_names, labels, ymaxes)

        box_struct_pairs = Annotator._get_box_struct_pairs(
            box_pairs, box_structs)

        y_stack_arr = np.array(
            [[box_struct['x'], box_struct['ymax'], 0]
             for box_struct in box_structs]
        ).T

        return y_stack_arr, box_struct_pairs, fig

    def new_plot(self, ax, box_pairs=None, plot='boxplot', data=None, x=None,
                 y=None, hue=None, order=None, hue_order=None, **plot_params):
        self.ax = ax

        if box_pairs is None:
            box_pairs = self.box_pairs

        basic_plot = self.get_basic_plot(ax, box_pairs, plot, data, x, y, hue,
                                         order, hue_order, **plot_params)
        self.y_stack_arr = basic_plot[0]
        self.box_struct_pairs = basic_plot[1]
        self.fig = basic_plot[2]

        self.reordering = self._sort_box_struct_pairs()

        self.line_offset = None
        self.line_offset_to_box = None
        self.perform_stat_test = None
        return self

    def reset_configuration(self):
        self._test = None
        self.test_short_name = None
        self._text_format = "star"
        self._default_pvalue_thresholds = True
        self._pvalue_format_string = '{:.3e}'
        self._simple_format_string = '{:.2f}'
        self._alpha = 0.05
        self._pvalue_thresholds = self._get_pvalue_thresholds(DEFAULT)

        self.annotations = None
        self._comparisons_correction = None
        self._loc = "inside"
        self.verbose = 1
        self._just_configured = True
        self.show_test_name = True
        self.use_fixed_offset = False
        self.line_height = 0.02
        self.text_offset = 1
        self.color = '0.2'
        self.line_width = 1.5
        self.fontsize = 'medium'
        self.custom_annotations = None

        return self

    def annotate(self, line_offset=None, line_offset_to_box=None):

        if self.should_warn_about_configuration:
            warnings.warn("Annotator was reconfigured without applying the "
                          "test (again) which will probably lead to "
                          "unexpected results")

        self.update_y_for_loc()

        ann_list = []
        ymaxs = []
        y_stack = []
        orig_ylim = self.ax.get_ylim()

        offset_func = self.OFFSET_FUNCS[self.loc]
        self.y_offset, self.line_offset_to_box = offset_func(
            line_offset, line_offset_to_box)

        if self.verbose and self.text_format == 'star':
            self.print_pvalue_legend()

        ax_to_data = Annotator._get_transform_func(self.ax, 'ax_to_data')

        self.validate_test_short_name()

        for box_structs, result in zip(self.box_struct_pairs,
                                       self.annotations):
            self._annotate_pair(box_structs, result,
                                ax_to_data=ax_to_data,
                                ann_list=ann_list,
                                ymaxs=ymaxs,
                                y_stack=y_stack,
                                orig_ylim=orig_ylim)

        y_stack_max = max(self.y_stack_arr[1, :])

        # reset transformation
        ax_to_data = Annotator._get_transform_func(self.ax, 'ax_to_data')
        ylims = ([(0, 0), (0, max(1.03 * y_stack_max, 1))]
                 if self.loc == 'inside'
                 else [(0, 0), (0, 1)])

        self.ax.set_ylim(ax_to_data.transform(ylims)[:, 1])

        return self.ax, self.annotations

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        check_alpha(alpha)
        self._alpha = alpha

    @property
    def comparisons_correction(self):
        return self._comparisons_correction

    @comparisons_correction.setter
    def comparisons_correction(self, comparisons_correction):
        """
        :param comparisons_correction: Method for multiple comparisons
            correction. One of `statsmodels` `multipletests` methods
            (w/ default FWER), or a `ComparisonsCorrection` instance.
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
    def test(self):
        return self._test

    @test.setter
    def test(self, test):
        """
        :param test: Name of a test for `StatTest.from_library` or a
            `StatTest` instance
        """
        self._test = test

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
            Annotator._get_pvalue_and_simple_formats(pvalue_format_string)
        )

    @property
    def pvalue_thresholds(self):
        return self._pvalue_thresholds

    @pvalue_thresholds.setter
    def pvalue_thresholds(self, pvalue_thresholds):
        """
        :param pvalue_thresholds: list of lists, or tuples.
            Default is:
            For "star" text_format: `[
                [1e-4, "****"],
                [1e-3, "***"],
                [1e-2, "**"],
                [0.05, "*"],
                [1, "ns"]
            ]`.

            For "simple" text_format : `[
                [1e-5, "1e-5"],
                [1e-4, "1e-4"],
                [1e-3, "0.001"],
                [1e-2, "0.01"],
                [5e-2, "0.05"]
            ]`
        """
        self._pvalue_thresholds = self._get_pvalue_thresholds(
            pvalue_thresholds)

    def configure(self, **parameters):
        """

        * `show_test_name`: Set to False to not show the (short) name of
            test
        * `line_height`: in axes fraction coordinates
        * `text_offset`: in points
        """
        unmatched_parameters = parameters.keys() - set(CONFIGURABLE_PARAMETERS)
        if unmatched_parameters:
            raise ValueError(f"Invalid parameter(s) "
                             f"`{render_collection(unmatched_parameters)}` "
                             f"to configure annotator.")

        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        self.activate_configured_warning()

        new_threshold = parameters.get("pvalue_thresholds")
        if new_threshold is not None:
            self._default_pvalue_thresholds = new_threshold == DEFAULT

        elif parameters.get("alpha"):
            warnings.warn("Changing alpha without updating pvalue_thresholds"
                          "can result in inconsistent plotting results")

        if parameters.get("text_format") is not None:
            self._update_pvalue_thresholds()

        return self

    def apply_test(self, num_comparisons='auto', **stats_params):
        """
        :param stats_params: Parameters for statistical test functions.

        :param num_comparisons: Override number of comparisons otherwise
            calculated with number of box_pairs
        """

        self._check_test_pvalues_perform()

        if stats_params is None:
            stats_params = dict()

        self.perform_stat_test = True

        self.annotations = self._get_results(num_comparisons=num_comparisons,
                                             **stats_params)
        self.deactivate_configured_warning()

        return self

    @property
    def should_warn_about_configuration(self):
        return self._just_configured

    def set_pvalues(self, pvalues,
                    num_comparisons='auto'):
        """
        :param pvalues: list or array of p-values for each box pair comparison.
        :param num_comparisons: Override number of comparisons otherwise
            calculated with number of box_pairs
        """

        self._check_test_pvalues_no_perform(pvalues)

        self.perform_stat_test = False

        self.annotations = self._get_results(
            test_short_name=self.test_short_name,
            num_comparisons=num_comparisons, pvalues=pvalues)

        self.deactivate_configured_warning()

        return self

    def set_custom_annotation(self, text_annot_custom):
        """
        :param text_annot_custom: List of values to annotate for each
            `box_pair`
        """
        self._check_correct_number_custom_annotations(text_annot_custom)
        self.custom_annotations = text_annot_custom
        self.deactivate_configured_warning()
        return self

    def activate_configured_warning(self):
        self._just_configured = True

    def deactivate_configured_warning(self):
        self._just_configured = False

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

    def _get_results(self, num_comparisons='auto', pvalues=None,
                     **stats_params) -> List[StatResult]:

        test_result_list = []

        if num_comparisons == "auto":
            num_comparisons = len(self.box_struct_pairs)

        for box_struct1, box_struct2 in self.box_struct_pairs:
            box1 = box_struct1['box']
            box2 = box_struct2['box']

            if self.perform_stat_test:
                result = self._get_stat_result_from_test(
                    box_struct1, box_struct2, num_comparisons, **stats_params)
            else:
                result = self._get_custom_results(box_struct1, pvalues)

            result.box1 = box1
            result.box2 = box2

            test_result_list.append(result)

        # Perform other types of correction methods for multiple testing
        Annotator._apply_comparisons_correction(self.comparisons_correction,
                                                test_result_list)

        return test_result_list

    def _get_text(self, result):

        if self.text_format == 'full':
            text = ("{} p = {}{}"
                    .format('{}', self.pvalue_format_string, '{}')
                    .format(result.test_short_name, result.pval,
                            result.significance_suffix))

        elif self.text_format == 'star':
            text = pval_annotation_text(result, self.pvalue_thresholds)

        elif self.text_format == 'simple':
            text = simple_text(result, self.simple_format_string,
                               self.pvalue_thresholds, self.test_short_name)
        else:
            text = None

        return text

    def _annotate_pair(self, box_structs, result, ax_to_data,
                       ann_list, ymaxs, y_stack, orig_ylim):

        x1 = box_structs[0]['x']
        x2 = box_structs[1]['x']
        xi1 = box_structs[0]['xi']
        xi2 = box_structs[1]['xi']

        if self.custom_annotations is not None:
            i_box_pair = box_structs[0]['i_box_pair']
            text = self.custom_annotations[i_box_pair]

        else:
            if self.verbose >= 1:
                _print_result_line(box_structs, result.formatted_output)

            text = self._get_text(result)

        # Find y maximum for all the y_stacks *in between* box1 and box2
        i_ymax_in_range_x1_x2 = xi1 + np.nanargmax(
            self.y_stack_arr[1, np.where((x1 <= self.y_stack_arr[0, :])
                                         & (self.y_stack_arr[0, :] <= x2))])
        ymax_in_range_x1_x2 = self.y_stack_arr[1, i_ymax_in_range_x1_x2]

        yref = ymax_in_range_x1_x2
        yref2 = yref

        # Choose the best offset depending on whether there is an annotation
        # below at the x position in the range [x1, x2] where the stack is the
        # highest
        if self.y_stack_arr[2, i_ymax_in_range_x1_x2] == 0:
            # there is only a box below
            offset = self.line_offset_to_box
        else:
            # there is an annotation below
            offset = self.y_offset

        y = yref2 + offset

        # Determine lines in axes coordinates
        ax_line_x = [x1, x1, x2, x2]
        ax_line_y = [y, y + self.line_height, y + self.line_height, y]

        points = [ax_to_data.transform((x, y))
                  for x, y
                  in zip(ax_line_x, ax_line_y)]

        line_x, line_y = [x for x, y in points], [y for x, y in points]

        if self.loc == 'inside':
            self.ax.plot(line_x, line_y, lw=self.line_width, c=self.color)
        else:
            line = lines.Line2D(line_x, line_y, lw=self.line_width,
                                c=self.color, transform=self.ax.transData)
            line.set_clip_on(False)
            self.ax.add_line(line)

        if text is not None:
            ann = self.ax.annotate(
                text, xy=(np.mean([x1, x2]), line_y[2]),
                xytext=(0, self.text_offset), textcoords='offset points',
                xycoords='data', ha='center', va='bottom',
                fontsize=self.fontsize, clip_on=False, annotation_clip=False)
            ann_list.append(ann)

            plt.draw()
            self.ax.set_ylim(orig_ylim)
            data_to_ax, ax_to_data, pix_to_ax = Annotator._get_transform_func(
                self.ax, 'all')

            y_top_annot = None
            got_mpl_error = False
            if not self.use_fixed_offset:
                try:
                    bbox = ann.get_window_extent()
                    bbox_ax = bbox.transformed(pix_to_ax)
                    y_top_annot = bbox_ax.ymax
                except RuntimeError:
                    got_mpl_error = True

            if self.use_fixed_offset or got_mpl_error:
                if self.verbose >= 1:
                    print("Warning: cannot get the text bounding box. Falling "
                          "back to a fixed y offset. Layout may be not "
                          "optimal.")

                # We will apply a fixed offset in points,
                # based on the font size of the annotation.
                fontsize_points = FontProperties(
                    size='medium').get_size_in_points()
                offset_trans = mtransforms.offset_copy(
                    self.ax.transAxes, fig=self.fig, x=0,
                    y=1.0 * fontsize_points + self.text_offset, units='points')

                y_top_display = offset_trans.transform(
                    (0, y + self.line_height))
                y_top_annot = (self.ax.transAxes.inverted()
                               .transform(y_top_display)[1])
        else:
            y_top_annot = y + self.line_height

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

    def update_y_for_loc(self):
        if self._loc == 'outside':
            self.y_stack_arr[1, :] = 1

    @staticmethod
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

    @staticmethod
    def _get_xpos_location(pos, xranges):
        """
        Finds the x-axis location of a categorical variable
        """
        for xrange in xranges:
            if (pos >= xrange[0]) & (pos <= xrange[1]):
                return xrange[2]

    @staticmethod
    def _get_ypos_for_line2d_or_rectangle(
            xpositions, data_to_ax, ax, fig, child):
        xunits = (max(list(xpositions.keys())) + 1) / len(xpositions)
        xranges = {(pos - xunits / 2, pos + xunits / 2, pos): box_name
                   for pos, box_name in xpositions.items()}
        box = ax.transData.inverted().transform(
            child.get_window_extent(fig.canvas.get_renderer()))

        if (box[:, 0].max() - box[:, 0].min()) > 1.1 * xunits:
            return None, None
        raw_xpos = np.round(box[:, 0].mean(), 1)
        xpos = Annotator._get_xpos_location(raw_xpos, xranges)
        if xpos not in xpositions:
            return None, None
        xname = xpositions[xpos]
        ypos = box[:, 1].max()
        ypos = data_to_ax.transform((0, ypos))[1]

        return xname, ypos

    @staticmethod
    def _get_ypos_for_path_collection(xpositions, data_to_ax, child):
        ymax = child.properties()['offsets'][:, 1].max()
        xpos = float(np.round(np.nanmean(
            child.properties()['offsets'][:, 0]), 1))

        xname = xpositions[xpos]
        ypos = data_to_ax.transform((0, ymax))[1]

        return xname, ypos

    @staticmethod
    def _generate_ymaxes(ax, fig, box_plotter, box_names):
        """
        given box plotter and the names of two categorical variables,
        returns highest y point drawn between those two variables before
        annotations.

        The highest data point is often not the highest item drawn
        (eg, error bars and/or bar charts).
        """
        xpositions = {
            np.round(Annotator._get_box_x_position(box_plotter, box_name),
                     1): box_name
            for box_name in box_names}

        ymaxes = {name: 0 for name in box_names}

        data_to_ax = Annotator._get_transform_func(ax, 'data_to_ax')

        for child in ax.get_children():
            if ((type(child) == PathCollection)
                    and len(child.properties()['offsets'])):
                xname, ypos = Annotator._get_ypos_for_path_collection(
                    xpositions, data_to_ax, child)

            elif (type(child) == lines.Line2D) or (type(child) == Rectangle):
                xname, ypos = Annotator._get_ypos_for_line2d_or_rectangle(
                    xpositions, data_to_ax, ax, fig, child)
                if xname is None:
                    continue
            else:
                continue

            if ypos > ymaxes[xname]:
                ymaxes[xname] = ypos
        return ymaxes

    @staticmethod
    def _get_group_data(b_plotter, cat):
        index = b_plotter.group_names.index(cat)

        return b_plotter.plot_data[index], index

    @staticmethod
    def _get_box_data_with_hue(b_plotter, box_name):
        cat = box_name[0]
        group_data, index = Annotator._get_group_data(b_plotter, cat)

        hue_level = box_name[1]
        hue_mask = b_plotter.plot_hues[index] == hue_level

        return group_data[hue_mask]

    @staticmethod
    def _get_box_data_without_hue(b_plotter, box_name):
        cat = box_name
        group_data, _ = Annotator._get_group_data(b_plotter, cat)

        return group_data

    @staticmethod
    def _get_box_data(b_plotter, box_name):
        """
        box_name can be either a name "cat" or a tuple ("cat", "hue")

        Almost a duplicate of seaborn's code, because there is not
        direct access to the box_data in the BoxPlotter class.
        """
        if b_plotter.plot_hues is None:
            data = Annotator._get_box_data_without_hue(b_plotter, box_name)
        else:
            data = Annotator._get_box_data_with_hue(b_plotter, box_name)

        box_data = remove_null(data)

        return box_data

    @staticmethod
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

        check_is_in(kind, ['data_to_ax', 'ax_to_data', 'pix_to_ax', 'all'],
                    'kind')

        if kind == 'pix_to_ax':
            return ax.transAxes.inverted()

        data_to_ax = ax.transData + ax.get_xaxis_transform().inverted()

        if kind == 'data_to_ax':
            return data_to_ax

        elif kind == 'ax_to_data':
            return data_to_ax.inverted()

        else:
            return data_to_ax, data_to_ax.inverted(), ax.transAxes.inverted()

    def _check_test_pvalues_perform(self):
        if self.test is None:
            raise ValueError("If `perform_stat_test` is True, "
                             "`test` must be specified.")

        if self.test not in IMPLEMENTED_TESTS and \
                not isinstance(self.test, StatTest):
            raise ValueError("test value should be a StatTest instance "
                             "or one of the following strings: {}."
                             .format(', '.join(IMPLEMENTED_TESTS)))

    def _check_test_pvalues_no_perform(self, pvalues):
        if pvalues is None:
            raise ValueError("If `perform_stat_test` is False, "
                             "custom `pvalues` must be specified.")
        if self.test is not None:
            raise ValueError("If `perform_stat_test` is False, "
                             "`test` must be None.")
        if len(pvalues) != len(self.box_pairs):
            raise ValueError("`pvalues` should be of the same length as "
                             "`box_pairs`.")

    def _check_correct_number_custom_annotations(self, text_annot_custom):
        if text_annot_custom is None:
            return

        if len(text_annot_custom) != len(self.box_struct_pairs):
            raise ValueError(
                "`text_annot_custom` should be of same length as `box_pairs`.")

    @staticmethod
    def _apply_type1_comparisons_correction(comparisons_correction,
                                            test_result_list):
        original_pvalues = [result.pval for result in test_result_list]

        significant_pvalues = comparisons_correction(original_pvalues)
        correction_name = comparisons_correction.name
        for is_significant, result in zip(significant_pvalues,
                                          test_result_list):
            result.correction_method = correction_name
            result.corrected_significance = is_significant

    @staticmethod
    def _apply_type0_comparisons_correction(comparisons_correction,
                                            test_result_list):
        alpha = comparisons_correction.alpha
        for result in test_result_list:
            result.correction_method = comparisons_correction.name
            result.corrected_significance = (
                    result.pval < alpha
                    or np.isclose(result.pval, alpha))

    @staticmethod
    def _apply_comparisons_correction(comparisons_correction,
                                      test_result_list):
        if comparisons_correction is None:
            return

        if comparisons_correction.type == 1:
            Annotator._apply_type1_comparisons_correction(
                comparisons_correction, test_result_list)

        else:
            Annotator._apply_type0_comparisons_correction(
                comparisons_correction, test_result_list)

    @staticmethod
    def _get_pvalue_and_simple_formats(pvalue_format_string):
        if pvalue_format_string is DEFAULT:
            pvalue_format_string = '{:.3e}'
            simple_format_string = '{:.2f}'
        else:
            simple_format_string = pvalue_format_string

        return pvalue_format_string, simple_format_string

    def _get_stat_result_from_test(self, box_struct1, box_struct2,
                                   num_comparisons,
                                   **stats_params) -> StatResult:
        box_data1 = box_struct1['box_data']
        box_data2 = box_struct2['box_data']

        result = stat_test(
            box_data1,
            box_data2,
            self.test,
            comparisons_correction=self.comparisons_correction,
            num_comparisons=num_comparisons,
            alpha=self.alpha,
            **stats_params
        )

        return result

    def _get_custom_results(self, box_struct1,
                            pvalues) -> StatResult:
        pvalue = pvalues[box_struct1['i_box_pair']]

        if self.has_type0_comparisons_correction():
            pvalue = self.comparisons_correction(pvalue, len(pvalues))

        result = StatResult(
            'Custom statistical test',
            self.test_short_name,
            None,
            None,
            pval=pvalue,
            alpha=self.alpha
        )

        return result

    @staticmethod
    def _get_box_struct_pairs(box_pairs, box_structs):
        box_structs_dic = {box_struct['box']: box_struct
                           for box_struct in box_structs}

        box_struct_pairs = []

        for i_box_pair, (box1, box2) in enumerate(box_pairs):
            box_struct1 = dict(box_structs_dic[box1], i_box_pair=i_box_pair)
            box_struct2 = dict(box_structs_dic[box2], i_box_pair=i_box_pair)

            if box_struct1['x'] <= box_struct2['x']:
                box_struct_pairs.append((box_struct1, box_struct2))
            else:
                box_struct_pairs.append((box_struct2, box_struct1))

        return box_struct_pairs

    @staticmethod
    def absolute_box_struct_pair_in_tuple_x_diff(box_struct_pair):
        return abs(box_struct_pair[1][1]['x'] - box_struct_pair[1][0]['x'])

    def _sort_box_struct_pairs(self):
        # Draw first the annotations with the shortest between-boxes distance,
        # in order to reduce overlapping between annotations.
        new_box_struct_pairs = sorted(
            enumerate(self.box_struct_pairs),
            key=self.absolute_box_struct_pair_in_tuple_x_diff)

        self.box_struct_pairs = [box_struct_pair
                                 for index, box_struct_pair
                                 in new_box_struct_pairs]

        reordering = [index for index, box_struct_pair in new_box_struct_pairs]
        return reordering

    def _get_pvalue_thresholds(self, pvalue_thresholds):
        if self._default_pvalue_thresholds:
            if self.text_format == "star":
                return [[1e-4, "****"], [1e-3, "***"],
                        [1e-2, "**"], [0.05, "*"], [1, "ns"]]
            else:
                return [[1e-5, "1e-5"], [1e-4, "1e-4"],
                        [1e-3, "0.001"], [1e-2, "0.01"],
                        [5e-2, "0.05"]]

        return pvalue_thresholds

    @staticmethod
    def _get_plotter(plot, x, y, hue, data, order, hue_order,
                     **plot_params):
        if not plot_params.pop("dodge", True):
            raise ValueError("`dodge` must be True in statannotations")

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

        elif plot == 'swarmplot':
            # noinspection PyProtectedMember
            box_plotter = sns.categorical._SwarmPlotter(
                x, y, hue, data, order, hue_order,
                orient=plot_params.get("orient"),
                dodge=True, color=None, palette=None)

        elif plot == 'stripplot':
            # noinspection PyProtectedMember
            box_plotter = sns.categorical._StripPlotter(
                x, y, hue, data, order, hue_order,
                jitter=plot_params.get("jitter", True),
                orient=plot_params.get("orient"),
                dodge=True, color=None, palette=None)

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
            raise NotImplementedError(
                f"Only {render_collection(IMPLEMENTED_PLOTTERS)} are "
                f"supported.")

        return box_plotter

    @staticmethod
    def _get_offsets_inside(line_offset, line_offset_to_box):
        if line_offset is None:
            line_offset = 0.05
            if line_offset_to_box is None:
                line_offset_to_box = 0.06
        else:
            if line_offset_to_box is None:
                line_offset_to_box = 0.06
        y_offset = line_offset
        return y_offset, line_offset_to_box

    @staticmethod
    def _get_offsets_outside(line_offset, line_offset_to_box):
        if line_offset is None:
            line_offset = 0.03
            if line_offset_to_box is None:
                line_offset_to_box = line_offset
        else:
            line_offset_to_box = line_offset

        y_offset = line_offset
        return y_offset, line_offset_to_box

    @staticmethod
    def _get_box_structs(box_plotter, box_names, labels, ymaxes):
        box_structs = [
            {
                'box': box_names[i],
                'label': labels[i],
                'x': Annotator._get_box_x_position(box_plotter, box_names[i]),
                'box_data': Annotator._get_box_data(box_plotter, box_names[i]),
                'ymax': ymaxes[box_names[i]]
            } for i in range(len(box_names))]

        # Sort the box data structures by position along the x axis
        box_structs = sorted(box_structs, key=lambda a: a['x'])

        # Add the index position in the list of boxes along the x axis
        box_structs = [dict(box_struct, xi=i)
                       for i, box_struct in enumerate(box_structs)]

        return box_structs

    @staticmethod
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

    def validate_test_short_name(self):
        if self.show_test_name:
            if self.test_short_name is None:
                self.test_short_name = (self.test
                                        if isinstance(self.test, str)
                                        else self.test.short_name)
        else:
            self.test_short_name = ""

    def has_type0_comparisons_correction(self):
        return self.comparisons_correction is not None \
               and not self.comparisons_correction.type

    def _update_pvalue_thresholds(self):
        self._pvalue_thresholds = self._get_pvalue_thresholds(
            self._pvalue_thresholds)


def _print_result_line(box_structs, formatted_output):
    label1 = box_structs[0]['label']
    label2 = box_structs[1]['label']
    print(f"{label1} v.s. {label2}: {formatted_output}")


def sort_pvalue_thresholds(pvalue_thresholds):
    return sorted(pvalue_thresholds,
                  key=lambda threshold_notation: threshold_notation[0])
