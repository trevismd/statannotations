import warnings
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import seaborn as sns
from matplotlib import lines
from matplotlib.font_manager import FontProperties

from statannotations.Annotation import Annotation
from statannotations.PValueFormat import PValueFormat, \
    CONFIGURABLE_PARAMETERS as PVALUE_CONFIGURABLE_PARAMETERS
from statannotations._Plotter import _Plotter, _SeabornPlotter
from statannotations.stats.ComparisonsCorrection import \
    get_validated_comparisons_correction
from statannotations.stats.StatResult import StatResult
from statannotations.stats.StatTest import StatTest
from statannotations.stats.test import apply_test, IMPLEMENTED_TESTS
from statannotations.stats.utils import check_alpha, check_pvalues, \
    check_num_comparisons, get_num_comparisons
from statannotations.utils import check_is_in, InvalidParametersError, \
    empty_dict_if_none

CONFIGURABLE_PARAMETERS = [
    'alpha',
    'color',
    'comparisons_correction',
    'line_height',
    'line_offset',
    'line_offset_to_group',
    'line_width',
    'loc',
    'pvalue_format',
    'show_test_name',
    'test',
    'test_short_name',
    'text_offset',
    'use_fixed_offset',
    'verbose',
]


_DEFAULT_VALUES = {
    "alpha": 0.05,
    "test_short_name": None,
    "annotations": None,
    "_comparisons_correction": None,
    "_loc": "inside",
    "verbose": 1,
    "_just_configured": True,
    "show_test_name": True,
    "use_fixed_offset": False,
    "line_height": 0.02,
    "_test": None,
    "text_offset": 1,
    "color": '0.2',
    "line_width": 1.5,
    "custom_annotations": None
}

ENGINE_PLOTTERS = {"seaborn": _SeabornPlotter}


class Annotator:
    """
    Optionally computes statistical test between pairs of data series, and add
    statistical annotation on top of the groups (boxes, bars...). The same
    exact arguments provided to the seaborn plotting function must be passed to
    the constructor.

    This Annotator works in one of the three following modes:
        - Add custom text annotations (`set_custom_annotations`)
        - Format pvalues and add them to the plot (`set_pvalues`)
        - Perform a statistical test and then add the results to the plot
            (`apply_test`)
    """

    def __init__(self, ax, pairs, plot='boxplot', data=None, x=None,
                 y=None, hue=None, order=None, hue_order=None,
                 engine="seaborn", verbose=True, **plot_params):
        """
        :param ax: Ax of existing plot
        :param pairs: can be of either form:
            For non-grouped plots: `[(cat1, cat2), (cat3, cat4)]`.
            For plots grouped by hue: `[
                ((cat1, hue1), (cat2, hue2)), ((cat3, hue3), (cat4, hue4))
            ]`
        :param plot: type of the plot (default 'boxplot')
        :param data: seaborn  plot's data
        :param x: seaborn plot's x
        :param y: seaborn plot's y
        :param hue: seaborn plot's hue
        :param order: seaborn plot's order
        :param hue_order: seaborn plot's hue_order
        :param engine: currently only "seaborn" is implemented
        :param verbose: verbosity flag
        :param plot_params: Other parameters for plotter engine
        """
        self.pairs = pairs
        self.ax = ax

        if self.ax is None:
            self._plotter = None
        else:
            self._plotter = self._get_plotter(engine, ax, pairs, plot, data,
                                              x, y, hue, order, hue_order,
                                              verbose=verbose, **plot_params)

        self._test = None
        self.perform_stat_test = None
        self.test_short_name = None
        self._pvalue_format = PValueFormat()
        self._alpha = _DEFAULT_VALUES["alpha"]

        self.annotations = None
        self._comparisons_correction = None

        self._loc = "inside"
        self._verbose = 1
        self._just_configured = True
        self.show_test_name = True
        self.use_fixed_offset = False
        self.line_offset_to_group = None
        self.line_offset = None
        self.line_height = _DEFAULT_VALUES["line_height"]
        self.text_offset = 1
        self.text_offset_impact_above = 0
        self.color = '0.2'
        self.line_width = 1.5
        self.value_offset = None
        self.custom_annotations = None

    @staticmethod
    def get_empty_annotator():
        """
        This instance will have to be initialized with `new_plot()` before
        being used. This behavior can be useful to create an Annotator before
        using it in a `FacetGrid` mapping.
        """
        return Annotator(None, None)

    def new_plot(self, ax, pairs=None, plot='boxplot', data=None, x=None,
                 y=None, hue=None, order=None, hue_order=None,
                 engine: str = "seaborn", **plot_params):
        self.ax = ax

        if pairs is None:
            pairs = self.pairs

        self._plotter: _Plotter = self._get_plotter(
            engine, ax, pairs, plot, data, x, y, hue, order, hue_order,
            **plot_params)

        self.line_offset = None
        self.line_offset_to_group = None
        self.perform_stat_test = None

        return self

    @property
    def orient(self):
        return self._plotter.orient

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        self._verbose = verbose
        self._plotter.verbose = verbose

    @property
    def _value_stack_arr(self):
        return self._plotter.value_stack_arr

    @property
    def fig(self):
        return self._plotter.fig

    @property
    def _struct_pairs(self):
        return self._plotter.struct_pairs

    def _get_output(self):
        return self.ax, self.annotations

    def reset_configuration(self):

        self._reset_default_values()
        self._pvalue_format = PValueFormat()

        return self

    def annotate(self, line_offset=None, line_offset_to_group=None):
        """Add configured annotations to the plot."""
        self._check_has_plotter()

        self._maybe_warn_about_configuration()

        self._update_value_for_loc()

        ann_list = []
        orig_value_lim = self._plotter.get_value_lim()

        offset_func = self.get_offset_func(self.loc)
        self.value_offset, self.line_offset_to_group = offset_func(
            line_offset, line_offset_to_group)

        if self._verbose:
            self.print_pvalue_legend()

        ax_to_data = self._plotter.get_transform_func('ax_to_data')

        self.validate_test_short_name()

        for annotation in self.annotations:
            self._annotate_pair(annotation,
                                ax_to_data=ax_to_data,
                                ann_list=ann_list,
                                orig_value_lim=orig_value_lim)

        # reset transformation
        y_stack_max = max(self._value_stack_arr[1, :])
        ax_to_data = self._plotter.get_transform_func('ax_to_data')
        value_lims = (
            ([(0, 0), (0, max(1.04 * y_stack_max, 1))]
             if self.loc == 'inside'
             else [(0, 0), (0, 1)])
            if self.orient == 'v'
            else
            ([(0, 0), (max(1.04 * y_stack_max, 1), 0)]
             if self.loc == 'inside'
             else [(0, 0), (1, 0)])
        )
        set_lims = self.ax.set_ylim if self.orient == 'v' else self.ax.set_xlim
        transformed = ax_to_data.transform(value_lims)
        set_lims(transformed[:, 1 if self.orient == 'v' else 0])

        return self._get_output()

    def apply_and_annotate(self):
        """Applies a configured statistical test and annotates the plot"""

        self.apply_test()
        return self.annotate()

    def configure(self, **parameters):
        """
        * `alpha`: Acceptable type 1 error for statistical tests, default 0.05
        * `color`
        * `comparisons_correction`: Method for multiple comparisons correction.
            One of `statsmodels` `multipletests` methods (w/ default FWER), or
            a `ComparisonsCorrection` instance.
        * `line_height`: in axes fraction coordinates
        * `line_offset`
        * `line_offset_to_group`
        * `line_width`
        * `loc`
        * `pvalue_format`: list of lists, or tuples. Default values are:
            * For "star" text_format: `[[1e-4, "****"], [1e-3, "***"],
                                        [1e-2, "**"], [0.05, "*"],
                                        [1, "ns"]]`.
            * For "simple" text_format : `[[1e-5, "1e-5"], [1e-4, "1e-4"],
                                           [1e-3, "0.001"], [1e-2, "0.01"],
                                           [5e-2, "0.05"]]`
        * `show_test_name`: Set to False to not show the (short) name of
            test
        * `test`
        * `text_offset`: in points
        * `test_short_name`
        * `use_fixed_offset`
        * `verbose`
        """
        self._check_has_plotter()

        if parameters.get("pvalue_format") is None:
            parameters["pvalue_format"] = {
                item: value
                for item, value in parameters.items()
                if item in PVALUE_CONFIGURABLE_PARAMETERS
            }

        unmatched_parameters = parameters.keys() - set(CONFIGURABLE_PARAMETERS)
        unmatched_parameters -= parameters["pvalue_format"].keys()

        if unmatched_parameters:
            raise InvalidParametersError(unmatched_parameters)

        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        self._activate_configured_warning()

        self._warn_alpha_thresholds_if_necessary(parameters)
        return self

    def apply_test(self, num_comparisons='auto', **stats_params):
        """
        :param stats_params: Parameters for statistical test functions.

        :param num_comparisons: Override number of comparisons otherwise
            calculated with number of pairs
        """
        self._check_has_plotter()
        self._check_test_pvalues_perform()

        if stats_params is None:
            stats_params = dict()

        self.perform_stat_test = True

        self.annotations = self._get_results(num_comparisons=num_comparisons,
                                             **stats_params)
        self._deactivate_configured_warning()

        return self

    def set_pvalues_and_annotate(self, pvalues, num_comparisons='auto'):
        self.set_pvalues(pvalues, num_comparisons)
        return self.annotate()

    def set_pvalues(self, pvalues, num_comparisons='auto'):
        """
        :param pvalues: list or array of p-values for each pair comparison.
        :param num_comparisons: Override number of comparisons otherwise
            calculated with number of pairs
        """
        self._check_has_plotter()

        self.perform_stat_test = False

        self._check_pvalues_no_perform(pvalues)
        self._check_test_no_perform()
        check_num_comparisons(num_comparisons)

        self.annotations = self._get_results(
            num_comparisons=get_num_comparisons(pvalues, num_comparisons),
            pvalues=pvalues, test_short_name=self.test_short_name)

        self._deactivate_configured_warning()

        return self

    def set_custom_annotations(self, text_annot_custom):
        """
        :param text_annot_custom: List of strings to annotate for each
            `pair`
        """
        self._check_has_plotter()

        self._check_correct_number_custom_annotations(text_annot_custom)
        self.annotations = [Annotation(struct, text) for
                            struct, text in zip(self._struct_pairs,
                                                text_annot_custom)]
        self.show_test_name = False
        self._deactivate_configured_warning()
        return self

    def annotate_custom_annotations(self, text_annot_custom):
        """
        :param text_annot_custom: List of strings to annotate for each
            `pair`
        """
        self.set_custom_annotations(text_annot_custom)
        return self.annotate()

    def get_configuration(self):
        configuration = {key: getattr(self, key)
                         for key in CONFIGURABLE_PARAMETERS}
        configuration["pvalue_format"] = self.pvalue_format.get_configuration()
        return configuration

    def get_annotations_text(self):
        if self.annotations is not None:
            return [self.annotations[new_idx].text
                    for new_idx in self._reordering]
        return None

    @property
    def _reordering(self):
        return self._plotter.reordering

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
        self._check_has_plotter()

        self._comparisons_correction = get_validated_comparisons_correction(
            comparisons_correction)

    @property
    def loc(self):
        return self._loc

    @loc.setter
    def loc(self, loc):
        check_is_in(loc, ['inside', 'outside'], label='argument `loc`')
        self._loc = loc

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
    def pvalue_format(self):
        return self._pvalue_format

    @pvalue_format.setter
    def pvalue_format(self, pvalue_format_config):
        self._pvalue_format.config(**pvalue_format_config)

    @property
    def _should_warn_about_configuration(self):
        return self._just_configured

    @classmethod
    def get_offset_func(cls, position):
        return {"inside":  cls._get_offsets_inside,
                "outside": cls._get_offsets_outside}[position]

    def _activate_configured_warning(self):
        self._just_configured = True

    def _deactivate_configured_warning(self):
        self._just_configured = False

    def print_pvalue_legend(self):
        if not self.custom_annotations:
            self._pvalue_format.print_legend_if_used()

    def _get_results(self, num_comparisons, pvalues=None,
                     **stats_params) -> List[Annotation]:

        test_result_list = []

        if num_comparisons == "auto":
            num_comparisons = len(self._struct_pairs)

        for group_struct1, group_struct2 in self._struct_pairs:
            group1 = group_struct1['group']
            group2 = group_struct2['group']

            if self.perform_stat_test:
                result = self._get_stat_result_from_test(
                    group_struct1, group_struct2, num_comparisons,
                    **stats_params)
            else:
                result = self._get_custom_results(group_struct1, pvalues)

            result.group1 = group1
            result.group2 = group2

            test_result_list.append(Annotation((group_struct1, group_struct2),
                                               result,
                                               formatter=self.pvalue_format))

        # Perform other types of correction methods for multiple testing
        self._apply_comparisons_correction(test_result_list)

        return test_result_list

    def _check_has_plotter(self):
        if self._plotter is None:
            raise RuntimeError(
                "Not plotter is defined. If `get_empty_annotator` was used, "
                "`new_plot` must be called to provide data")

    def _annotate_pair(self, annotation, ax_to_data, ann_list, orig_value_lim):

        if self._verbose >= 1:
            annotation.print_labels_and_content()

        group_coord_1 = annotation.structs[0]['group_coord']
        group_coord_2 = annotation.structs[1]['group_coord']
        group_i1 = annotation.structs[0]['group_i']
        group_i2 = annotation.structs[1]['group_i']

        # Find y maximum for all the y_stacks *in between* group1 and group2
        i_value_max_in_range_g1_g2 = group_i1 + np.nanargmax(
            self._value_stack_arr[1,
                                  np.where((group_coord_1
                                            <= self._value_stack_arr[0, :])
                                           & (self._value_stack_arr[0, :]
                                              <= group_coord_2))])

        value = self._get_value_for_pair(i_value_max_in_range_g1_g2)

        # Determine lines in axes coordinates
        ax_line_group = [group_coord_1, group_coord_1,
                         group_coord_2, group_coord_2]
        ax_line_value = [value, value + self.line_height,
                         value + self.line_height, value]

        lists = ((ax_line_group, ax_line_value) if self.orient == 'v'
                 else (ax_line_value, ax_line_group))

        points = [ax_to_data.transform((x, y))
                  for x, y
                  in zip(*lists)]

        line_x, line_y = zip(*points)
        self._plot_line(line_x, line_y)

        xy_params = self._get_xy_params(group_coord_1, group_coord_2, line_x,
                                        line_y)

        ann = self.ax.annotate(
            annotation.text, textcoords='offset points',
            xycoords='data', ha='center', va='bottom',
            fontsize=self._pvalue_format.fontsize, clip_on=False,
            annotation_clip=False, **xy_params)

        if annotation.text is not None:
            ann_list.append(ann)
            plt.draw()
            set_lim = {'v': 'set_ylim',
                       'h': 'set_xlim'}[self.orient]

            getattr(self.ax, set_lim)(orig_value_lim)

            value_top_annot = self._annotate_pair_text(ann, value)
        else:
            value_top_annot = value + self.line_height

        # Fill the highest value position of the annotation into value_stack
        # for all positions in the range group_coord_1 to group_coord_2
        self._value_stack_arr[
            1, (group_coord_1 <= self._value_stack_arr[0, :])
            & (self._value_stack_arr[0, :] <= group_coord_2)] = value_top_annot

        # Increment the counter of annotations in the value_stack array
        self._value_stack_arr[2, group_i1:group_i2 + 1] += 1

    def _update_value_for_loc(self):
        if self._loc == 'outside':
            self._value_stack_arr[1, :] = 1

    def _check_test_pvalues_perform(self):
        if self.test is None:
            raise ValueError("If `perform_stat_test` is True, "
                             "`test` must be specified.")

        if self.test not in IMPLEMENTED_TESTS and \
                not isinstance(self.test, StatTest):
            raise ValueError("test value should be a StatTest instance "
                             "or one of the following strings: {}."
                             .format(', '.join(IMPLEMENTED_TESTS)))

    def _check_pvalues_no_perform(self, pvalues):
        if pvalues is None:
            raise ValueError("If `perform_stat_test` is False, "
                             "custom `pvalues` must be specified.")
        check_pvalues(pvalues)
        if len(pvalues) != len(self.pairs):
            raise ValueError("`pvalues` should be of the same length as "
                             "`pairs`.")

    def _check_test_no_perform(self):
        if self.test is not None:
            raise ValueError("If `perform_stat_test` is False, "
                             "`test` must be None.")

    def _check_correct_number_custom_annotations(self, text_annot_custom):
        if text_annot_custom is None:
            return

        if len(text_annot_custom) != len(self._struct_pairs):
            raise ValueError(
                "`text_annot_custom` should be of same length as `pairs`.")

    def _apply_comparisons_correction(self, annotations):
        if self.comparisons_correction is None:
            return
        else:
            self.comparisons_correction.apply(
                [annotation.data for annotation in annotations])

    def _get_stat_result_from_test(self, group_struct1, group_struct2,
                                   num_comparisons,
                                   **stats_params) -> StatResult:

        result = apply_test(
            group_struct1['group_data'],
            group_struct2['group_data'],
            self.test,
            comparisons_correction=self.comparisons_correction,
            num_comparisons=num_comparisons,
            alpha=self.alpha,
            **stats_params
        )

        return result

    def _get_custom_results(self, group_struct1, pvalues) -> StatResult:
        pvalue = pvalues[group_struct1['i_group_pair']]

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
    def _get_offsets_inside(line_offset, line_offset_to_group):
        if line_offset is None:
            line_offset = 0.05
            if line_offset_to_group is None:
                line_offset_to_group = 0.06
        else:
            if line_offset_to_group is None:
                line_offset_to_group = 0.06
        return line_offset, line_offset_to_group

    @staticmethod
    def _get_offsets_outside(line_offset, line_offset_to_group):
        if line_offset is None:
            line_offset = 0.03
            if line_offset_to_group is None:
                line_offset_to_group = line_offset
        else:
            line_offset_to_group = line_offset

        y_offset = line_offset
        return y_offset, line_offset_to_group

    def validate_test_short_name(self):
        if self.test_short_name is not None:
            return

        if self.show_test_name and self.pvalue_format.text_format != "star":
            if self.test:
                self.test_short_name = (self.test
                                        if isinstance(self.test, str)
                                        else self.test.short_name)

        if self.test_short_name is None:
            self.test_short_name = ""

    def has_type0_comparisons_correction(self):
        return self.comparisons_correction is not None \
               and not self.comparisons_correction.type

    def _plot_line(self, line_x, line_y):
        if self.loc == 'inside':
            self.ax.plot(line_x, line_y, lw=self.line_width, c=self.color)
        else:
            line = lines.Line2D(line_x, line_y, lw=self.line_width,
                                c=self.color, transform=self.ax.transData)
            line.set_clip_on(False)
            self.ax.add_line(line)

    def _annotate_pair_text(self, ann, value):

        value_top_annot = None
        got_mpl_error = False

        if not self.use_fixed_offset:
            try:
                bbox = ann.get_window_extent()
                pix_to_ax = self._plotter.get_transform_func('pix_to_ax')
                bbox_ax = bbox.transformed(pix_to_ax)
                value_coord_max = {'v': 'ymax', 'h': 'xmax'}[self.orient]
                value_top_annot = getattr(bbox_ax, value_coord_max)

            except RuntimeError:
                got_mpl_error = True

        if self.use_fixed_offset or got_mpl_error:
            if self._verbose >= 1:
                print("Warning: cannot get the text bounding box. Falling "
                      "back to a fixed y offset. Layout may be not "
                      "optimal.")

            fontsize_points = FontProperties(
                size='medium').get_size_in_points()

            direction = {'h': -1, 'v': 1}[self.orient]
            x, y = [0, fontsize_points + self.text_offset][::direction]
            offset_trans = mtransforms.offset_copy(trans=self.ax.transAxes,
                                                   fig=self.fig,
                                                   units='points', x=x, y=y)

            value_top_display = offset_trans.transform(
                (value + self.line_height, value + self.line_height))

            value_coord = {'h': 0, 'v': 1}[self.orient]

            value_top_annot = (self.ax.transAxes.inverted()
                               .transform(value_top_display)[value_coord])

        self.text_offset_impact_above = (
                value_top_annot - value - self.value_offset - self.line_height)

        return value_top_annot

    def _reset_default_values(self):
        for attribute, default_value in _DEFAULT_VALUES.items():
            setattr(self, attribute, default_value)

    def _get_value_for_pair(self, i_ymax_in_range_x1_x2):

        ymax_in_range_x1_x2 = self._value_stack_arr[1, i_ymax_in_range_x1_x2]

        # Choose the best offset depending on whether there is an annotation
        # below at the x position in the range [x1, x2] where the stack is the
        # highest
        if self._value_stack_arr[2, i_ymax_in_range_x1_x2] == 0:
            # there is only a group below
            offset = self.line_offset_to_group
        else:
            # there is an annotation below
            offset = self.value_offset + self.text_offset_impact_above

        return ymax_in_range_x1_x2 + offset

    @staticmethod
    def _warn_alpha_thresholds_if_necessary(parameters):
        if parameters.get("alpha"):
            pvalue_format = parameters.get("pvalue_format")
            if (pvalue_format is None
                    or pvalue_format.get("pvalue_thresholds") is None):
                warnings.warn("Changing alpha without updating "
                              "pvalue_thresholds can result in inconsistent "
                              "plotting results")

    @staticmethod
    def _get_plotter(engine, *args, **kwargs):
        engine_plotter = ENGINE_PLOTTERS.get(engine)
        if engine_plotter is None:
            raise NotImplementedError(f"{engine} engine not implemented.")
        return engine_plotter(*args, **kwargs)

    def _get_xy_params_horizontal(self, group_coord_1, group_coord_2,
                                  line_x: np.ndarray):
        return {
            'xy': (line_x[2], np.mean([group_coord_1, group_coord_2])),
            'xytext': (self.text_offset, 0),
            'rotation':  270,
            'rotation_mode': 'anchor'
        }

    def _get_xy_params_vertical(self, group_coord_1, group_coord_2,
                                line_y: np.ndarray):
        return {
            'xy': (np.mean([group_coord_1, group_coord_2]), line_y[2]),
            'xytext': (0, self.text_offset),
        }

    def _get_xy_params(self, group_coord_1, group_coord_2, line_x: np.ndarray,
                       line_y: np.ndarray):
        if self.orient == 'h':
            return self._get_xy_params_horizontal(group_coord_1, group_coord_2,
                                                  line_x)

        return self._get_xy_params_vertical(group_coord_1, group_coord_2,
                                            line_y)

    def _maybe_warn_about_configuration(self):
        if self._should_warn_about_configuration:
            warnings.warn("Annotator was reconfigured without applying the "
                          "test (again) which will probably lead to "
                          "unexpected results")

    def plot_and_annotate_facets(
            self, plot: str, plot_params: dict, configuration: dict,
            annotation_func: str, *args, annotation_params: dict = None,
            ax_op_before: List[Union[str, Optional[list],
                                     Optional[dict]]] = None,
            ax_op_after: List[Union[str, Optional[list],
                                    Optional[dict]]] = None,
            annotate_params: dict = None, **kwargs):
        """
        Plots using seaborn and annotates in a single call, to be used within
        a `FacetGrid`.
        First, initialize the Annotator with `Annotator(None, pairs)` to define
        the pairs, then map this function onto the `FacetGrid`.

        :param plot: seaborn plotting function to call
        :param plot_params: parameters for plotting function call
        :param configuration: parameters for Annotator.configure
        :param annotation_func: name of annotation function to be called, from:
            * 'set_custom_annotations'
            * 'set_pvalues'
            * 'apply_test'
        :param annotation_params: parameters for the annotation function
        :param ax_op_before: list of [func_name, args, kwargs] to apply on `ax`
            before annotating
        :param ax_op_after: list of [func_name, args, kwargs] to apply on `ax`
            after annotating
        :param annotate_params: parameters for `Annotator.annotate`
        :param args: additional parameters for the seaborn function
        :param kwargs: additional parameters for the seaborn function
        """
        annotate_params = empty_dict_if_none(annotate_params)
        annotation_params = empty_dict_if_none(annotation_params)

        ax = getattr(sns, plot)(*args, **plot_params, **kwargs)

        _apply_ax_operations(ax, ax_op_before)

        self.new_plot(ax, plot=plot, **plot_params, data=kwargs['data'])
        self.configure(**configuration)
        getattr(self, annotation_func)(**annotation_params)
        self.annotate(**annotate_params)

        _apply_ax_operations(ax, ax_op_after)
        return self._get_output()

    @staticmethod
    def plot_and_annotate(
            plot: str, pairs: list, plot_params: dict, configuration: dict,
            annotation_func: str, annotation_params: dict = None,
            ax_op_before: List[Union[str, Optional[list],
                                     Optional[dict]]] = None,
            ax_op_after: List[Union[str, Optional[list],
                                    Optional[dict]]] = None,
            annotate_params: dict = None):
        """
        Plots using seaborn and annotates in a single call.

        :param plot: seaborn plotting function to call
        :param pairs: pairs to compare (see Annotator)
        :param plot_params: parameters for plotting function call
        :param configuration: parameters for Annotator.configure
        :param annotation_func: name of annotation function to be called, from:
            * 'set_custom_annotations'
            * 'set_pvalues'
            * 'apply_test'
        :param annotation_params: parameters for the annotation function
        :param ax_op_before: list of [func_name, args, kwargs] to apply on `ax`
            before annotating
        :param ax_op_after: list of [func_name, args, kwargs] to apply on `ax`
            after annotating
        :param annotate_params: parameters for `Annotator.annotate`
        """
        annotate_params = empty_dict_if_none(annotate_params)
        annotation_params = empty_dict_if_none(annotation_params)

        ax = getattr(sns, plot)(**plot_params)

        _apply_ax_operations(ax, ax_op_before)
        annotator = Annotator(ax, pairs, plot=plot, **plot_params)
        annotator.configure(**configuration)
        getattr(annotator, annotation_func)(**annotation_params)
        annotator.annotate(**annotate_params)

        _apply_ax_operations(ax, ax_op_after)
        return (*annotator._get_output(), annotator)


def _apply_ax_operations(ax, operations):
    if operations is None:
        return
    for operation in operations:
        _ensure_ax_operation_format(operation)
        getattr(ax, operation[0])(*operation[1],
                                  **empty_dict_if_none(operation[2]))


def _ensure_ax_operation_format(op):
    if len(op) != 3:
        raise ValueError("Please provide a string, a list and a "
                         "dictionary in order to use an ax operation")
    if not isinstance(op[0], str):
        raise ValueError("The first element of an ax operation is the "
                         "name of the method passed as a string")

    if not isinstance(op[1], list):
        raise ValueError("The second element of an ax operation is the "
                         "arguments for the ax method passed as a list")

    if not isinstance(op[2], dict) and op[2] is not None:
        raise ValueError("The third element of an ax operation is the "
                         "keyword arguments for the ax method passed as a "
                         "dictionary")
