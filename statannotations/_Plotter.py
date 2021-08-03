import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import lines
from matplotlib.collections import PathCollection
from matplotlib.patches import Rectangle

from statannotations._GroupsPositions import _GroupsPositions
from statannotations.utils import check_not_none, check_order_in_data, \
    check_pairs_in_data, render_collection, check_is_in, remove_null

IMPLEMENTED_PLOTTERS = {
    'seaborn': ['barplot', 'boxplot', 'stripplot', 'swarmplot', 'violinplot']
}


class _Plotter:
    def __init__(self, ax, pairs, data=None, x=None, y=None, hue=None,
                 order=None, hue_order=None, verbose=False, **plot_params):
        self.ax = ax
        self._fig = plt.gcf()
        check_not_none("pairs", pairs)
        group_coord = y if plot_params.get("orient") == "h" else x
        check_order_in_data(data, group_coord, order)
        check_pairs_in_data(pairs, data, group_coord, hue, hue_order)
        self.pairs = pairs
        self._struct_pairs = None
        self.verbose = verbose
        self.orient = plot_params.get("orient", "v")

    def get_transform_func(self, kind: str):
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
            return self.ax.transAxes.inverted()

        transform = {'v': self.ax.get_xaxis_transform,
                     'h': self.ax.get_yaxis_transform}[self.orient]

        data_to_ax = \
            self.ax.transData + transform().inverted()
        if kind == 'data_to_ax':
            return data_to_ax

        elif kind == 'ax_to_data':
            return data_to_ax.inverted()

        else:
            return (data_to_ax, data_to_ax.inverted(),
                    self.ax.transAxes.inverted())

    @property
    def fig(self):
        return self._fig

    @property
    def struct_pairs(self):
        return self._struct_pairs


class _SeabornPlotter(_Plotter):
    def __init__(self, ax, pairs, plot='boxplot', data=None, x=None,
                 y=None, hue=None, order=None, hue_order=None, verbose=False,
                 **plot_params):

        _Plotter.__init__(self, ax, pairs, data, x, y, hue, order, hue_order,
                          verbose, **plot_params)

        self.check_plot_is_implemented(plot)
        self.plot = plot
        self.plotter = self._get_plotter(plot, x, y, hue, data, order,
                                         hue_order, **plot_params)

        self.group_names, self.labels = self._get_group_names_and_labels()
        self.groups_positions = _GroupsPositions(self.plotter,
                                                 self.group_names)
        self.reordering = None
        self.value_maxes = self._generate_value_maxes()

        self.structs = self._get_structs()
        self.pairs = pairs
        self._struct_pairs = self._get_group_struct_pairs()

        self._value_stack_arr = np.array(
            [[struct['group_coord'], struct['value_max'], 0]
             for struct in self.structs]
        ).T

        self.reordering = None
        self._sort_group_struct_pairs()

    @property
    def value_stack_arr(self):
        return self._value_stack_arr

    # noinspection PyProtectedMember
    def _get_plotter(self, plot, x, y, hue, data, order, hue_order,
                     **plot_params):
        dodge = plot_params.pop("dodge", None)
        self.fix_and_warn(dodge, hue, plot)

        if plot == 'boxplot':
            plotter = sns.categorical._BoxPlotter(

                x, y, hue, data, order, hue_order,
                orient=plot_params.get("orient"),
                width=plot_params.get("width", 0.8),
                dodge=True,
                fliersize=plot_params.get("fliersize", 5),
                linewidth=plot_params.get("linewidth"),
                saturation=.75, color=None, palette=None)

        elif plot == 'swarmplot':
            plotter = sns.categorical._SwarmPlotter(
                x, y, hue, data, order, hue_order,
                orient=plot_params.get("orient"),
                dodge=True, color=None, palette=None)

        elif plot == 'stripplot':
            plotter = sns.categorical._StripPlotter(
                x, y, hue, data, order, hue_order,
                jitter=plot_params.get("jitter", True),
                orient=plot_params.get("orient"),
                dodge=True, color=None, palette=None)

        elif plot == 'barplot':
            plotter = sns.categorical._BarPlotter(
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

        elif plot == "violinplot":
            plotter = sns.categorical._ViolinPlotter(
                x, y, hue, data, order, hue_order,
                bw=plot_params.get("bw", "scott"),
                cut=plot_params.get("cut", 2),
                scale=plot_params.get("scale", "area"),
                scale_hue=plot_params.get("scale_hue", True),
                gridsize=plot_params.get("gridsize", 100),
                width=plot_params.get("width", 0.8),
                inner=plot_params.get("inner", None),
                split=plot_params.get("split", False),
                dodge=True, orient=plot_params.get("orient"),
                linewidth=plot_params.get("linewidth"), color=None,
                palette=None, saturation=.75)

        else:
            raise NotImplementedError(
                f"Only {render_collection(IMPLEMENTED_PLOTTERS)} are "
                f"supported.")

        return plotter

    def _get_group_names_and_labels(self):

        if self.plotter.plot_hues is None:
            group_names = self.plotter.group_names
            labels = group_names

        else:
            labels = []
            group_names = []
            for group_name, hue_name in \
                    itertools.product(self.plotter.group_names,
                                      self.plotter.hue_names):
                group_names.append((group_name, hue_name))
                labels.append(f'{group_name}_{hue_name}')

        return group_names, labels

    def _generate_value_maxes(self):
        """
        given plotter and the names of two categorical variables,
        returns highest y point drawn between those two variables before
        annotations.

        The highest data point is often not the highest item drawn
        (eg, error bars and/or bar charts).
        """

        value_maxes = {name: 0 for name in self.group_names}

        data_to_ax = self.get_transform_func('data_to_ax')

        if self.plot == 'violinplot':
            value_maxes = self._get_value_maxes_violin(value_maxes, data_to_ax)

        else:
            for child in self.ax.get_children():

                group_name, value_pos = self._get_value_pos(child, data_to_ax)

                if (value_pos is not None
                        and value_pos > value_maxes[group_name]):
                    value_maxes[group_name] = value_pos

        return value_maxes

    def _get_structs(self):
        structs = [
            {
                'group': group_name,
                'label': self.labels[b_idx],
                'group_coord': (self.groups_positions
                                .get_group_axis_position(group_name)),
                'group_data': self._get_group_data(group_name),
                'value_max': self.value_maxes[group_name]
            }
            for b_idx, group_name in enumerate(self.group_names)]

        # Sort the group data structures by position along the groups axis
        structs = sorted(structs, key=lambda struct: struct['group_coord'])

        # Add the index position in the list of groups along the groups axis
        structs = [dict(struct, group_i=i)
                   for i, struct in enumerate(structs)]

        return structs

    def _get_group_struct_pairs(self):
        group_structs_dic = {struct['group']: struct
                             for struct in self.structs}

        group_struct_pairs = []

        for i_group_pair, (group1, group2) in enumerate(self.pairs):
            group_struct1 = dict(group_structs_dic[group1],
                                 i_group_pair=i_group_pair)

            group_struct2 = dict(group_structs_dic[group2],
                                 i_group_pair=i_group_pair)

            if group_struct1['group_coord'] <= group_struct2['group_coord']:
                group_struct_pairs.append((group_struct1, group_struct2))

            else:
                group_struct_pairs.append((group_struct2, group_struct1))

        return group_struct_pairs

    def _get_group_data_with_hue(self, group_name):
        cat = group_name[0]
        group_data, index = self._get_group_data_from_plotter(cat)

        hue_level = group_name[1]
        hue_mask = self.plotter.plot_hues[index] == hue_level

        return group_data[hue_mask]

    def _get_group_data_from_plotter(self, cat):
        index = self.plotter.group_names.index(cat)

        return self.plotter.plot_data[index], index

    def _get_group_data_without_hue(self, group_name):
        group_data, _ = self._get_group_data_from_plotter(group_name)

        return group_data

    def _get_group_data(self, group_name):
        """
        group_name can be either a name "cat" or a tuple ("cat", "hue")

        Almost a duplicate of seaborn's code, because there is not
        direct access to the group_data in the respective Plotter class.
        """
        if self.plotter.plot_hues is None:
            data = self._get_group_data_without_hue(group_name)
        else:
            data = self._get_group_data_with_hue(group_name)

        group_data = remove_null(data)

        return group_data

    def _get_value_maxes_violin(self, value_maxes, data_to_ax):
        for group_idx, group_name in enumerate(self.plotter.group_names):
            if self.plotter.hue_names:
                for hue_idx, hue_name in enumerate(self.plotter.hue_names):
                    value_pos = max(self.plotter.support[group_idx][hue_idx])
                    value_maxes[(group_name, hue_name)] = \
                        data_to_ax.transform((0, value_pos))[1]
            else:
                value_pos = max(self.plotter.support[group_idx])
                value_maxes[group_name] = data_to_ax.transform((0,
                                                                value_pos))[1]
        return value_maxes

    def _get_value_pos(self, child, data_to_ax):
        if (type(child) == PathCollection
                and len(child.properties()['offsets'])):
            return self._get_value_pos_for_path_collection(
                child, data_to_ax)

        elif type(child) in (lines.Line2D, Rectangle):
            return self._get_value_pos_for_line2d_or_rectangle(
                child, data_to_ax)

        return None, None

    def _get_value_pos_for_path_collection(self, child, data_to_ax):
        group_coord = {"v": 0, "h": 1}[self.orient]
        value_coord = (group_coord + 1) % 2

        direction = {"v": 1, "h": -1}[self.orient]

        value_max = child.properties()['offsets'][:, value_coord].max()
        group_pos = float(np.round(np.nanmean(
            child.properties()['offsets'][:, group_coord]), 1))

        if group_pos not in self.groups_positions.axis_positions:
            if self.verbose:
                warnings.warn(
                    "Invalid x-position found. Are the same parameters passed "
                    "to seaborn and statannotations calls? or are there few "
                    "data points?")
            group_pos = self.groups_positions.find_closest(group_pos)

        group_name = self.groups_positions.axis_positions[group_pos]

        value_pos = data_to_ax.transform((0, value_max)[::direction])

        return group_name, value_pos[value_coord]

    def _get_value_pos_for_line2d_or_rectangle(self, child, data_to_ax):
        bbox = self.ax.transData.inverted().transform(
            child.get_window_extent(self.fig.canvas.get_renderer()))

        group_coord = {"v": 0, "h": 1}[self.orient]
        direction = {"v": 1, "h": -1}[self.orient]

        value_coord = (group_coord + 1) % 2

        if ((bbox[:, group_coord].max() - bbox[:, group_coord].min())
                > 1.1 * self.groups_positions.axis_units):
            return None, None

        raw_group_pos = np.round(bbox[:, group_coord].mean(), 1)
        group_pos = self.groups_positions.get_axis_pos_location(raw_group_pos)

        if group_pos not in self.groups_positions.axis_positions:
            return None, None

        group_name = self.groups_positions.axis_positions[group_pos]
        value_pos = bbox[:, value_coord].max()

        value_pos = data_to_ax.transform((0, value_pos)[::direction])
        return group_name, value_pos[value_coord]

    def _sort_group_struct_pairs(self):
        # Draw first the annotations with the shortest between-groups distance,
        # in order to reduce overlapping between annotations.
        new_group_struct_pairs = sorted(
            enumerate(self._struct_pairs),
            key=self._absolute_group_struct_pair_in_tuple_x_diff)

        self.reordering, self._struct_pairs = zip(*new_group_struct_pairs)

    @staticmethod
    def _absolute_group_struct_pair_in_tuple_x_diff(group_struct_pair):
        return abs(group_struct_pair[1][1]['group_coord']
                   - group_struct_pair[1][0]['group_coord'])

    @staticmethod
    def check_plot_is_implemented(plot, engine="seaborn"):
        if plot not in IMPLEMENTED_PLOTTERS[engine]:
            raise NotImplementedError(
                f"Only {render_collection(IMPLEMENTED_PLOTTERS[engine])} are "
                f"supported with {engine} engine.")

    @staticmethod
    def fix_and_warn(dodge, hue, plot):
        if dodge is False and hue:
            raise ValueError("`dodge` cannot be False in statannotations.")

        if plot in ("swarmplot", 'stripplot') and hue:
            if dodge is None:
                warnings.warn(
                    "Implicitly setting dodge to True as it is necessary in "
                    "statannotations. It must have been True for the seaborn "
                    "call to yield consistent results when using `hue`.")

    def get_value_lim(self):
        if self.orient == 'v':
            return self.ax.get_ylim()
        return self.ax.get_xlim()
