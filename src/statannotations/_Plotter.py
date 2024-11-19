from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.patches import Rectangle

from statannotations._GroupsPositions import _GroupsPositions
from statannotations.utils import (
    check_not_none,
    check_order_in_data,
    check_pairs_in_data,
    render_collection,
    check_is_in,
    check_redundant_hue,
)
from .compat import get_plotter

if TYPE_CHECKING:  # pragma: no cover
    from typing import Literal

    from .compat import TupleGroup, Struct

logger = logging.getLogger(__name__)

IMPLEMENTED_PLOTTERS = {
    'seaborn': ['barplot', 'boxplot', 'stripplot', 'swarmplot', 'violinplot']
}

class _Plotter:
    orient: Literal['v', 'h']

    def __init__(self, ax, pairs, data=None, x=None, y=None, hue=None,
                 order=None, hue_order=None, verbose=False, **plot_params):
        self.ax = ax
        self._fig = plt.gcf()
        check_not_none('pairs', pairs)

        orient = plot_params.get('orient', 'v')
        if orient in ('y', 'h'):
            self.orient = 'h'
        elif orient in ('x', 'v'):
            self.orient = 'v'
        else:  # pragma: no cover
            logger.debug(f"Fallback to 'v', `orient` should be one of 'h' or 'v', got: {orient}")
            self.orient = 'v'

        group_coord = y if self.orient == 'h' else x
        self.is_redundant_hue = check_redundant_hue(data, group_coord, hue, hue_order)
        if self.is_redundant_hue:
            hue = None
            hue_order = None

        check_order_in_data(data, group_coord, order)
        check_pairs_in_data(pairs, data, group_coord, hue, hue_order)
        self.pairs = pairs
        self._struct_pairs = None
        self.verbose = verbose

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

        check_is_in(
            kind,
            ['data_to_ax', 'ax_to_data', 'pix_to_ax', 'all'],
            label='kind',
        )

        if kind == 'pix_to_ax':
            return self.ax.transAxes.inverted()

        transform = {
            'v': self.ax.get_xaxis_transform,
            'h': self.ax.get_yaxis_transform,
        }[self.orient]

        data_to_ax = self.ax.transData + transform().inverted()
        if kind == 'data_to_ax':
            return data_to_ax

        elif kind == 'ax_to_data':
            return data_to_ax.inverted()

        else:  # pragma: no cover
            return (data_to_ax, data_to_ax.inverted(),
                    self.ax.transAxes.inverted())

    @property
    def fig(self):
        return self._fig

    @property
    def struct_pairs(self):
        return self._struct_pairs


class _SeabornPlotter(_Plotter):
    structs: list[Struct]

    def __init__(
        self, ax, pairs, plot='boxplot', data=None, x=None,
        y=None, hue=None, order=None, hue_order=None, verbose=False,
        **plot_params,
    ):
        super().__init__(
            ax, pairs, data, x, y, hue, order, hue_order, verbose, **plot_params
        )
        if self.is_redundant_hue:
            hue = None
            hue_order = None

        self.check_plot_is_implemented(plot)
        self.plot = plot
        self.plotter = self._get_plotter(
            plot,
            x=x,
            y=y,
            hue=hue,
            data=data,
            order=order,
            hue_order=hue_order,
            is_redundant_hue=self.is_redundant_hue,
            **plot_params,
        )

        self.groups_positions = _GroupsPositions(
            self.plotter.group_names,
            self.plotter.hue_names,
            width=self.plotter.width,
            gap=self.plotter.gap,
            dodge=self.plotter.dodge,
            use_native_offsets=self.plotter.use_native_offsets,
        )
        self.tuple_group_names = self.groups_positions.tuple_group_names
        self.reordering = None
        self.value_maxes = self._generate_value_maxes()


        self.structs = self._get_structs()
        self.pairs = self.plotter.parse_pairs(pairs, self.structs)
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
    def _get_plotter(self, plot, **kwargs):
        return get_plotter(plot, **kwargs)

    def _get_structs(self) -> list[Struct]:
        structs: list[Struct] = [
            {
                'group': group_name,
                'label': label,
                'group_coord': position,
                'group_data': self.plotter.get_group_data(group_name),
                'value_max': float(self.value_maxes[group_name]),
            }
            for group_name, label, position in self.groups_positions.iter_groups()
        ]

        # Sort the group data structures by position along the groups axis
        structs = sorted(structs, key=lambda struct: struct['group_coord'])

        # Add the index position in the list of groups along the groups axis
        structs = [{**struct, 'group_i': i} for i, struct in enumerate(structs)]

        return structs

    def _get_group_struct_pairs(self):
        group_structs_dic: dict[TupleGroup, Struct] = {
            struct['group']: struct for struct in self.structs
        }

        group_struct_pairs = []

        for i_group_pair, (group1, group2) in enumerate(self.pairs):
            group_struct1 = dict(group_structs_dic[group1],
                                 i_group_pair=i_group_pair)

            group_struct2 = dict(group_structs_dic[group2],
                                 i_group_pair=i_group_pair)

            keep_order = group_struct1['group_coord'] <= group_struct2['group_coord']
            if keep_order:
                group_struct_pairs.append((group_struct1, group_struct2))
            else:
                group_struct_pairs.append((group_struct2, group_struct1))

        return group_struct_pairs

    def _generate_value_maxes(self):
        """
        given plotter and the names of two categorical variables,
        returns highest y point drawn between those two variables before
        annotations.

        The highest data point is often not the highest item drawn
        (eg, error bars and/or bar charts).
        """

        value_maxes = {name: 0 for name in self.tuple_group_names}

        data_to_ax = self.get_transform_func('data_to_ax')

        if self.plot == 'violinplot' and self.plotter.has_violin_support:
            return self.plotter._populate_value_maxes_violin(value_maxes, data_to_ax)

        for child in self.ax.get_children():
            group_name, value_pos = self._get_value_pos(child, data_to_ax)
            if value_pos is None or group_name is None:
                continue
            if value_pos > value_maxes[group_name]:
                value_maxes[group_name] = value_pos

        return value_maxes

    def _get_value_pos(self, child, data_to_ax):
        if (
            isinstance(child, PathCollection)
            and len(child.properties()['offsets'])
        ):
            return self._get_value_pos_for_path_collection(
                child, data_to_ax)

        elif isinstance(child, PolyCollection):  # pragma: no cover
            # Should be for violinplot body but not working
            return None, None

        elif isinstance(child, (lines.Line2D, Rectangle)):
            return self._get_value_pos_for_line2d_or_rectangle(
                child, data_to_ax)

        return None, None

    def _get_value_pos_for_path_collection(self, child: PathCollection, data_to_ax):
        group_coord = {'v': 0, 'h': 1}[self.orient]
        value_coord = (group_coord + 1) % 2

        direction = {'v': 1, 'h': -1}[self.orient]

        offsets = child.get_offsets()
        # remove nans
        offsets = offsets[np.all(np.isfinite(offsets), axis=1), :]
        if len(offsets) == 0:  # pragma: no cover
            logger.debug("skip the empty or all-NaNs PathCollection")
            return None, None

        value_max = offsets[:, value_coord].max()
        group_pos = float(np.round(np.mean(offsets[:, group_coord]), 1))

        # Set strict=True to skip the legend artists
        group_name = self.groups_positions.find_group_at_pos(
            group_pos,
            verbose=self.verbose,
            strict=True,
        )

        value_pos = data_to_ax.transform((0, value_max)[::direction])

        return group_name, value_pos[value_coord]

    def _get_value_pos_for_line2d_or_rectangle(self, child, data_to_ax):
        bbox = self.ax.transData.inverted().transform(
            child.get_window_extent(self.fig.canvas.get_renderer()))

        # Get the group_name from the coordinates of the group
        group_coord = {'v': 0, 'h': 1}[self.orient]
        rect_min = bbox[:, group_coord].min()
        rect_max = bbox[:, group_coord].max()
        rect_width = rect_max - rect_min
        rect_center = (rect_max + rect_min) / 2

        if not self.groups_positions.compatible_width(rect_width):
            logger.debug(f"rectangle width is larger than the typical group artist: {rect_width}")
            return None, None

        group_name = self.groups_positions.find_group_at_pos(
            rect_center,
            verbose=False,
        )

        # Get the value from the coordinate of the values
        value_coord = (group_coord + 1) % 2
        value_pos = bbox[:, value_coord].max()
        direction = {'v': 1, 'h': -1}[self.orient]

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
    def check_plot_is_implemented(plot, engine='seaborn'):
        if plot not in IMPLEMENTED_PLOTTERS[engine]:
            msg = (
                f'Only {render_collection(IMPLEMENTED_PLOTTERS[engine])} are '
                f'supported with {engine} engine.'
            )
            raise NotImplementedError(msg)

    def get_value_lim(self):
        if self.orient == 'v':
            return self.ax.get_ylim()
        return self.ax.get_xlim()

    def set_value_lim(self, value) -> None:
        if self.orient == 'v':
            self.ax.set_ylim(value)
        else:
            self.ax.set_xlim(value)
