import numpy as np

from statannotations.utils import get_closest


class _GroupsPositions:
    def __init__(self, plotter, group_names):
        self._plotter = plotter
        self._hue_names = self._plotter.hue_names

        if self._hue_names is not None:
            nb_hues = len(self._hue_names)
            if nb_hues == 1:
                raise ValueError(
                    "Using hues with only one hue is not supported.")

            self.hue_offsets = self._plotter.hue_offsets
            self._axis_units = self.hue_offsets[1] - self.hue_offsets[0]

        self._groups_positions = {
            np.round(self.get_group_axis_position(group_name), 1): group_name
            for group_name in group_names
        }

        self._groups_positions_list = sorted(self._groups_positions.keys())

        if self._hue_names is None:
            self._axis_units = ((max(list(self._groups_positions.keys())) + 1)
                                / len(self._groups_positions))

        self._axis_ranges = {
            (pos - self._axis_units / 2,
             pos + self._axis_units / 2,
             pos): group_name
            for pos, group_name in self._groups_positions.items()}

    @property
    def axis_positions(self):
        return self._groups_positions

    @property
    def axis_units(self):
        return self._axis_units

    def get_axis_pos_location(self, pos):
        """
        Finds the x-axis location of a categorical variable
        """
        for axis_range in self._axis_ranges:
            if (pos >= axis_range[0]) & (pos <= axis_range[1]):
                return axis_range[2]

    def get_group_axis_position(self, group):
        """
        group_name can be either a name "cat" or a tuple ("cat", "hue")
        """
        if self._plotter.plot_hues is None:
            cat = group
            hue_offset = 0
        else:
            cat = group[0]
            hue_level = group[1]
            hue_offset = self._plotter.hue_offsets[
                self._plotter.hue_names.index(hue_level)]

        group_pos = self._plotter.group_names.index(cat) + hue_offset
        return group_pos

    def find_closest(self, pos):
        return get_closest(list(self._groups_positions_list), pos)
