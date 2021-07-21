import numpy as np

from statannotations.utils import get_closest


class _XPositions:
    def __init__(self, plotter, group_names):
        self._plotter = plotter
        self._hue_names = self._plotter.hue_names

        if self._hue_names is not None:
            nb_hues = len(self._hue_names)
            if nb_hues == 1:
                raise ValueError(
                    "Using hues with only one hue is not supported.")

            self.hue_offsets = self._plotter.hue_offsets
            self._xunits = self.hue_offsets[1] - self.hue_offsets[0]

        self._xpositions = {
            np.round(self.get_group_x_position(group_name), 1): group_name
            for group_name in group_names
        }

        self._xpositions_list = sorted(self._xpositions.keys())

        if self._hue_names is None:
            self._xunits = ((max(list(self._xpositions.keys())) + 1)
                            / len(self._xpositions))

        self._xranges = {
            (pos - self._xunits / 2, pos + self._xunits / 2, pos): group_name
            for pos, group_name in self._xpositions.items()}

    @property
    def xpositions(self):
        return self._xpositions

    @property
    def xunits(self):
        return self._xunits

    def get_xpos_location(self, pos):
        """
        Finds the x-axis location of a categorical variable
        """
        for xrange in self._xranges:
            if (pos >= xrange[0]) & (pos <= xrange[1]):
                return xrange[2]

    def get_group_x_position(self, group):
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

    def find_closest(self, xpos):
        return get_closest(list(self._xpositions_list), xpos)
