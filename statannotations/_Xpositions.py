import numpy as np


class _XPositions:
    def __init__(self, plotter, group_names):
        self.plotter = plotter
        self.xpositions = {
            np.round(self.get_group_x_position(group_name), 1): group_name
            for group_name in group_names
        }

        self.xunits = \
            (max(list(self.xpositions.keys())) + 1) / len(self.xpositions)

        self.xranges = {
            (pos - self.xunits / 2, pos + self.xunits / 2, pos): group_name
            for pos, group_name in self.xpositions.items()}

    def get_xpos_location(self, pos):
        """
        Finds the x-axis location of a categorical variable
        """
        for xrange in self.xranges:
            if (pos >= xrange[0]) & (pos <= xrange[1]):
                return xrange[2]

    def get_group_x_position(self, group):
        """
        group_name can be either a name "cat" or a tuple ("cat", "hue")
        """
        if self.plotter.plot_hues is None:
            cat = group
            hue_offset = 0
        else:
            cat = group[0]
            hue_level = group[1]
            hue_offset = self.plotter.hue_offsets[
                self.plotter.hue_names.index(hue_level)]

        group_pos = self.plotter.group_names.index(cat) + hue_offset
        return group_pos
