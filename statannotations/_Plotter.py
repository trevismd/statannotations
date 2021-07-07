import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import lines
from matplotlib.collections import PathCollection
from matplotlib.patches import Rectangle

from statannotations._Xpositions import _XPositions
from statannotations.utils import check_not_none, check_order_in_data, \
    check_box_pairs_in_data, render_collection, check_is_in, remove_null

IMPLEMENTED_PLOTTERS = ['barplot', 'boxplot', 'stripplot', 'swarmplot',
                        'violinplot']


class _Plotter:
    def __init__(self, ax, pairs, plot='boxplot', data=None, x=None,
                 y=None, hue=None, order=None, hue_order=None, **plot_params):

        check_plot_is_implemented(plot)
        check_not_none("pairs", pairs)
        check_order_in_data(data, x, order)
        check_box_pairs_in_data(pairs, data, x, hue, hue_order)

        self.ax = ax
        self.fig = plt.gcf()

        self.plot = plot
        self.plotter = self._get_plotter(plot, x, y, hue, data, order,
                                         hue_order, **plot_params)

        self.box_names, self.labels = self._get_box_names_and_labels()
        self.xpositions = _XPositions(self.plotter, self.box_names)
        self.reordering = None
        self.ymaxes = self._generate_ymaxes()

        self.box_structs = self._get_box_structs()
        self.box_pairs = pairs
        self.box_struct_pairs = self._get_box_struct_pairs()

        self.reordering = None
        self.sort_box_struct_pairs()

        self.y_stack_arr = np.array(
            [[box_struct['x'], box_struct['ymax'], 0]
             for box_struct in self.box_structs]
        ).T

    # noinspection PyProtectedMember
    def _get_plotter(self, plot, x, y, hue, data, order, hue_order,
                     **plot_params):
        if not plot_params.pop("dodge", True):
            raise ValueError("`dodge` must be True in statannotations")

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

    def _get_box_names_and_labels(self):
        group_names = self.plotter.group_names
        hue_names = self.plotter.hue_names

        if self.plotter.plot_hues is None:
            box_names = group_names
            labels = box_names
        else:
            box_names = [(group_name, hue_name) for group_name in group_names
                         for hue_name in hue_names]
            labels = ['{}_{}'.format(group_name, hue_name)
                      for (group_name, hue_name) in box_names]

        return box_names, labels

    def _generate_ymaxes(self):
        """
        given box plotter and the names of two categorical variables,
        returns highest y point drawn between those two variables before
        annotations.

        The highest data point is often not the highest item drawn
        (eg, error bars and/or bar charts).
        """

        ymaxes = {name: 0 for name in self.box_names}

        data_to_ax = self.get_transform_func('data_to_ax')

        if self.plot == 'violinplot':
            ymaxes = self._get_ymaxes_violin(ymaxes, data_to_ax)

        else:
            for child in self.ax.get_children():

                xname, ypos = self._get_child_ypos(child, data_to_ax)

                if ypos is not None and ypos > ymaxes[xname]:
                    ymaxes[xname] = ypos

        return ymaxes

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

        data_to_ax = \
            self.ax.transData + self.ax.get_xaxis_transform().inverted()

        if kind == 'data_to_ax':
            return data_to_ax

        elif kind == 'ax_to_data':
            return data_to_ax.inverted()

        else:
            return (data_to_ax, data_to_ax.inverted(),
                    self.ax.transAxes.inverted())

    def _get_box_structs(self):
        box_structs = [
            {
                'box': box_name,
                'label': self.labels[b_idx],
                'x': self.xpositions.get_box_x_position(box_name),
                'box_data': self._get_box_data(box_name),
                'ymax': self.ymaxes[box_name]
            } for b_idx, box_name in enumerate(self.box_names)]

        # Sort the box data structures by position along the x axis
        box_structs = sorted(box_structs, key=lambda a: a['x'])

        # Add the index position in the list of boxes along the x axis
        box_structs = [dict(box_struct, xi=i)
                       for i, box_struct in enumerate(box_structs)]

        return box_structs

    def _get_box_struct_pairs(self):
        box_structs_dic = {box_struct['box']: box_struct
                           for box_struct in self.box_structs}

        box_struct_pairs = []

        for i_box_pair, (box1, box2) in enumerate(self.box_pairs):
            box_struct1 = dict(box_structs_dic[box1], i_box_pair=i_box_pair)
            box_struct2 = dict(box_structs_dic[box2], i_box_pair=i_box_pair)

            if box_struct1['x'] <= box_struct2['x']:
                box_struct_pairs.append((box_struct1, box_struct2))
            else:
                box_struct_pairs.append((box_struct2, box_struct1))

        return box_struct_pairs

    def _get_box_data_with_hue(self, box_name):
        cat = box_name[0]
        group_data, index = self._get_group_data(cat)

        hue_level = box_name[1]
        hue_mask = self.plotter.plot_hues[index] == hue_level

        return group_data[hue_mask]

    def _get_group_data(self, cat):
        index = self.plotter.group_names.index(cat)

        return self.plotter.plot_data[index], index

    def _get_box_data_without_hue(self, box_name):
        group_data, _ = self._get_group_data(box_name)

        return group_data

    def _get_box_data(self, box_name):
        """
        box_name can be either a name "cat" or a tuple ("cat", "hue")

        Almost a duplicate of seaborn's code, because there is not
        direct access to the box_data in the BoxPlotter class.
        """
        if self.plotter.plot_hues is None:
            data = self._get_box_data_without_hue(box_name)
        else:
            data = self._get_box_data_with_hue(box_name)

        box_data = remove_null(data)

        return box_data

    def _get_ymaxes_violin(self, ymaxes, data_to_ax):
        for group_idx, group_name in enumerate(self.plotter.group_names):
            if self.plotter.hue_names:
                for hue_idx, hue_name in enumerate(self.plotter.hue_names):
                    ypos = max(self.plotter.support[group_idx][hue_idx])
                    ymaxes[(group_name, hue_name)] = \
                        data_to_ax.transform((0, ypos))[1]
            else:
                ypos = max(self.plotter.support[group_idx])
                ymaxes[group_name] = data_to_ax.transform((0, ypos))[1]
        return ymaxes

    def _get_child_ypos(self, child, data_to_ax):
        if ((type(child) == PathCollection)
                and len(child.properties()['offsets'])):
            return self._get_ypos_for_path_collection(
                child, data_to_ax)

        elif (type(child) == lines.Line2D) or (type(child) == Rectangle):
            return self._get_ypos_for_line2d_or_rectangle(
                child, data_to_ax)

        return None, None

    def _get_ypos_for_path_collection(self, child, data_to_ax):
        ymax = child.properties()['offsets'][:, 1].max()
        xpos = float(np.round(np.nanmean(
            child.properties()['offsets'][:, 0]), 1))

        xname = self.xpositions.xpositions[xpos]
        ypos = data_to_ax.transform((0, ymax))[1]

        return xname, ypos

    def _get_ypos_for_line2d_or_rectangle(self, child, data_to_ax):
        box = self.ax.transData.inverted().transform(
            child.get_window_extent(self.fig.canvas.get_renderer()))

        if (box[:, 0].max() - box[:, 0].min()) > 1.1 * self.xpositions.xunits:
            return None, None
        raw_xpos = np.round(box[:, 0].mean(), 1)
        xpos = self.xpositions.get_xpos_location(raw_xpos)
        if xpos not in self.xpositions.xpositions:
            return None, None
        xname = self.xpositions.xpositions[xpos]
        ypos = box[:, 1].max()
        ypos = data_to_ax.transform((0, ypos))[1]

        return xname, ypos

    def sort_box_struct_pairs(self):
        # Draw first the annotations with the shortest between-boxes distance,
        # in order to reduce overlapping between annotations.
        new_box_struct_pairs = sorted(
            enumerate(self.box_struct_pairs),
            key=self.absolute_box_struct_pair_in_tuple_x_diff)

        self.reordering, self.box_struct_pairs = zip(*new_box_struct_pairs)

    @staticmethod
    def absolute_box_struct_pair_in_tuple_x_diff(box_struct_pair):
        return abs(box_struct_pair[1][1]['x'] - box_struct_pair[1][0]['x'])


def check_plot_is_implemented(plot):
    if plot not in IMPLEMENTED_PLOTTERS:
        raise NotImplementedError(
            f"Only {render_collection(IMPLEMENTED_PLOTTERS)} are supported.")
