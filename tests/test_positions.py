import unittest

import seaborn as sns
import matplotlib.pylab as plt

from statannotations.Annotator import Annotator
from statannotations._Plotter import IMPLEMENTED_PLOTTERS


class TestPositionPlotter(unittest.TestCase):
    def setUp(self) -> None:
        self.df = sns.load_dataset("tips")

        self.params_df = {
            "data": self.df,
            "x": "day",
            "y": "total_bill",
            "order": ["Sun", "Thur", "Fri", "Sat"],
            "hue_order": ["Male", "Female"],
            "hue": "sex",
        }

        self.pairs = [
            (("Thur", "Male"), ("Thur", "Female")),
            (("Sun", "Male"), ("Sun", "Female")),
            (("Thur", "Male"), ("Fri", "Female")),
            (("Fri", "Male"), ("Fri", "Female")),
            (("Sun", "Male"), ("Fri", "Female")),
        ]

    def tearDown(self) -> None:
        plt.clf()
        return super().tearDown()

    def test_seaborn_violinplot(self):
        plotting = {**self.params_df, "dodge": True}
        ax = sns.violinplot(**plotting)
        self.annot = Annotator(ax, plot='violinplot', pairs=self.pairs, **plotting)
        self.annot.configure(test='Mann-Whitney', text_format='star', loc='inside')
        self.annot.apply_and_annotate()

        value_maxes = self.annot._plotter.value_maxes
        assert ("Sun", "Male") in value_maxes
        assert value_maxes[("Sun", "Male")] > 0.80

    def test_seaborn_all_plots(self):
        for plotter in IMPLEMENTED_PLOTTERS["seaborn"]:
            plotting = {**self.params_df, "dodge": True}
            ax = getattr(sns, plotter)(**plotting)
            self.annot = Annotator(ax, plot=plotter, pairs=self.pairs, **plotting)

            positions = self.annot._plotter.groups_positions._data

            assert positions.group.tolist() == [
                ("Sun", "Male"),
                ("Sun", "Female"),
                ("Thur", "Male"),
                ("Thur", "Female"),
                ("Fri", "Male"),
                ("Fri", "Female"),
                ("Sat", "Male"),
                ("Sat", "Female"),
            ]
            assert positions.pos.tolist() == [-0.2, 0.2, 0.8, 1.2, 1.8, 2.2, 2.8, 3.2]
