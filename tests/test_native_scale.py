import unittest

import matplotlib.pylab as plt
import seaborn as sns

from statannotations.Annotator import Annotator
from statannotations._Plotter import IMPLEMENTED_PLOTTERS

seaborn_v0_13 = unittest.skipIf(
    tuple(map(int, sns.__version__.split(".")[:2])) < (0, 13),
    reason="at least seaborn-0.13 required",
)

plot_types = IMPLEMENTED_PLOTTERS["seaborn"]

class TestNativeScalePlotter(unittest.TestCase):
    def setUp(self) -> None:
        df = sns.load_dataset("tips")
        self.df = df.query("size in [2, 3, 5]")

        self.params_df = {
            "data": self.df,
            "x": "size",
            "y": "total_bill",
            "hue_order": ["Male", "Female"],
            "hue": "sex",
        }

        self.pairs = [
            ((3, "Male"), (5, "Male")),
            ((2, "Male"), (2, "Female")),
        ]

    def tearDown(self) -> None:
        plt.clf()
        return super().tearDown()

    @seaborn_v0_13
    def test_native_scale(self):
        native_scale = True
        for plotter in plot_types:
            with self.subTest(plotter=plotter):
                plotting = {
                    **self.params_df,
                    "dodge": True,
                    "native_scale": native_scale,
                }
                ax = getattr(sns, plotter)(**plotting)
                self.annot = Annotator(ax, plot=plotter, pairs=self.pairs, **plotting)

                positions = self.annot._plotter.groups_positions._data

                assert positions.group.tolist() == [
                    (2, "Male"),
                    (2, "Female"),
                    (3, "Male"),
                    (3, "Female"),
                    (5, "Male"),
                    (5, "Female"),
                ]
                if native_scale:
                    expected_positions = [1.8, 2.2, 2.8, 3.2, 4.8, 5.2]
                else:
                    expected_positions = [-0.2, 0.2, 0.8, 1.2, 1.8, 2.2]
                assert positions.pos.tolist() == expected_positions

    @seaborn_v0_13
    def test_no_native_scale(self):
        native_scale = False
        for plotter in plot_types:
            plotting = {
                **self.params_df,
                "dodge": True,
                "native_scale": native_scale,
            }
            ax = getattr(sns, plotter)(**plotting)
            self.annot = Annotator(ax, plot=plotter, pairs=self.pairs, **plotting)

            positions = self.annot._plotter.groups_positions._data

            assert positions.group.tolist() == [
                (2, "Male"),
                (2, "Female"),
                (3, "Male"),
                (3, "Female"),
                (5, "Male"),
                (5, "Female"),
            ]
            if native_scale:
                expected_positions = [1.8, 2.2, 2.8, 3.2, 4.8, 5.2]
            else:
                expected_positions = [-0.2, 0.2, 0.8, 1.2, 1.8, 2.2]
            assert positions.pos.tolist() == expected_positions

    @seaborn_v0_13
    def test_gap(self):
        plotting = {
            **self.params_df,
            "dodge": True,
            "gap": 0.2,
        }
        ax = sns.boxplot(**plotting)
        self.annot = Annotator(ax, plot="boxplot", pairs=self.pairs, **plotting)

        positions = self.annot._plotter.groups_positions._data

        assert positions.group.tolist() == [
            (2, "Male"),
            (2, "Female"),
            (3, "Male"),
            (3, "Female"),
            (5, "Male"),
            (5, "Female"),
        ]
        expected_positions = [-0.2, 0.2, 0.8, 1.2, 1.8, 2.2]
        assert positions.pos.tolist() == expected_positions
