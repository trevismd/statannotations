import unittest
import pandas as pd

from statannotations._Plotter import _SeabornPlotter, IMPLEMENTED_PLOTTERS
import seaborn as sns


class TestPlotterArrays(unittest.TestCase):
    def setUp(self) -> None:
        # noinspection DuplicatedCode
        self.df = pd.DataFrame.from_dict(
            {1: {'x': "a", 'y': 15, 'color': 'blue'},
             2: {'x': "a", 'y': 16, 'color': 'blue'},
             3: {'x': "b", 'y': 17, 'color': 'blue'},
             4: {'x': "b", 'y': 18, 'color': 'blue'},
             5: {'x': "a", 'y': 15, 'color': 'red'},
             6: {'x': "a", 'y': 16, 'color': 'red'},
             7: {'x': "b", 'y': 17, 'color': 'red'},
             8: {'x': "b", 'y': 18, 'color': 'red'}
             }).T
        self.df["y"] = self.df["y"].astype(int)

        self.plotting = {
            "data": None,
            "x": self.df["x"],
            "y": self.df["y"],
            "hue": self.df["color"],
        }
        self.pairs = [(("a", "blue"), ("a", "red")),
                      (("b", "blue"), ("b", "red")),
                      (("a", "blue"), ("b", "blue"))]

    def test_seaborn_plots(self):
        for plotter in IMPLEMENTED_PLOTTERS["seaborn"]:
            if plotter in ("stripplot", "swarmplot"):
                plotting = {**self.plotting, "dodge": True}
            else:
                plotting = self.plotting
            ax = getattr(sns, plotter)(**plotting)
            _SeabornPlotter(ax, self.pairs, plot=plotter, verbose=False,
                            **plotting)
