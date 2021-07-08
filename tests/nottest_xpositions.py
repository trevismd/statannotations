import unittest

import pandas as pd
import seaborn as sns

from statannotations._Xpositions import _XPositions
from statannotations._Plotter import _Plotter


class TestXPositions(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame.from_dict({
            1: {"x": "a", "y": 15, "color": "blue"},
             2: {"x": "a", "y": 16, "color": "blue"},
             3: {"x": "b", "y": 17, "color": "blue"},
             4: {"x": "b", "y": 18, "color": "blue"},
         }).T
        self.plot_params = {
            "data": self.df,
            "x": "x",
            "y": "y",
        }

    def test_one_hue_raise(self):
        self.plot_params = {**self.plot_params, 'order': None, 'hue': 'color',
                            'hue_order': None}
        ax = sns.boxplot(**self.plot_params)
        with self.assertRaises(ValueError):
            _Plotter._get_plotter(ax, 'boxplot', **self.plot_params)
