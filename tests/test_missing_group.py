import unittest

import pandas as pd
import seaborn as sns

from statannotations.Annotator import Annotator


# noinspection DuplicatedCode
class TestMissingGroup(unittest.TestCase):
    """Test validation of parameters"""

    def setUp(self):
        self.data = pd.DataFrame(
            [
                [1, 1.0, 0],
                [1, 2.0, 1],
                [1, 3.0, 2],
                [2, 4.0, 0],
                [2, 5.0, 1],
            ],
            columns=["x", "y", "hue"],
        )
        self.ax = sns.boxplot(data=self.data)

        self.pairs = [
            ((1, 0), (2, 0)),
            ((1, 0), (1, 2)),
        ]
        self.params_df = {
            "data": self.data,
            "x": "x",
            "y": "y",
            "hue": "hue",
        }

        self.params_arrays = {
            "data": None,
            "x": self.data["x"],
            "y": self.data["y"],
            "hue": self.data["hue"],
        }

    def tearDown(self) -> None:
        sns.mpl.pyplot.clf()
        return super().tearDown()

    def test_init_df(self):
        self.ax = sns.boxplot(**self.params_df)
        self.annot = Annotator(self.ax, pairs=self.pairs, **self.params_df)

    def test_init_array(self):
        self.ax = sns.boxplot(**self.params_arrays)
        self.annot = Annotator(self.ax, pairs=self.pairs, **self.params_arrays)
