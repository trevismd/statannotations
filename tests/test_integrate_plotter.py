import unittest

import pandas as pd
import seaborn as sns

from statannotations.Annotator import Annotator


# noinspection DuplicatedCode
class Test(unittest.TestCase):
    """Test the Annotator's functionality (depending on _Plotter)."""

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
        self.plotting = {
            "data": self.df,
            "x": "x",
            "y": "y",
            "hue": 'color',
        }
        self.df.y = self.df.y.astype(float)

    def test_dodge_false_raises(self):
        ax = sns.barplot(dodge=False, **self.plotting)
        with self.assertRaisesRegex(ValueError, "dodge"):
            self.annotator = Annotator(
                ax, dodge=False, plot="barplot",
                pairs=[(("a", "blue"), ("a", "red")),
                       (("b", "blue"), ("b", "red")),
                       (("a", "blue"), ("b", "blue"))],
                **self.plotting)

    def test_wrong_plotter_engine(self):
        ax = sns.barplot(**self.plotting)
        with self.assertRaisesRegex(NotImplementedError, "plotly"):
            self.annotator = Annotator(
                ax, plot="barplot", engine="plotly",
                pairs=[(("a", "blue"), ("a", "red")),
                       (("b", "blue"), ("b", "red")),
                       (("a", "blue"), ("b", "blue"))],
                **self.plotting)

    def test_orient_horizontal(self):
        plotting = {**self.plotting, 'orient': 'h',
                    'x': 'y', 'y': 'x', 'dodge': True}
        ax = sns.stripplot(**plotting)
        self.annotator = Annotator(
            ax, plot="stripplot",
            pairs=[(("a", "blue"), ("a", "red")),
                   (("b", "blue"), ("b", "red")),
                   (("a", "blue"), ("b", "blue"))],
            **plotting)
        self.annotator.configure(test="Mann-Whitney")
        self.annotator.apply_and_annotate()

    def test_fixed_offset(self):
        ax = sns.barplot(**self.plotting)
        self.annotator = Annotator(
            ax, plot="barplot",
            pairs=[(("a", "blue"), ("a", "red")),
                   (("b", "blue"), ("b", "red")),
                   (("a", "blue"), ("b", "blue"))],
            **self.plotting)
        self.annotator.configure(test="Mann-Whitney", use_fixed_offset=True)
        self.annotator.apply_and_annotate()
