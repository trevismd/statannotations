import re
import unittest

import pandas as pd
import seaborn as sns

from statannotations.Annotator import Annotator


from statannotations.utils import DEFAULT


# noinspection DuplicatedCode
class Test(unittest.TestCase):
    """Test that the annotations match the pvalues and format."""

    def setUp(self) -> None:
        self.df = pd.DataFrame.from_dict(
            {1: {"x": "a", "y": 15, "color": "blue"},
             2: {"x": "a", "y": 16, "color": "blue"},
             3: {"x": "b", "y": 17, "color": "blue"},
             4: {"x": "b", "y": 18, "color": "blue"},
             5: {"x": "a", "y": 15, "color": "red"},
             6: {"x": "a", "y": 16, "color": "red"},
             7: {"x": "b", "y": 17, "color": "red"},
             8: {"x": "b", "y": 18, "color": "red"}
             }).T
        plotting = {
            "data": self.df,
            "x": "x",
            "y": "y",
            "hue": "color"
        }
        self.ax = sns.boxplot(**plotting)
        self.annotator = Annotator(
            self.ax, box_pairs=[(("a", "blue"), ("a", "red")),
                                (("b", "blue"), ("b", "red")),
                                (("a", "blue"), ("b", "blue"))],
            **plotting)
        self.pvalues = [0.03, 0.04, 0.9]

    def test_format_simple(self):
        self.annotator.configure(pvalue_format={"text_format": "simple"})
        results = self.annotator._get_results("auto", pvalues=self.pvalues)
        self.assertEqual(["p ≤ 0.05", "p ≤ 0.05", "p = 0.90"],
                         [self.annotator._get_text(result)
                          for result in results])
    
    def test_format_simple_in_annotator(self):
        self.annotator.configure(text_format="simple")
        results = self.annotator._get_results("auto", pvalues=self.pvalues)
        self.assertEqual(["p ≤ 0.05", "p ≤ 0.05", "p = 0.90"],
                         [self.annotator._get_text(result)
                          for result in results])

    def test_wrong_parameter(self):
        with self.assertRaisesRegex(
                ValueError, re.escape("parameter(s) `that`")):
            self.annotator.configure(pvalue_format={"that": "whatever"})

    def test_format_string(self):
        self.annotator.configure(text_format="simple",
                                 pvalue_format_string="{:.3f}")
        self.assertEqual("{:.3f}", self.annotator.pvalue_format.pvalue_format_string)
        results = self.annotator._get_results("auto", pvalues=self.pvalues)
        self.assertEqual(["p ≤ 0.05", "p ≤ 0.05", "p = 0.900"],
                         [self.annotator._get_text(result)
                          for result in results])

    def test_format_string_default(self):
        self.annotator.configure(text_format="simple",
                                 pvalue_format_string="{:.3f}")
        self.annotator.pvalue_format.config(pvalue_format_string=DEFAULT)

        results = self.annotator._get_results("auto", pvalues=self.pvalues)
        self.assertEqual(["p ≤ 0.05", "p ≤ 0.05", "p = 0.90"],
                         [self.annotator._get_text(result)
                          for result in results])
