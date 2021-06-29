import unittest

import pandas as pd
import seaborn as sns

from statannotations.Annotator import Annotator
from statannotations.stats.ComparisonsCorrection import ComparisonsCorrection


class Test(unittest.TestCase):
    """Test that the annotations match the pvalues and format."""

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
        plotting = {
            "data": self.df,
            "x": "x",
            "y": "y",
            "hue": 'color'
        }
        self.ax = sns.boxplot(**plotting)
        self.annotator = Annotator(
            self.ax, box_pairs=[(("a", "blue"), ("a", "red")),
                                (("b", "blue"), ("b", "red")),
                                (("a", "blue"), ("b", "blue"))],
            **plotting)
        self.pvalues = [0.03, 0.04, 0.9]

    def test_ns_without_correction_star(self):
        results = self.annotator._get_results(pvalues=self.pvalues)
        self.assertEqual(["*", "*", "ns"],
                         [self.annotator._get_text(result)
                          for result in results])

    def test_signif_with_type1_correctio_star(self):
        bh = ComparisonsCorrection("BH")
        self.annotator.configure(comparisons_correction=bh)
        self.annotator.set_pvalues(self.pvalues)
        self.assertEqual(["* (ns)", "* (ns)", "ns"],
                         [self.annotator._get_text(result)
                          for result in self.annotator.annotations])

    def test_signif_with_type0_correction_star(self):
        bonferroni = ComparisonsCorrection("bonferroni")
        self.annotator.configure(comparisons_correction=bonferroni)

        self.annotator.set_pvalues(self.pvalues)

        self.assertEqual(["ns", "ns", "ns"],
                         [self.annotator._get_text(result)
                          for result in self.annotator.annotations])

    def test_ns_without_correction_simple(self):
        self.annotator.configure(text_format="simple")
        results = self.annotator._get_results(pvalues=self.pvalues)
        self.assertEqual(["p ≤ 0.05", "p ≤ 0.05", "p = 0.90"],
                         [self.annotator._get_text(result)
                          for result in results])

    def test_signif_with_type1_correction_simple(self):
        bh = ComparisonsCorrection("BH")
        self.annotator.configure(comparisons_correction=bh,
                                 text_format="simple")
        self.annotator.set_pvalues(self.pvalues)
        reordering = self.annotator.reordering

        expected = [["p ≤ 0.05 (ns)", "p ≤ 0.05 (ns)", "p = 0.90"][index]
                    for index in reordering]

        self.assertEqual(expected,
                         [self.annotator._get_text(result)
                          for result in self.annotator.annotations])

    def test_signif_with_type0_correction_simple(self):
        bonferroni = ComparisonsCorrection("bonferroni")
        self.annotator.configure(comparisons_correction=bonferroni,
                                 text_format="simple")

        self.annotator.set_pvalues(self.pvalues)
        reordering = self.annotator.reordering

        expected = [["p = 0.09", "p = 0.12", "p = 1.00"][index]
                    for index in reordering]

        self.assertEqual(expected,
                         [self.annotator._get_text(result)
                          for result in self.annotator.annotations])
