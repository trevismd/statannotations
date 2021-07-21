import unittest

import pandas as pd
import seaborn as sns

from statannotations.Annotator import Annotator
from statannotations.stats.ComparisonsCorrection import ComparisonsCorrection


# noinspection DuplicatedCode
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
            self.ax, pairs=[(("a", "blue"), ("a", "red")),
                            (("b", "blue"), ("b", "red")),
                            (("a", "blue"), ("b", "blue"))],
            **plotting)
        self.pvalues = [0.03, 0.04, 0.9]

    def test_ns_without_correction_star(self):
        annotations = self.annotator._get_results("auto", pvalues=self.pvalues)
        self.assertEqual(["*", "*", "ns"],
                         [annotation.text for annotation in annotations])

    def test_signif_with_type1_correction_star(self):
        bh = ComparisonsCorrection("BH")
        self.annotator.configure(comparisons_correction=bh)
        self.annotator.set_pvalues(self.pvalues)
        self.assertEqual(["* (ns)", "* (ns)", "ns"],
                         self.annotator.get_annotations_text())

    def test_signif_with_type1_correction_star_incorrect_num_comparisons(self):
        bh = ComparisonsCorrection("BH")
        self.annotator.configure(comparisons_correction=bh)
        with self.assertRaisesRegex(ValueError, "positive"):
            self.annotator.set_pvalues(self.pvalues, num_comparisons=0)

    def test_signif_with_type1_correction_star_abnormal_num_comparisons(self):
        bh = ComparisonsCorrection("BH")
        self.annotator.configure(comparisons_correction=bh)
        with self.assertWarnsRegex(UserWarning, "Manually-specified"):
            self.annotator.set_pvalues(self.pvalues, num_comparisons=1)

    def test_signif_with_type0_correction_star(self):
        bonferroni = ComparisonsCorrection("bonferroni")
        self.annotator.configure(comparisons_correction=bonferroni)

        self.annotator.set_pvalues(self.pvalues)

        self.assertEqual(["ns", "ns", "ns"],
                         self.annotator.get_annotations_text())

    def test_signif_with_type1_correction_simple(self):
        bh = ComparisonsCorrection("BH")
        self.annotator.configure(comparisons_correction=bh,
                                 pvalue_format={'text_format': 'simple'})
        self.annotator.set_pvalues(self.pvalues)

        expected = ["p ≤ 0.05 (ns)", "p ≤ 0.05 (ns)", "p = 0.90"]

        self.assertEqual(expected, self.annotator.get_annotations_text())

    def test_signif_with_type0_correction_simple(self):
        bonferroni = ComparisonsCorrection("bonferroni")
        self.annotator.configure(comparisons_correction=bonferroni,
                                 pvalue_format={'text_format': 'simple'})

        self.annotator.set_pvalues(self.pvalues)

        expected = ["p = 0.09", "p = 0.12", "p = 1.00"]

        self.assertEqual(expected, self.annotator.get_annotations_text())

    def test_reapply_annotations(self):
        pass
