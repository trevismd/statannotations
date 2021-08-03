import io
import re
import unittest.mock

import pandas as pd
import seaborn as sns

from statannotations.Annotator import Annotator
from statannotations.PValueFormat import PValueFormat
from statannotations.utils import DEFAULT, InvalidParametersError


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
            self.ax, pairs=[(("a", "blue"), ("a", "red")),
                            (("b", "blue"), ("b", "red")),
                            (("a", "blue"), ("b", "blue"))], verbose=False,
            **plotting)
        self.pvalues = [0.03, 0.04, 0.9]

    def test_format_simple(self):
        self.annotator.configure(pvalue_format={"text_format": "simple"})
        annotations = self.annotator._get_results("auto", pvalues=self.pvalues)
        self.assertEqual(["p ≤ 0.05", "p ≤ 0.05", "p = 0.90"],
                         [annotation.text for annotation in annotations])
    
    def test_format_simple_in_annotator(self):
        self.annotator.configure(text_format="simple")
        annotations = self.annotator._get_results("auto", pvalues=self.pvalues)
        self.assertEqual(["p ≤ 0.05", "p ≤ 0.05", "p = 0.90"],
                         [annotation.text for annotation in annotations])

    def test_wrong_parameter(self):
        with self.assertRaisesRegex(
                InvalidParametersError, re.escape('parameter(s) "that"')):
            self.annotator.configure(pvalue_format={"that": "whatever"})

    def test_format_string(self):
        self.annotator.configure(text_format="simple",
                                 pvalue_format_string="{:.3f}")
        self.assertEqual("{:.3f}",
                         self.annotator.pvalue_format.pvalue_format_string)
        annotations = self.annotator._get_results("auto", pvalues=self.pvalues)
        self.assertEqual(["p ≤ 0.05", "p ≤ 0.05", "p = 0.900"],
                         [annotation.text for annotation in annotations])

    def test_format_string_default(self):
        self.annotator.configure(text_format="simple",
                                 pvalue_format_string="{:.3f}")
        self.annotator.pvalue_format.config(pvalue_format_string=DEFAULT)

        annotations = self.annotator._get_results("auto", pvalues=self.pvalues)
        self.assertEqual(["p ≤ 0.05", "p ≤ 0.05", "p = 0.90"],
                         [annotation.text for annotation in annotations])

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def assert_print_pvalue(self, pvalue_format, expected_output, mock_stdout):
        pvalue_format.print_legend_if_used()
        self.assertEqual(expected_output, mock_stdout.getvalue())

    def test_print_pvalue_default(self):
        pvalue_format = PValueFormat()
        self.assert_print_pvalue(pvalue_format,
                                 "p-value annotation legend:\n"
                                 "      ns: p <= 1.00e+00\n"
                                 "       *: 1.00e-02 < p <= 5.00e-02\n"
                                 "      **: 1.00e-03 < p <= 1.00e-02\n"
                                 "     ***: 1.00e-04 < p <= 1.00e-03\n"
                                 "    ****: p <= 1.00e-04\n\n")

    def test_print_pvalue_star(self):
        pvalue_format = PValueFormat()
        pvalue_format.config(text_format="star")
        self.assert_print_pvalue(pvalue_format,
                                 "p-value annotation legend:\n"
                                 "      ns: p <= 1.00e+00\n"
                                 "       *: 1.00e-02 < p <= 5.00e-02\n"
                                 "      **: 1.00e-03 < p <= 1.00e-02\n"
                                 "     ***: 1.00e-04 < p <= 1.00e-03\n"
                                 "    ****: p <= 1.00e-04\n\n")

    def test_print_pvalue_other(self):
        pvalue_format = PValueFormat()
        pvalue_format.config(text_format="simple")
        self.assert_print_pvalue(pvalue_format, "")

    def test_get_configuration(self):
        pvalue_format = PValueFormat()
        self.assertDictEqual(pvalue_format.get_configuration(),
                             {'fontsize': 'medium',
                              'pvalue_format_string': '{:.3e}',
                              'simple_format_string': '{:.2f}',
                              'text_format': 'star',
                              'pvalue_thresholds': [
                                  [1e-4, "****"],
                                  [1e-3, "***"],
                                  [1e-2, "**"],
                                  [0.05, "*"],
                                  [1, "ns"]]
                              }
                             )

    def test_config_pvalue_thresholds(self):
        pvalue_format = PValueFormat()
        pvalue_format.config(pvalue_thresholds=[
            [0.001, "<= 0.001"], [0.05, "<= 0.05"], [1, 'ns']
        ])
        self.assert_print_pvalue(pvalue_format,
                                 "p-value annotation legend:\n"
                                 "      ns: p <= 1.00e+00\n"
                                 " <= 0.05: 1.00e-03 < p <= 5.00e-02\n"
                                 "<= 0.001: p <= 1.00e-03\n\n")
