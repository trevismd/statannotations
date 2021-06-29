import re
import unittest

import pandas as pd
import seaborn as sns

from statannotations.Annotator import Annotator, DEFAULT


class TestAnnotator(unittest.TestCase):
    """Test validation of parameters"""

    def setUp(self):
        self.data = [[1, 2, 3], [2, 5, 7]]
        self.ax = sns.boxplot(data=self.data)
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

    def test_init(self):
        self.annot = Annotator(self.ax, [(0, 1)], data=self.data)

    def test_init_barplot(self):
        ax = sns.barplot(data=self.data)
        self.annot = Annotator(ax, [(0, 1)], plot="barplot", data=self.data)

    def test_test_name_provided(self):
        self.test_init()
        with self.assertRaisesRegex(ValueError, "test"):
            self.annot.apply_test()

    def test_unmatched_x_in_box_pairs_without_hue(self):
        with self.assertRaisesRegex(ValueError, "(specified in `box_pairs`)"):
            self.annot = Annotator(self.ax, [(0, 2)], data=self.data)

    def test_order_in_x(self):
        with self.assertRaisesRegex(ValueError, "(specified in `order`)"):
            self.annot = Annotator(self.ax, [(0, 2)], data=self.data,
                                   order=[0, 1, 2])

    def test_working_hue_orders(self):
        self.annot = Annotator(self.ax, [(("a", "blue"), ("b", "blue"))],
                               data=self.df, x="x", y="y",
                               order=["a", "b"], hue='color',
                               hue_order=['red', 'blue'])

    def test_unmatched_hue_in_hue_order(self):
        with self.assertRaisesRegex(ValueError, "(specified in `hue_order`)"):
            self.annot = Annotator(self.ax, [(("a", "blue"), ("b", "blue"))],
                                   data=self.df, x="x", y="y",
                                   order=["a", "b"], hue='color',
                                   hue_order=['red', 'yellow'])

    def test_unmatched_hue_in_box_pairs(self):
        with self.assertRaisesRegex(ValueError, "(specified in `box_pairs`)"):
            self.annot = Annotator(self.ax, [(("a", "yellow"), ("b", "blue"))],
                                   data=self.df, x="x", y="y",
                                   order=["a", "b"], hue='color',
                                   hue_order=['red', 'blue'])

    def test_unmatched_x_in_box_pairs_with_hue(self):
        with self.assertRaisesRegex(ValueError, "(specified in `box_pairs`)"):
            self.annot = Annotator(self.ax, [(("c", "blue"), ("b", "blue"))],
                                   data=self.df, x="x", y="y",
                                   order=["a", "b"], hue='color',
                                   hue_order=['red', 'blue'])

    def test_location(self):
        self.test_init()
        with self.assertRaisesRegex(ValueError, "argument `loc`"):
            self.annot.configure(loc="somewhere")

    def test_unknown_parameter(self):
        self.test_init()
        with self.assertRaisesRegex(
                ValueError, re.escape("parameter(s) `that`")):
            self.annot.configure(that="this")

    def test_format(self):
        self.test_init()

        with self.assertRaisesRegex(ValueError, "argument `text_format`"):
            self.annot.configure(text_format="that")

    def test_apply_comparisons_correction(self):
        self.assertIsNone(Annotator._apply_comparisons_correction(None, []))

    def test_correct_num_custom_annotations(self):
        self.test_init()
        with self.assertRaisesRegex(ValueError, "same length"):
            self.annot.set_custom_annotation(["One", "Two"])

    def test_not_implemented_plot(self):
        with self.assertRaises(NotImplementedError):
            Annotator(self.ax, [(0, 1)], data=self.data, plot="violinplot")

    def test_reconfigure_alpha(self):
        self.test_init()
        with self.assertWarnsRegex(UserWarning, "pvalue_thresholds"):
            self.annot.configure(alpha=0.1)
        self.annot.reset_configuration()
        self.assertEqual(0.05, self.annot.alpha)

    def test_reconfigure_alpha_with_thresholds(self):
        self.test_init()
        self.annot.configure(alpha=0.1, pvalue_thresholds=DEFAULT)
        self.annot.reset_configuration()
        self.assertEqual(0.05, self.annot.alpha)
