import unittest

import seaborn as sns

# noinspection PyProtectedMember
from statannotations.statannotations import \
    _apply_comparisons_correction, Annotator


class TestParametersValidation(unittest.TestCase):
    """Test validation of parameters"""
    def setUp(self):
        self.data = [[1, 2, 3], [2, 5, 7]]
        self.ax = sns.boxplot(data=self.data)

    def test_init(self):
        self.annot = Annotator(self.ax, [(0, 1)], data=self.data)

    def test_test_name_provided(self):
        self.test_init()
        with self.assertRaisesRegex(ValueError, "test"):
            self.annot.apply_test()

    def test_box_pairs_in_x(self):
        with self.assertRaisesRegex(ValueError, "(specified in `box_pairs`)"):
            self.annot = Annotator(self.ax, [(0, 2)], data=self.data)

    def test_order_in_x(self):
        with self.assertRaisesRegex(ValueError, "(specified in `order`)"):
            self.annot = Annotator(self.ax, [(0, 2)], data=self.data,
                                   order=[0, 1, 2])

    def test_location(self):
        self.test_init()
        with self.assertRaisesRegex(ValueError, "argument `loc`"):
            self.annot.configure(loc="somewhere")

    def test_unknown_parameter(self):
        self.test_init()
        with self.assertRaisesRegex(ValueError, "Parameter `that`"):
            self.annot.configure(that="this")

    def test_format(self):
        self.test_init()

        with self.assertRaisesRegex(ValueError, "argument `text_format`"):
            self.annot.configure(text_format="that")

    def test_apply_comparisons_correction(self):
        self.assertIsNone(_apply_comparisons_correction(None, []))

    def test_correct_num_custom_annotations(self):
        self.test_init()
        with self.assertRaisesRegex(ValueError, "same length"):
            self.annot.set_custom_annotation(["One", "Two"])
