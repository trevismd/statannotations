import unittest

import seaborn as sns

from statannotations import add_stat_annotation


class TestParametersValidation(unittest.TestCase):
    """Test validation of parameters"""
    def setUp(self):
        self.data = [[1, 2, 3], [2, 5, 7]]
        self.ax = sns.boxplot(data=self.data)

    def test_box_pairs_provided(self):
        with self.assertRaisesRegex(ValueError, "box_pairs"):
            add_stat_annotation(self.ax, data=self.data)

        with self.assertRaisesRegex(ValueError, "box_pairs"):
            add_stat_annotation(self.ax, data=self.data, test="t-test_ind")

    def test_test_name_provided(self):
        with self.assertRaisesRegex(ValueError, "test"):
            add_stat_annotation(self.ax, data=self.data, box_pairs=[(0, 1)])

    def test_box_pairs_in_x(self):
        with self.assertRaisesRegex(ValueError, "(specified in `box_pairs`)"):
            add_stat_annotation(self.ax, data=self.data, box_pairs=[(0, 2)])

    def test_order_in_x(self):
        with self.assertRaisesRegex(ValueError, "(specified in `order`)"):
            add_stat_annotation(self.ax, data=self.data, box_pairs=[(0, 1)],
                                order=[0, 1, 2])

    def test_location(self):
        with self.assertRaisesRegex(ValueError, "argument `loc`"):
            add_stat_annotation(self.ax, data=self.data, box_pairs=[(0, 1)],
                                test="t-test_ind", loc="somewhere")

    def test_format(self):
        with self.assertRaisesRegex(ValueError, "argument `text_format`"):
            add_stat_annotation(self.ax, data=self.data, box_pairs=[(0, 1)],
                                test="t-test_ind", text_format="that")
