import unittest

import numpy as np

from statannotations.stats.test import apply_test
from statannotations.stats.StatTest import StatTest


class TestStatsTest(unittest.TestCase):

    def setUp(self):
        self.x = [1, 2, 3]
        self.y = [2, 2, 2]

    def test_apply_test_from_library(self):
        apply_test(self.x, self.y, "Mann-Whitney")

    def test_apply_test_with_correction_from_library(self):
        apply_test(self.x, self.y, "Mann-Whitney", "BH")

    def test_apply_test_without_test(self):
        res = apply_test(self.x, self.y, None, "BH")
        self.assertIs(np.nan, res.pvalue)

    def test_apply_test_with_test(self):
        test = StatTest.from_library("Mann-Whitney")
        self.assertAlmostEqual(
            0.7962534147376392,
            apply_test(self.x, self.y, test).pvalue
        )

    def test_apply_test_with_correction(self):
        self.assertAlmostEqual(
            1.0,
            apply_test(self.x, self.y, "Mann-Whitney", "bonferroni",
                       num_comparisons=2).pvalue
        )
