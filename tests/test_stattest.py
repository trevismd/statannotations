import unittest

import numpy as np

from statannotations.stats.StatResult import StatResult
from statannotations.stats.StatTest import StatTest
from statannotations.stats.tests import IMPLEMENTED_TESTS


class TestStatTest(unittest.TestCase):
    """Test `statannotations.stats.StatTest` class."""

    def setUp(self):
        self.x = [1, 2, 3]
        self.y = [2, 2, 2]

    @staticmethod
    def a_func(x, y):
        return 4*len(x), 0.3*(np.mean(y))

    def test_from_library_works(self):
        test = StatTest.from_library(IMPLEMENTED_TESTS[0])
        assert isinstance(test, StatTest)

    def test_from_library_not(self):
        with self.assertRaises(NotImplementedError):
            StatTest.from_library('This is not a test name')

    def test_make_test(self):
        test = StatTest(self.a_func, 'A test', 'test')
        res = test(self.x, self.y)
        self.assertIsInstance(res, StatResult)
        stat, pval = res.stat_value, res.pvalue
        self.assertEqual((stat, pval), (12, 0.6))
