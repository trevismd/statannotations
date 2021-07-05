import io
import unittest.mock
import numpy as np

from statannotations.stats.StatResult import StatResult
from statannotations.stats.StatTest import StatTest, wilcoxon
from statannotations.stats.test import IMPLEMENTED_TESTS


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

    def test_short_name_exists(self):
        test = StatTest.from_library("Mann-Whitney")
        self.assertEqual("M.W.W.", test.short_name)

    def test_wilcoxon_legacy_set_wilcox(self):
        test = StatTest(wilcoxon, "Wilcoxon (legacy)", "Wilcox (L)")
        res1 = test(self.x, self.y, zero_method="wilcox")
        test2 = StatTest.from_library("Wilcoxon")
        res2 = test2(self.x, self.y)
        self.assertEqual(res1.pvalue, res2.pvalue)

    def test_wilcoxon_legacy_no_zero(self):
        test = StatTest(wilcoxon, "Wilcoxon (legacy)", "Wilcox (L)")
        res1 = test(self.x, self.y)
        test2 = StatTest.from_library("Wilcoxon")
        res2 = test2(self.x, self.y)
        self.assertEqual(res1.pvalue, res2.pvalue)

    def test_wilcoxon_legacy_wilcox_pratt(self):
        test = StatTest(wilcoxon, "Wilcoxon (legacy)", "Wilcox (L)")
        self.assert_print_pvalue(test, "AUTO", "pratt")

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def assert_print_pvalue(self, test, zero_method, expected_output_regex,
                            mock_stdout):
        with self.assertWarns(UserWarning):
            test(self.x, self.y, zero_method=zero_method)
        self.assertRegex(mock_stdout.getvalue(), expected_output_regex)
