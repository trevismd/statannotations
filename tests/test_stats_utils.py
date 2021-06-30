import unittest

from statannotations.stats.utils import check_alpha, return_results, get_num_comparisons


class TestStatUtils(unittest.TestCase):
    def setUp(self):
        self.one_pvalue = 0.05
        self.two_pvalues = [0.05, 0.06]

    def test_check_alpha(self):
        with self.assertRaises(ValueError):
            check_alpha(1.2)

    def test_result_scalar(self):
        self.assertEqual(0.05, return_results([0.05]))

    def test_result_array(self):
        self.assertEqual(self.two_pvalues, return_results(self.two_pvalues))

    def test_auto_num_comparisons(self):
        self.assertEqual(2, get_num_comparisons(self.two_pvalues, "auto"))

    def test_manual_num_comparisons_abnormal(self):
        with self.assertWarnsRegex(UserWarning, "Manually-specified"):
            get_num_comparisons(self.two_pvalues, 1)

    def test_manual_num_comparisons(self):
        self.assertEqual(3, get_num_comparisons(self.two_pvalues, 3))
