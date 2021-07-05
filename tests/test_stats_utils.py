import unittest

from statannotations.stats.utils import check_alpha, return_results, \
    get_num_comparisons, check_pvalues, check_num_comparisons


class TestStatUtils(unittest.TestCase):
    def setUp(self):
        self.one_pvalue = 0.05
        self.two_pvalues = [0.05, 0.06]

    def test_check_alpha_too_large(self):
        with self.assertRaises(ValueError):
            check_alpha(1.2)

    def test_check_alpha_negative(self):
        with self.assertRaises(ValueError):
            check_alpha(-0.2)

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

    def test_check_pvalues_format(self):
        with self.assertRaises(ValueError):
            check_pvalues([[1, 2, 3], 1])

    def test_get_num_comparisons_fewer_warns(self):
        with self.assertWarnsRegex(UserWarning, "smaller"):
            get_num_comparisons([0.1, 0.2, 0.3], 2)

    def test_check_num_comparisons_string_not_auto_raises(self):
        with self.assertRaises(ValueError):
            check_num_comparisons("not_auto")

    def test_check_num_comparisons_string_auto_not_raises(self):
        check_num_comparisons("auto")

    def test_check_num_comparisons_list_raises(self):
        self.assertRaises(ValueError, check_num_comparisons, [2])

    def test_check_num_comparisons_float_raises(self):
        self.assertRaises(ValueError, check_num_comparisons, 2.3)

    def test_check_num_comparisons_zero_raises(self):
        self.assertRaises(ValueError, check_num_comparisons, 0)
