import unittest
import warnings

import numpy.testing as npt

from statannot.comparisons_corrections.funcs import bonferroni, holm_bonferroni, benjamin_hochberg


class TestBonferroni(unittest.TestCase):
    """Test Bonferroni correction function."""
    def test_returns_scalar_with_scalar_input(self):
        corrected = bonferroni(0.5)
        with self.assertRaisesRegex(TypeError, 'has no len'):
            # If `corrected` is a scalar, calling `len` should raise an error.
            len(corrected)

    def test_returns_correct_values_with_auto_num_comparisons(self):
        raw_p_values = [0.1, 0.05, 0.5]
        expected = [0.3, 0.15, 1.0]
        observed = bonferroni(raw_p_values)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_values_with_manual_num_comparisons_int(self):
        raw_p_values = [0.1, 0.05, 0.5]
        expected = [0.5, 0.25, 1.0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            observed = bonferroni(raw_p_values, 5)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_values_with_manual_num_comparisons_float(self):
        raw_p_values = [0.1, 0.05, 0.5]
        expected = [0.5, 0.25, 1.0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            observed = bonferroni(raw_p_values, 5.0)
        npt.assert_allclose(observed, expected)


class TestHolmBonferroni(unittest.TestCase):
    """Test Holm-Bonferroni correction function."""
    def test_returns_scalar_with_scalar_input(self):
        corrected = holm_bonferroni(0.5)
        with self.assertRaisesRegex(TypeError, 'has no len'):
            # If `corrected` is a scalar, calling `len` should raise an error.
            len(corrected)

    def test_returns_correct_values_with_alpha_005_and_auto_num_comparisons(self):
        raw_p_values = [0.1, 0.05, 0.5]
        expected = [False, False, False]
        observed = holm_bonferroni(raw_p_values)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_values_with_alpha_005_and__manual_num_comparisons_int(self):
        raw_p_values = [0.1, 0.05, 0.5]
        expected = [False, False, False]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            observed = holm_bonferroni(raw_p_values, 5)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_values_with_alpha_005_and__manual_num_comparisons_float(self):
        raw_p_values = [0.1, 0.05, 0.5]
        expected = [False, False, False]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            observed = holm_bonferroni(raw_p_values, 5.0)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_one_true_value_with_alpha_005_and_auto_num_comparisons(self):
        raw_p_values = [0.1, 0.01, 0.5]
        expected = [False, True, False]
        observed = holm_bonferroni(raw_p_values)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_two_true_values_with_alpha_005_and_auto_num_comparisons(self):
        raw_p_values = [0.1, 0.01, 0.02]
        expected = [False, True, True]
        observed = holm_bonferroni(raw_p_values)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_three_true_values_with_alpha_005_and_auto_num_comparisons(self):
        raw_p_values = [0.05, 0.01, 0.02]
        expected = [True, True, True]
        observed = holm_bonferroni(raw_p_values)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_one_true_value_bis(self):
        raw_p_values = [0.05, 0.01, 0.03]
        expected = [False, True, False]
        observed = holm_bonferroni(raw_p_values)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_three_true_value_with_alpha_006(self):
        raw_p_values = [0.05, 0.01, 0.03]
        expected = [True, True, True]
        observed = holm_bonferroni(raw_p_values, alpha=0.06)
        npt.assert_allclose(observed, expected)


class TestBenjaminHochberg(unittest.TestCase):
    """Test Benjamin-Hochberg correction function."""
    def test_returns_scalar_with_scalar_input(self):
        corrected = benjamin_hochberg(0.5)
        with self.assertRaisesRegex(TypeError, 'has no len'):
            # If `corrected` is a scalar, calling `len` should raise an error.
            len(corrected)

    def test_returns_correct_values_with_alpha_005_and_auto_num_comparisons(self):
        raw_p_values = [0.1, 0.05, 0.5]
        expected = [False, False, False]
        observed = benjamin_hochberg(raw_p_values)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_values_with_alpha_005_and__manual_num_comparisons_int(self):
        raw_p_values = [0.1, 0.05, 0.5]
        expected = [False, False, False]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            observed = benjamin_hochberg(raw_p_values, 5)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_values_with_alpha_005_and__manual_num_comparisons_float(self):
        raw_p_values = [0.1, 0.05, 0.5]
        expected = [False, False, False]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            observed = benjamin_hochberg(raw_p_values, 5.0)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_one_true_value_with_alpha_005_and_auto_num_comparisons(self):
        raw_p_values = [0.1, 0.01, 0.5]
        expected = [False, True, False]
        observed = benjamin_hochberg(raw_p_values)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_two_true_values_with_alpha_005_and_auto_num_comparisons(self):
        raw_p_values = [0.1, 0.01, 0.02]
        expected = [False, True, True]
        observed = benjamin_hochberg(raw_p_values)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_three_true_values_with_alpha_005_and_auto_num_comparisons(self):
        raw_p_values = [0.05, 0.01, 0.02]
        expected = [True, True, True]
        observed = benjamin_hochberg(raw_p_values)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_three_true_values_with_alpha_005_and_more_comparisons(self):
        raw_p_values = [0.03, 0.03, 0.03]
        expected = [True, True, True]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            observed = benjamin_hochberg(raw_p_values, 5.0)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_three_true_one_false_values_with_alpha_005_and_more_comparisons(self):
        raw_p_values = [0.03, 0.03, 0.03, 0.041]
        expected = [True, True, True, False]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            observed = benjamin_hochberg(raw_p_values, 5)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_three_false_values_with_alpha_006_and_more_comparisons(self):
        raw_p_values = [0.03, 0.03, 0.04]
        expected = [False, False, False]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            observed = benjamin_hochberg(raw_p_values, 5, alpha=0.06)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_three_true_value_with_alpha_006(self):
        raw_p_values = [0.06, 0.01, 0.04]
        expected = [True, True, True]
        observed = benjamin_hochberg(raw_p_values, alpha=0.06)
        npt.assert_allclose(observed, expected)
