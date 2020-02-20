import unittest
import warnings

import numpy.testing as npt

from statannot import statannot

class TestBonferroni(unittest.TestCase):
    """Test Bonferroni correction function."""
    def test_returns_scalar_with_scalar_input(self):
        corrected = statannot.bonferroni(0.5)
        with self.assertRaisesRegex(TypeError, 'has no len'):
            # If `corrected` is a scalar, calling `len` should raise an error.
            len(corrected)

    def test_returns_correct_values_with_auto_num_comparisons(self):
        raw_p_values = [0.1, 0.05, 0.5]
        expected = [0.3, 0.15, 1.0]
        observed = statannot.bonferroni(raw_p_values)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_values_with_manual_num_comparisons_int(self):
        raw_p_values = [0.1, 0.05, 0.5]
        expected = [0.5, 0.25, 1.0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            observed = statannot.bonferroni(raw_p_values, 5)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_values_with_manual_num_comparisons_float(self):
        raw_p_values = [0.1, 0.05, 0.5]
        expected = [0.5, 0.25, 1.0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            observed = statannot.bonferroni(raw_p_values, 5.0)
        npt.assert_allclose(observed, expected)

