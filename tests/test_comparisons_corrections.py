import unittest
import warnings
from functools import partial

import numpy.testing as npt
from statsmodels.stats.multitest import multipletests

from statannotations.stats.ComparisonsCorrection import \
    ComparisonsCorrection, get_validated_comparisons_correction


class TestBonferroni(unittest.TestCase):
    """Test Bonferroni correction implementation."""

    def setUp(self) -> None:
        self.bonferroni = ComparisonsCorrection("bonferroni")

        self.bonferroni_manual = ComparisonsCorrection(
            partial(multipletests, method="bonferroni"),
            name="Bonferroni (manual)", method_type=0)

    def test_returns_scalar_with_scalar_input(self):
        corrected = self.bonferroni(0.5)
        with self.assertRaisesRegex(TypeError, 'has no len'):
            # If `corrected` is a scalar, calling `len` should raise an error.
            len(corrected)

    def test_returns_correct_values_with_auto_num_comparisons(self):
        raw_p_values = [0.1, 0.05, 0.5]
        expected = [0.3, 0.15, 1.0]
        observed = self.bonferroni(raw_p_values)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_values_with_auto_num_comparisons_man(self):
        raw_p_values = [0.1, 0.05, 0.5]
        expected = [0.3, 0.15, 1.0]
        observed = self.bonferroni_manual(raw_p_values)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_values_with_manual_num_comparisons_int(self):
        raw_p_values = [0.1, 0.05, 0.5]
        expected = [0.5, 0.25, 1.0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            observed = self.bonferroni(raw_p_values, 5)

        npt.assert_allclose(observed, expected)

    def test_returns_correct_values_with_manual_num_comparisons_int_man(self):
        raw_p_values = [0.1, 0.05, 0.5]
        expected = [0.5, 0.25, 1.0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            observed = self.bonferroni_manual(raw_p_values, 5)

        npt.assert_allclose(observed, expected)

    def test_returns_correct_values_with_manual_num_comparisons_float(self):
        raw_p_values = [0.1, 0.05, 0.5]
        expected = [0.5, 0.25, 1.0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            observed = self.bonferroni(raw_p_values, 5.0)
        npt.assert_allclose(observed, expected)


class TestHolmBonferroni(unittest.TestCase):
    """Test Holm-Bonferroni correction implementation."""

    def setUp(self) -> None:
        self.holm_bonferroni = ComparisonsCorrection("holm")
        self.holm_bonferroni_006 = ComparisonsCorrection("holm", alpha=0.06)

        self.holm_bonferroni_006_manual = ComparisonsCorrection(
            partial(multipletests, method="holm"),
            name="Holm-Bonferroni (manual)", method_type=1,
            corr_kwargs={"alpha": 0.06})

        self.holm_bonferroni_006_manual_bis = ComparisonsCorrection(
            partial(multipletests, method="holm", alpha=0.06),
            name="Holm-Bonferroni (manual)", method_type=1)

    def test_returns_scalar_with_scalar_input(self):
        corrected = self.holm_bonferroni(0.5)
        with self.assertRaisesRegex(TypeError, 'has no len'):
            # If `corrected` is a scalar, calling `len` should raise an error.
            len(corrected)

    def test_returns_correct_values_with_alpha_005_and_auto_num_comparisons(self):
        raw_p_values = [0.1, 0.05, 0.5]
        expected = [False, False, False]
        observed = self.holm_bonferroni(raw_p_values)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_values_with_alpha_005_and__manual_num_comparisons_int(self):
        raw_p_values = [0.1, 0.05, 0.5]
        expected = [False, False, False]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            observed = self.holm_bonferroni(raw_p_values, 5)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_values_with_alpha_005_and__manual_num_comparisons_float(self):
        raw_p_values = [0.1, 0.05, 0.5]
        expected = [False, False, False]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            observed = self.holm_bonferroni(raw_p_values, 5.0)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_one_true_value_with_alpha_005_and_auto_num_comparisons(self):
        raw_p_values = [0.1, 0.01, 0.5]
        expected = [False, True, False]
        observed = self.holm_bonferroni(raw_p_values)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_two_true_values_with_alpha_005_and_auto_num_comparisons(self):
        raw_p_values = [0.1, 0.01, 0.02]
        expected = [False, True, True]
        observed = self.holm_bonferroni(raw_p_values)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_three_true_values_with_alpha_005_and_auto_num_comparisons(self):
        raw_p_values = [0.05, 0.01, 0.02]
        expected = [True, True, True]
        observed = self.holm_bonferroni(raw_p_values)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_one_true_value_bis(self):
        raw_p_values = [0.05, 0.01, 0.03]
        expected = [False, True, False]
        observed = self.holm_bonferroni(raw_p_values)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_three_true_value_with_alpha_006(self):
        raw_p_values = [0.05, 0.01, 0.03]
        expected = [True, True, True]
        observed = self.holm_bonferroni_006(raw_p_values)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_three_true_value_with_alpha_006_man(self):
        raw_p_values = [0.05, 0.01, 0.03]
        expected = [True, True, True]
        observed = self.holm_bonferroni_006_manual(raw_p_values)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_three_true_value_with_alpha_006_man_bis(self):
        raw_p_values = [0.05, 0.01, 0.03]
        expected = [True, True, True]
        observed = self.holm_bonferroni_006_manual_bis(raw_p_values)
        npt.assert_allclose(observed, expected)


class TestBenjaminiHochberg(unittest.TestCase):
    """Test Benjamini-Hochberg correction implementation."""

    @staticmethod
    def my_bh_from_sm(pvalues):
        # This function directly returns an array of booleans, so it does not
        # match statsmodels api of multipletests
        return multipletests(pvalues, method="fdr_bh", alpha=0.06)[0]

    def setUp(self) -> None:
        self.benjamini_hochberg = ComparisonsCorrection("Benjamini-Hochberg")
        self.benjamini_hochberg_006 = ComparisonsCorrection(
            "Benjamini-Hochberg", alpha=0.06)

        self.benjamini_hochberg_006_man_diff_api = ComparisonsCorrection(
            self.my_bh_from_sm, name="Benjamini-Hochberg (manual)",
            method_type=1, statsmodels_api=False)

    def test_returns_scalar_with_scalar_input(self):
        corrected = self.benjamini_hochberg(0.5)
        with self.assertRaisesRegex(TypeError, 'has no len'):
            # If `corrected` is a scalar, calling `len` should raise an error.
            len(corrected)

    def test_returns_correct_values_with_alpha_005_and_auto_num_comparisons(self):
        raw_p_values = [0.1, 0.05, 0.5]
        expected = [False, False, False]
        observed = self.benjamini_hochberg(raw_p_values)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_values_with_alpha_005_and__manual_num_comparisons_int(self):
        raw_p_values = [0.1, 0.05, 0.5]
        expected = [False, False, False]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            observed = self.benjamini_hochberg(raw_p_values, 5)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_values_with_alpha_005_and__manual_num_comparisons_float(self):
        raw_p_values = [0.1, 0.05, 0.5]
        expected = [False, False, False]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            observed = self.benjamini_hochberg(raw_p_values, 5.0)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_one_true_value_with_alpha_005_and_auto_num_comparisons(self):
        raw_p_values = [0.1, 0.01, 0.5]
        expected = [False, True, False]
        observed = self.benjamini_hochberg(raw_p_values)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_two_true_values_with_alpha_005_and_auto_num_comparisons(self):
        raw_p_values = [0.1, 0.01, 0.02]
        expected = [False, True, True]
        observed = self.benjamini_hochberg(raw_p_values)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_three_true_values_with_alpha_005_and_auto_num_comparisons(self):
        raw_p_values = [0.05, 0.01, 0.02]
        expected = [True, True, True]
        observed = self.benjamini_hochberg(raw_p_values)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_three_true_values_with_alpha_005_and_more_comparisons(self):
        raw_p_values = [0.03, 0.03, 0.03]
        expected = [True, True, True]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            observed = self.benjamini_hochberg(raw_p_values, 5.0)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_three_true_one_false_values_with_alpha_005_and_more_comparisons(self):
        raw_p_values = [0.03, 0.03, 0.03, 0.041]
        expected = [True, True, True, False]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            observed = self.benjamini_hochberg(raw_p_values, 5)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_three_false_values_with_alpha_006_and_more_comparisons(self):
        raw_p_values = [0.03, 0.03, 0.04]
        expected = [False, False, False]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            observed = self.benjamini_hochberg_006(raw_p_values, 5)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_three_true_value_with_alpha_006(self):
        raw_p_values = [0.06, 0.01, 0.04]
        expected = [True, True, True]
        observed = self.benjamini_hochberg_006(raw_p_values)
        npt.assert_allclose(observed, expected)

    def test_returns_correct_three_true_value_with_alpha_006_man_diff_api(self):
        raw_p_values = [0.06, 0.01, 0.04]
        expected = [True, True, True]
        observed = self.benjamini_hochberg_006_man_diff_api(raw_p_values)
        npt.assert_allclose(observed, expected)

    def test_values_of_documentation(self):
        raw_p_values = [6.477e-01, 4.690e-02, 2.680e-02]
        expected = [False, False, False]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            observed = self.benjamini_hochberg(raw_p_values)
        npt.assert_allclose(observed, expected)

    def test_min_num_comparisons(self):
        raw_p_values = [6.477e-01, 4.690e-02, 2.680e-02]
        with self.assertRaisesRegex(ValueError, "be at least "):
            self.benjamini_hochberg(raw_p_values, num_comparisons=2)

    def test_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            ComparisonsCorrection('that method')


class TestGetValidated(unittest.TestCase):
    def test_get_validated_comparisons_correction_none_to_none(self):
        self.assertIsNone(get_validated_comparisons_correction(None))

    def test_get_validated_comparisons_correction_incorrect_string(self):
        self.assertRaises(ValueError,
                          get_validated_comparisons_correction,
                          "that method")

    def test_get_validated_comparisons_correction_incorrect_format(self):
        self.assertRaises(ValueError,
                          get_validated_comparisons_correction,
                          ["that method"])

    def test_get_validated_comparisons_correction_str_to_instance(self):
        self.assertIsInstance(get_validated_comparisons_correction("BH"),
                              ComparisonsCorrection)

    def test_get_validated_comparisons_correction_to_results(self):
        raw_p_values = [6.477e-01, 4.690e-02, 2.680e-02]
        expected = [False, False, False]
        cc = get_validated_comparisons_correction("BH")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            observed = cc(raw_p_values)
        npt.assert_allclose(observed, expected)
