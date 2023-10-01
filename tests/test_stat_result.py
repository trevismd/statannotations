import unittest

from statannotations.stats.ComparisonsCorrection import ComparisonsCorrection
from statannotations.stats.StatResult import StatResult


class TestStatResult(unittest.TestCase):
    """Test `statannotations.stats.StatResult` class"""

    def setUp(self) -> None:
        self.benjamini_hochberg = ComparisonsCorrection("Benjamini-Hochberg")
        self.stat_result = StatResult("Test X", "X", "Stat", 1, 0.02, alpha=0.05)
        self.stat_result.correction_method = self.benjamini_hochberg.name
        self.stat_result_non_significant = StatResult(
            "Test X", "X", "Stat", 1, 0.06, alpha=0.05)
        self.stat_result_non_significant.correction_method = (
            self.benjamini_hochberg.name
        )

    def test_ns_if_ns(self):
        self.stat_result.corrected_significance = False
        assert self.stat_result.formatted_output == (
            "Test X with Benjamini-Hochberg correction, P_val:2.000e-02 (ns) "
            "Stat=1.000e+00"
        )

    def test_nothing_if_s(self):
        self.stat_result.corrected_significance = True
        assert self.stat_result.formatted_output == (
            "Test X with Benjamini-Hochberg correction, P_val:2.000e-02 "
            "Stat=1.000e+00"
        )

    def test_is_significant(self):
        assert self.stat_result.is_significant

    def test_is_non_significant(self):
        self.assertFalse(self.stat_result_non_significant.is_significant)

    def test_corrected_significance(self):
        self.stat_result.corrected_significance = False
        self.assertFalse(self.stat_result.is_significant)
