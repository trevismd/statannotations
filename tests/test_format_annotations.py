import unittest

from statannotations.format_annotations import simple_text
from statannotations.stats.StatResult import StatResult


class TestFormatAnnotations(unittest.TestCase):
    """Test `statannotations.format_annotations` functions"""

    def setUp(self):
        self.result = StatResult("This long description", "This", "stat", 3,
                                 0.3, alpha=0.05)

    def test_format_simple_setting_threshold(self):

        self.assertEqual(
            simple_text(self.result, "{:.2f}", [(0.5, 0.5)]),
            "This p ≤ 0.5")

    def test_format_simple_setting_multiple_thresholds(self):

        self.assertEqual(
            simple_text(self.result,
                        "{:.2f}", [(0.1, 0.1), (0.5, 0.5)]),
            "This p ≤ 0.5")

    def test_format_simple_non_significant(self):

        self.assertEqual(
            simple_text(self.result, "{:.2f}", [(0.2, 0.2)]),
            "This p = 0.30")
