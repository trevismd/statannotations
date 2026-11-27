import unittest

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None

from statannotations.utils import \
    raise_expected_got, check_is_in, check_not_none, get_x_values, \
    get_closest, render_collection, check_order_in_data


class TestUtils(unittest.TestCase):
    def test_raise_expected_got(self):
        with self.assertRaisesRegex(ValueError,
                                    "Expected `this`; got `that` instead."):
            raise_expected_got("this", None, "that")

    def test_raise_expected_got_for(self):
        with self.assertRaisesRegex(
                ValueError,
                "Expected `this` for 'this_variable'; got `that` instead."):
            raise_expected_got("this", "this_variable", "that")

    def test_check_is_in(self):
        with self.assertRaisesRegex(ValueError, "'then']; got `this`"):
            check_is_in("this", ["that", "there", "then"])

        check_is_in("this", ["that", "there", "then", "this"])

    def test_check_not_none(self):
        with self.assertRaises(ValueError):
            check_not_none("this", None)

        check_not_none("this", "that")

    def test_get_x_values(self):
        self.assertSetEqual({1, 2, 3}, get_x_values(None, [1, 3, 2]))

    def test_get_x_values_polars_like_warns(self):
        class FakePolarsDataFrame:
            __module__ = "polars.dataframe.frame"

            def __len__(self):
                return 2

        with self.assertWarnsRegex(UserWarning, r"(?i)polars"):
            result = get_x_values(FakePolarsDataFrame(), "x")
        self.assertSetEqual({0, 1}, result)

    @unittest.skipIf(pl is None, "polars is not installed")
    def test_get_x_values_polars_dataframe_warns(self):
        df = pl.DataFrame({"x": ["a", "b"], "y": [1, 2]})
        with self.assertWarnsRegex(UserWarning, r"(?i)polars"):
            with self.assertRaises(ValueError):
                check_order_in_data(df, "x", ["a", "b"])

    def test_render_collection(self):
        self.assertEqual('"a", "b"', render_collection(("a", "b")))


class TestClosest(unittest.TestCase):
    def setUp(self):
        self.a_list = [1, 2, 7, 11.1, 11.6, 11.9, 12, 14, 30]

    def test_smaller_than_0(self):
        value = 0
        self.assertEqual(1, get_closest(self.a_list, value))

    def test_larger_than_max(self):
        value = 31
        self.assertEqual(30, get_closest(self.a_list, value))

    def test_btween1(self):
        value = 12.1
        self.assertAlmostEqual(12, get_closest(self.a_list, value))

    def test_middle_take_smallest(self):
        value = 11.95
        self.assertAlmostEqual(11.9, get_closest(self.a_list, value))
