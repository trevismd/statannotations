import unittest

from statannotations.utils import \
    raise_expected_got, check_is_in, check_not_none, get_x_values


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
