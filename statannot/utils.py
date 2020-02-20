def raise_expected_got(expected, for_, got, error_type=ValueError):
    """Raise a standardized error message.

    Raise an `error_type` error with the message
        Expected `expected` for `for_`; got `got` instead.
    Or, if `for_` is `None`,
        Expected `expected`; got `got` instead.

    """
    if for_ is not None:
        raise error_type(
            'Expected {} for {}; got {} instead.'.format(expected, for_, got)
        )
    else:
        raise error_type(
            'Expected {}; got {} instead.'.format(expected, got)
        )


def assert_is_in(x, valid_values, error_type=ValueError, label=None):
    """Raise an error if x is not in valid_values."""
    if x not in valid_values:
            raise_expected_got('one of {}'.format(valid_values), label, x)
