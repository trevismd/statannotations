def raise_expected_got(expected, for_, got, error_type=ValueError):
    """Raise a standardized error message.

    Raise an `error_type` error with the message:
        Expected `expected` for `for_`; got `got` instead.

    """
    raise error_type(
        'Expected {} for {}; got {} instead.'.format(expected, for_, got)
    )
