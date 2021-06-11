import itertools
from typing import List, Union

import pandas as pd


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
        raise_expected_got('one of {}'.format(valid_values), label, x,
                           error_type=error_type)


def remove_null(series):
    return series[pd.notnull(series)]


def check_order_box_pairs_in_data(data: pd.DataFrame,
                                  x: str,
                                  box_pairs: Union[list, tuple],
                                  order: List[str] = None,
                                  hue: str = None,
                                  hue_order: List[str] = None):
    """
    Checks that values referred to in `order` and `box_pairs` exist in data.
    """

    x_values = list(data[x].unique())

    if order is not None:
        for x_value in order:
            if x_value not in x_values:
                raise ValueError(f"Missing x value \"{x_value}\" in {x} "
                                 f"(specified in `order`)")

    seen_x_values = set()  # Maybe smaller than x_values

    if hue is None:
        for x_value in itertools.chain(*box_pairs):
            if x_value not in seen_x_values and x_value not in x_values:
                raise ValueError(f"Missing x value \"{x_value}\" in {x}"
                                 f" (specified in `box_pairs`) in data")
            seen_x_values.add(x_value)

    else:
        hue_values = list(data[hue].unique())
        seen_hues = set()
        for x_value, hue_value in itertools.chain(itertools.chain(*box_pairs)):
            if x_value not in seen_x_values and x_value not in x_values:
                raise ValueError(f"Missing x value \"{x_value}\" in {x}"
                                 f" (specified in `box_pairs`)")
            seen_x_values.add(x_value)

            if hue_value not in seen_hues and hue_value not in hue_values:
                raise ValueError(f"Missing hue value \"{hue_value}\" in {hue}"
                                 f" (specified in `box_pairs`)")
            seen_hues.add(hue_value)

        # I assume hue is always defined with hue_order
        if hue_order is not None:
            for hue_value in hue_order:
                if hue_value not in seen_hues and hue_value not in hue_values:
                    raise ValueError(f"Missing hue value \"{hue_value}\" in {hue}"
                                     f" (specified in `hue_order`)")
