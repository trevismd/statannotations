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
    rendered_for = f" for '{for_}'" if for_ is not None else ""
    quotes = "" if "[" in expected else "`"
    raise error_type(f"Expected {quotes}{expected}{quotes}{rendered_for}; "
                     f"got `{got}` instead.")


def check_is_in(x, valid_values, error_type=ValueError, label=None):
    """Raise an error if x is not in valid_values."""
    if x not in valid_values:
        raise_expected_got(
            f"one of {valid_values}", label, x, error_type=error_type)


def remove_null(series):
    return series[pd.notnull(series)]


def get_x_values(data, x) -> set:
    if isinstance(data, pd.DataFrame):
        return set(data[x].unique())
    elif data is not None:
        return set([_ for _ in range(len(data))])
    else:
        return set(x)


def check_order_in_data(data, x, order) -> None:
    if order is None:
        return

    x_values = get_x_values(data, x)
    unmatched_in_order = set(order) - x_values

    if unmatched_in_order:
        raise ValueError(f"Missing x value(s) "
                         f"`{render_set(unmatched_in_order)}` in {x} "
                         f"(specified in `order`)")


def _check_box_pairs_in_data_no_hue(box_pairs: Union[list, tuple],
                                    data: Union[List[list],
                                                pd.DataFrame] = None,
                                    x: Union[str, list] = None) -> None:
    x_values = get_x_values(data, x)
    pairs_x_values = set(itertools.chain(*box_pairs))
    unmatched_x_values = pairs_x_values - x_values
    if unmatched_x_values:
        raise ValueError(f"Missing x value(s) "
                         f"`{render_set(unmatched_x_values)}` in {x}"
                         f" (specified in `box_pairs`) in data")


def _check_box_pairs_in_data_with_hue(box_pairs: Union[list, tuple],
                                      data: Union[List[list],
                                                  pd.DataFrame] = None,
                                      x: Union[str, list] = None,
                                      hue: str = None) -> set:

    x_values = get_x_values(data, x)
    seen_x_values = set()

    hue_values = set(data[hue].unique())

    for x_value, hue_value in itertools.chain(itertools.chain(*box_pairs)):
        if x_value not in seen_x_values and x_value not in x_values:
            raise ValueError(f"Missing x value `{x_value}` in {x}"
                             f" (specified in `box_pairs`)")
        seen_x_values.add(x_value)

        if hue_value not in hue_values:
            raise ValueError(f"Missing hue value `{hue_value}` in {hue}"
                             f" (specified in `box_pairs`)")

    return hue_values


def _check_hue_order_in_data(hue, hue_values: set,
                             hue_order: List[str] = None) -> None:
    if hue_order is not None:
        unmatched = set(hue_order) - set(hue_values)
        if unmatched:
            raise ValueError(f"Missing hue value(s) "
                             f"`{render_set(unmatched)}`"
                             f" in {hue} (specified in `hue_order`)")


def check_box_pairs_in_data(box_pairs: Union[list, tuple],
                            data: Union[List[list], pd.DataFrame] = None,
                            x: Union[str, list] = None,
                            hue: str = None,
                            hue_order: List[str] = None):
    """
    Checks that values referred to in `order` and `box_pairs` exist in data.
    """

    if hue is None and hue_order is None:
        _check_box_pairs_in_data_no_hue(box_pairs, data, x)
    else:
        hue_values = _check_box_pairs_in_data_with_hue(box_pairs, data, x, hue)
        _check_hue_order_in_data(hue, hue_values, hue_order)


def check_not_none(name, value):
    if value is None:
        raise ValueError(f"{name} must be defined and different from `None`")


def check_valid_text_format(text_format):
    check_is_in(
        text_format,
        ['full', 'simple', 'star'],
        label='argument `text_format`'
    )


def render_set(a_set):
    return '","'.join(map(str, a_set))
