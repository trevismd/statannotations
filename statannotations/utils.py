import itertools
from bisect import bisect_left
from typing import List, Union

import pandas as pd


DEFAULT = object()


class InvalidParametersError(Exception):
    def __init__(self, parameters):
        super().__init__(
            f"Invalid parameter(s) {render_collection(parameters)} to "
            f"configure annotator.")


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
        if x is None:
            return set(data.columns)
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
                         f"`{render_collection(unmatched_in_order)}` in {x} "
                         f"(specified in `order`)")


def _check_pairs_in_data_no_hue(pairs: Union[list, tuple],
                                data: Union[List[list],
                                            pd.DataFrame] = None,
                                x: Union[str, list] = None) -> None:

    x_values = get_x_values(data, x)
    pairs_x_values = set(itertools.chain(*pairs))
    unmatched_x_values = pairs_x_values - x_values
    if unmatched_x_values:
        raise ValueError(f"Missing x value(s) "
                         f"`{render_collection(unmatched_x_values)}` in {x}"
                         f" (specified in `pairs`) in data")


def _check_pairs_in_data_with_hue(pairs: Union[list, tuple],
                                  data: Union[List[list],
                                              pd.DataFrame] = None,
                                  group_coord: Union[str, list] = None,
                                  hue: str = None) -> set:

    x_values = get_x_values(data, group_coord)
    seen_group_values = set()

    hue_values = set(data[hue].unique())

    for group_value, hue_value in itertools.chain(itertools.chain(*pairs)):
        if (group_value not in seen_group_values
                and group_value not in x_values):
            raise ValueError(
                f"Missing group value `{group_value}` in {group_coord}"
                f" (specified in `pairs`)")
        seen_group_values.add(group_value)

        if hue_value not in hue_values:
            raise ValueError(f"Missing hue value `{hue_value}` in {hue}"
                             f" (specified in `pairs`)")

    return hue_values


def _check_hue_order_in_data(hue, hue_values: set,
                             hue_order: List[str] = None) -> None:
    if hue_order is not None:
        unmatched = set(hue_order) - set(hue_values)
        if unmatched:
            raise ValueError(f"Missing hue value(s) "
                             f"`{render_collection(unmatched)}`"
                             f" in {hue} (specified in `hue_order`)")


def check_pairs_in_data(pairs: Union[list, tuple],
                        data: Union[List[list], pd.DataFrame] = None,
                        coord: Union[str, list] = None,
                        hue: str = None,
                        hue_order: List[str] = None):
    """
    Checks that values referred to in `order` and `pairs` exist in data.
    """

    if hue is None and hue_order is None:
        _check_pairs_in_data_no_hue(pairs, data, coord)
    else:
        hue_values = _check_pairs_in_data_with_hue(pairs, data, coord, hue)
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


def render_collection(collection):
    substr = '", "'.join(map(str, collection))
    return f'"{substr}"'


def get_closest(a_list, value):
    """
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.
    from https://stackoverflow.com/a/12141511/9981846
    """
    pos = bisect_left(a_list, value)
    if pos == 0:
        return a_list[0]
    if pos == len(a_list):
        return a_list[-1]
    before, after = a_list[pos - 1: pos + 1]
    if after - value < value - before:
        return after
    return before


def empty_dict_if_none(data):
    if data is None:
        return {}
    return data
