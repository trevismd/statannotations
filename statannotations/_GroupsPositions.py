from __future__ import annotations

from collections.abc import Iterator, Sequence
import itertools
from typing import TYPE_CHECKING
import warnings

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .compat import TupleGroup, TGroupValue, THueValue


def get_group_names_and_labels(
    group_names: Sequence[TGroupValue],
    hue_names: Sequence[THueValue],
) -> tuple[list[TupleGroup], list[str]]:
    tuple_group_names: list[TupleGroup]
    if len(hue_names) == 0:
        tuple_group_names = [(name,) for name in group_names]
        labels = [str(name) for name in group_names]

    else:
        labels = []
        tuple_group_names = []
        for group_name, hue_name in itertools.product(group_names, hue_names):
            tuple_group_names.append((group_name, hue_name))
            labels.append(f'{group_name}_{hue_name}')

    return tuple_group_names, labels


class _GroupsPositions:
    POSITION_TOLERANCE: float = 0.1

    width: float
    tuple_group_names: list[TupleGroup]
    labels: list[str]
    _data: pd.DataFrame

    def __init__(
        self,
        group_names: Sequence[TGroupValue],
        hue_names: Sequence[THueValue],
        *,
        dodge: bool = True,
        gap: float = 0.0,
        width: float = 0.8,
        native_group_offsets: Sequence | None = None,
    ) -> None:
        self.gap = gap
        self.dodge = dodge

        self._group_names = group_names
        self._hue_names = hue_names
        self.use_hue = len(hue_names) == 0

        # Compute the coordinates of the groups (without hue) and the width
        self.group_offsets, self.width = self._set_group_offsets(
            group_names, native_group_offsets, width
        )
        # Create the tuple (group, hue) and the labels
        self.tuple_group_names, self.labels = get_group_names_and_labels(group_names, hue_names)

        # Create dataframe with the groups, labels and positions
        # this should be done last, when the other attributes are defined
        self._data, self._artist_width = self._set_data(dodge=dodge, gap=gap)

    def _set_group_offsets(
        self,
        group_names: Sequence,
        native_group_offsets: Sequence | None,
        width: float,
    ) -> tuple[Sequence, float]:
        """Set the group offsets from native scale and scale the width."""
        group_offsets = list(range(len(group_names)))
        if native_group_offsets is not None:
            curated_offsets = [v for v in native_group_offsets]
            if len(curated_offsets) != len(group_names):
                msg = (
                    'The values of the categories with "native_scale=True" do not correspond '
                    'to the category names. Maybe some values are not finite?'
                )
                warnings.warn(msg)
            else:
                group_offsets = curated_offsets
                if len(curated_offsets) > 1:
                    native_width = np.min(np.diff(curated_offsets))
                    width *= native_width

        return group_offsets, width

    def _set_data(self, dodge: bool, gap: float) -> tuple[pd.DataFrame, float]:
        n_repeat = max(len(self._hue_names), 1)
        group_positions = np.array(self.group_offsets)
        positions = np.repeat(group_positions, n_repeat)
        artist_width = float(self.width)
        data = pd.DataFrame(
            {
                'group': self.tuple_group_names,
                'label': self.labels,
                'pos': positions,
            },
        )
        if dodge and self.use_hue:
            n_hues = max(len(self._hue_names), 1)
            artist_width /= n_hues
            # evenly space range centered in zero (subtracting the mean)
            offset = artist_width * (np.arange(n_hues) - (n_hues - 1) / 2)
            tiled_offset = np.tile(offset, len(self._group_names))
            data['pos'] += tiled_offset
        if gap and gap >= 0 and gap <= 1:
            artist_width *= 1 - gap

        return data, artist_width

    def find_group_at_pos(self, pos: float, *, verbose: bool = False) -> TupleGroup | None:
        positions = self._data['pos']
        if len(positions) == 0:
            return None
        # Get the index of the closest position
        index = (positions - pos).abs().idxmin()
        found_pos = positions.loc[index]

        if verbose and abs(found_pos - pos) > self.POSITION_TOLERANCE:
            # The requested position is not an artist position
            msg = (
                'Invalid x-position found. Are the same parameters passed to '
                'seaborn and statannotations calls? Or are there few data points? '
                f'The closest group position to {pos} is {found_pos}'
            )
            warnings.warn(msg)
        return self._data.loc[index, 'group']

    def get_group_axis_position(self, group: TupleGroup) -> float:
        """Get the position of the group.

        group_name can be either a tuple ("group",) or a tuple ("group", "hue")
        """
        group_names = self._data['group']
        if group not in group_names:
            msg = f'Group {group} was not found in the list: {group_names}'
            raise ValueError(msg)
        index = (group_names == group).idxmax()
        pos = float(self._data.loc[index, 'pos'])
        # round the position
        return round(pos / self.POSITION_TOLERANCE) * self.POSITION_TOLERANCE

    @property
    def artist_width(self) -> float:
        return float(self._artist_width)

    def compatible_width(self, width: float) -> bool:
        """Check if the rectangle width is smaller than the artist width."""
        return abs(width) <= 1.1 * self._artist_width

    def iter_groups(self) -> Iterator[tuple[TupleGroup, str, float]]:
        """Iterate the groups and return a tuple (group_tuple, group_label, group_position)."""
        yield from self._data[['group', 'label', 'pos']].itertuples(index=False, name=None)
