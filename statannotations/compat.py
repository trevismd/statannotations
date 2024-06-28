from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable
import warnings

import numpy as np
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from typing import NotRequired, Union, TypedDict, TypeVar

    import pandas as pd

    TGroupValue = TypeVar('TGroupValue')
    THueValue = TypeVar('THueValue')
    TupleGroup = Union[tuple[TGroupValue], tuple[TGroupValue, THueValue]]

    class Struct(TypedDict):
        group: TupleGroup
        label: str
        group_coord: float
        group_data: Sequence
        value_max: float
        group_i: NotRequired[int]

from .utils import remove_null

## Only allow seaborn <0.12 or >=0.13
sns_version = tuple(int(v) for v in sns.__version__.split('.')[:3])

if sns_version < (0, 12, 0):
    from seaborn.categorical import (
        _BoxPlotter,
        _SwarmPlotter,
        _StripPlotter,
        _BarPlotter,
        _ViolinPlotter,
        _CategoricalPlotter,
    )

elif sns_version < (0, 13, 0):
    from seaborn.categorical import (
        _BoxPlotter,
        _BarPlotter,
        _ViolinPlotter,
        _CategoricalPlotterNew,
        _CategoricalPlotter,
    )
    _SwarmPlotter = _StripPlotter = _CategoricalPlotterNew

else:
    _CategoricalPlotter = sns.categorical._CategoricalPlotter


def fix_and_warn(dodge, hue, plot):
    if dodge is False and hue is not None:
        raise ValueError('`dodge` cannot be False in statannotations.')

    is_plot_type = plot in ('swarmplot', 'stripplot')
    if is_plot_type and hue is not None and dodge is not True:
        msg = (
            'Implicitly setting dodge to True as it is necessary in '
            'statannotations. It must have been True for the seaborn '
            'call to yield consistent results when using `hue`.'
        )
        warnings.warn(msg)

def _get_categorical_plotter(
    plot_type: str,
    *,
    x,
    y,
    hue,
    data,
    order,
    **plot_params,
) -> _CategoricalPlotter:
    # Keyword arguments for the plotters
    kwargs = {
        'data': data,
        'order': order,
        'orient': plot_params.get('orient'),
        'color': None,
    }
    variables = {'x': x, 'y': y, 'hue': hue}
    new_kwargs = {
        'variables': variables,
        'legend': plot_params.get('legend'),
        **kwargs,
    }

    # Boxplot
    if plot_type == 'boxplot':
        # Pre-0.13 seaborn API
        if sns_version < (0, 13, 0):
            kwargs = {
                'hue_order': plot_params.get('hue_order'),
                'dodge': True,
                'palette': None,
                'saturation': 0.75,
                'linewidth': plot_params.get('linewidth'),
                'width': plot_params.get('width', 0.8),
                'fliersize': plot_params.get('fliersize', 5),
                **variables,
                **kwargs,
            }
            return _BoxPlotter(**kwargs)

        # 0.13 seaborn API
        return _CategoricalPlotter(**new_kwargs)

    elif plot_type == 'barplot':
        # Pre-0.13 seaborn API
        if sns_version < (0, 13, 0):
            kwargs = {
                'hue_order': plot_params.get('hue_order'),
                'dodge': True,
                'palette': None,
                'saturation': 0.75,
                'estimator': plot_params.get('estimator', np.mean),
                'n_boot': plot_params.get('n_boot', 1000),
                'units': plot_params.get('units'),
                'seed': plot_params.get('seed'),
                'errcolor': '.26',
                'errwidth': plot_params.get('errwidth'),
                'capsize': None,
                **variables,
                **kwargs,
            }

            # Pre-0.12 seaborn API
            if sns_version < (0, 12, 0):
                kwargs = {
                    'ci': plot_params.get('ci', 95),
                    **kwargs,
                }
                return _BarPlotter(**kwargs)

            # 0.12 seaborn API
            kwargs = {
                'errorbar': plot_params.get('errorbar', ('ci', 95)),
                'width': plot_params.get('width', 0.8),
                **kwargs,
            }
            return _BarPlotter(**kwargs)

        # 0.13 seaborn API
        return _CategoricalPlotter(**new_kwargs)

    elif plot_type == 'violinplot':
        # Pre-0.13 seaborn API
        if sns_version < (0, 13, 0):
            kwargs = {
                'hue_order': plot_params.get('hue_order'),
                'dodge': True,
                'palette': None,
                'saturation': 0.75,
                'linewidth': plot_params.get('linewidth'),
                'width': plot_params.get('width', 0.8),
                'bw': plot_params.get('bw', 'scott'),
                'cut': plot_params.get('cut', 2),
                'scale': plot_params.get('scale', 'area'),
                'scale_hue': plot_params.get('scale_hue', True),
                'gridsize': plot_params.get('gridsize', 100),
                'inner': plot_params.get('inner', None),
                'split': plot_params.get('split', False),
                **variables,
                **kwargs,
            }
            return _ViolinPlotter(**kwargs)

        # 0.13 seaborn API
        return _CategoricalPlotter(**new_kwargs)

    elif plot_type == 'swarmplot':
        # Pre-0.12 seaborn API
        if sns_version < (0, 12, 0):
            kwargs = {
                'hue_order': plot_params.get('hue_order'),
                'dodge': True,
                'palette': None,
                **variables,
                **kwargs,
            }
            return _SwarmPlotter(**kwargs)

        # Pre-0.13 seaborn API
        if sns_version < (0, 13, 0):
            if 'color' in new_kwargs:
                new_kwargs.pop('color')
            return _CategoricalPlotterNew(require_numeric=False, **new_kwargs)

        # 0.13 seaborn API
        return _CategoricalPlotter(**new_kwargs)

    elif plot_type == 'stripplot':
        # Pre-0.12 seaborn API
        if sns_version < (0, 12, 0):
            kwargs = {
                'hue_order': plot_params.get('hue_order'),
                'dodge': True,
                'palette': None,
                'jitter': plot_params.get('jitter', True),
                **variables,
                **kwargs,
            }
            return _StripPlotter(**kwargs)

        # Pre-0.13 seaborn API
        if sns_version < (0, 13, 0):
            if 'color' in new_kwargs:
                new_kwargs.pop('color')
            return _CategoricalPlotterNew(require_numeric=False, **new_kwargs)

        # 0.13 seaborn API
        return _CategoricalPlotter(**new_kwargs)

    msg = f'Plot type {plot_type!r} is not supported'
    raise NotImplementedError(msg)


class Wrapper:
    """Compatibility wrapper for seaborn.categorical._CategoricalPlotter."""

    width: float
    gap: float
    dodge: bool
    is_redundant_hue: bool
    native_group_offsets: Sequence | None

    def __init__(
        self,
        plot_type: str,
        *,
        is_redundant_hue: bool = False,
        **kwargs,
    ) -> None:
        dodge = kwargs.get('dodge')
        hue = kwargs.get('hue')
        fix_and_warn(dodge, hue, plot_type)
        # dodge needs to be True for statannotations
        self.dodge = True

        self.gap = kwargs.get('gap', 0.0)
        self.width = kwargs.get('width', 0.8)
        self.native_group_offsets = None
        self.is_redundant_hue = is_redundant_hue

    @property
    def has_violin_support(self) -> bool:
        """Whether the max values of a violin plot can be extracted from the plotter."""
        return hasattr(self, '_populate_value_maxes_violin')

    @property
    def has_hue(self) -> bool:
        raise NotImplementedError

    @property
    def group_names(self) -> list:
        raise NotImplementedError

    @property
    def hue_names(self) -> list:
        raise NotImplementedError

    def get_group_data(self, group_name):
        """Get the data for the (group[, hue]) tuple.

        group_name can be either a tuple ("cat",) or a tuple ("cat", "hue")
        """
        raise NotImplementedError

    def parse_pairs(
        self,
        pairs: list[tuple],
        structs: list[Struct],
        *,
        formatter: Callable | None = None,
    ) -> list[tuple[TupleGroup, TupleGroup]]:
        struct_groups = [struct['group'] for struct in structs]
        ret: list[tuple[TupleGroup, TupleGroup]] = []

        def format_group(value) -> TupleGroup:
            """Format the group (but not the optional hue)."""
            return value if isinstance(value, tuple) else (value,)

        for pair in pairs:
            if not isinstance(pair, Sequence) or len(pair) != 2:
                msg = f'pair {pair} is not a 2-tuple, skipping.'
                warnings.warn(msg)
                continue
            # Format the groups
            new_pair = tuple(format_group(v) for v in pair)

            # Check that the groups are valid group names
            valid_group = True
            for i, group in enumerate(new_pair):
                if group not in struct_groups:
                    msg = (
                        f'cannot find group{i} of pair in the group tuples: '
                        f'{group} not in {struct_groups}'
                    )
                    warnings.warn(msg)
                    valid_group = False
            if not valid_group:
                continue

            ret.append(new_pair)

        if len(ret) == 0:
            msg = (
                f'pairs are empty after parsing: original_pairs={pairs}\n'
                f'not in group_list={struct_groups}'
            )
            raise ValueError(msg)
        return ret


class CategoricalPlotterWrapper_v11(Wrapper):
    """Compatibility wrapper for seaborn v11."""

    _plotter: _CategoricalPlotter

    def __init__(self,
        plot_type: str,
        *,
        x,
        y,
        hue,
        data,
        order,
        **plot_params,
    ) -> None:
        super().__init__(plot_type, hue=hue, **plot_params)

        self._plotter = _get_categorical_plotter(
            plot_type,
            x=x,
            y=y,
            hue=hue,
            data=data,
            order=order,
            **plot_params,
        )

    @property
    def has_hue(self) -> bool:
        return self._plotter.plot_hues is not None

    @property
    def group_names(self) -> list:
        return self._plotter.group_names

    @property
    def hue_names(self) -> list:
        if self.is_redundant_hue:
            return []
        if not self.has_hue:
            return []
        # Can be None, force to be an empty list
        return self._plotter.hue_names or []

    def _get_group_data_from_plotter(
        self,
        cat: TGroupValue | tuple[TGroupValue],
    ) -> tuple[Any, int]:
        if isinstance(cat, tuple):
            cat = cat[0]
        index = self.group_names.index(cat)

        return self._plotter.plot_data[index], index

    def _get_group_data_with_hue(self, group_name: tuple):
        cat = group_name[0]
        group_data, index = self._get_group_data_from_plotter(cat)

        hue_level = group_name[1]
        hue_mask = self._plotter.plot_hues[index] == hue_level

        return group_data[hue_mask]

    def _get_group_data_without_hue(self, group_name):
        group_data, _ = self._get_group_data_from_plotter(group_name)

        return group_data

    def get_group_data(self, group_name):
        """Get the data for the (group[, hue]) tuple.

        group_name can be either a tuple ("cat",) or a tuple ("cat", "hue")

        Almost a duplicate of seaborn's code, because there is not
        direct access to the group_data in the respective Plotter class.
        """
        if not isinstance(group_name, tuple) or len(group_name) == 1:
            data = self._get_group_data_without_hue(group_name)
        else:
            data = self._get_group_data_with_hue(group_name)

        group_data = remove_null(data)

        return group_data

    def _populate_value_maxes_violin(self, value_maxes: dict[TupleGroup, float], data_to_ax) -> dict[TupleGroup, float]:
        """Populate the max values for violinplot.

        The keys should follow the tuples of `get_group_names_and_labels`.
        """
        dict_keys = list(value_maxes.keys())
        for group_idx, group_name in enumerate(self._plotter.group_names):
            if self.has_hue:
                for hue_idx, hue_name in enumerate(self._plotter.hue_names):
                    value_pos = max(self._plotter.support[group_idx][hue_idx])
                    key = (group_name, hue_name)
                    if key not in value_maxes:
                        msg = f'key {key} was not found in the dict: {dict_keys}'
                        warnings.warn(msg)
                        continue
                    value_maxes[key] = data_to_ax.transform((0, value_pos))[1]
            else:
                value_pos = max(self._plotter.support[group_idx])
                key = group_name
                if key not in value_maxes:
                    msg = f'key {key} was not found in the dict: {dict_keys}'
                    warnings.warn(msg)
                    continue
                value_maxes[key] = data_to_ax.transform((0, value_pos))[1]
        return value_maxes


class CategoricalPlotterWrapper_v12(Wrapper):
    """Compatibility wrapper for seaborn v12."""

    _plotter: _CategoricalPlotter

    def __init__(self,
        plot_type: str,
        *,
        x,
        y,
        hue,
        data,
        order,
        **plot_params,
    ) -> None:
        super().__init__(plot_type, hue=hue, **plot_params)
        self._group_names = None

        self._plotter = _get_categorical_plotter(
            plot_type,
            x=x,
            y=y,
            hue=hue,
            data=data,
            order=order,
            **plot_params,
        )

        if isinstance(self._plotter, _CategoricalPlotterNew):
            # Order the group variables
            native_scale = plot_params.get('native_scale', False)
            if native_scale:
                msg = '`native_scale=True` is not supported with seaborn==0.12, update to seaborn>=0.13'
                raise ValueError(msg)
            formatter = plot_params.get('formatter')
            if formatter is not None:
                msg = '`formatter` is not supported with seaborn==0.12, update to seaborn>=0.13'
                raise ValueError(msg)
            self._order_variable(order=order, native_scale=native_scale, formatter=formatter)

            # Native scaling of the group variable
            if native_scale:
                self.native_group_offsets = self._plotter.plot_data[self.axis]

    @property
    def has_hue(self) -> bool:
        if self.is_redundant_hue:
            return False
        if hasattr(self._plotter, 'hue_names'):
            # Can be None, force to be an empty list
            if self._plotter.hue_names:
                return True
            return False

        else:
            # _CategoricalPlotterNew
            return self._plotter.variables.get('hue') is not None

    @property
    def group_names(self) -> list:
        if hasattr(self._plotter, 'group_names'):
            return self._plotter.group_names

        else:
            # _CategoricalPlotterNew
            group_names = self._plotter.var_levels[self._plotter.cat_axis]
            if isinstance(group_names, pd.Index):
                return group_names.tolist()
            return group_names

    @property
    def hue_names(self):
        if self.is_redundant_hue:
            return []

        if hasattr(self._plotter, 'hue_names'):
            # Can be None, force to be an empty list
            return self._plotter.hue_names or []

        else:
            if 'hue' not in self._plotter.var_levels:
                return []
            hue_names = self._plotter.var_levels['hue']
            if isinstance(hue_names, pd.Index):
                return hue_names.tolist()
            return hue_names

    def _order_variable(
        self,
        *,
        order,
        native_scale: bool = False,
        formatter: Callable | None = None,
    ) -> None:
        # Do not order if not categorical and native scale
        if self._plotter.var_types.get(self._plotter.cat_axis) != 'categorical' and native_scale:
            return

        # Order the group variable
        self._plotter.scale_categorical(self._plotter.cat_axis, order=order, formatter=None)

    def _get_group_data_from_plotter(
        self,
        cat: TGroupValue | tuple[TGroupValue],
    ) -> tuple[Any, int]:
        if isinstance(cat, tuple):
            cat = cat[0]
        index = self.group_names.index(cat)

        return self._plotter.plot_data[index], index

    def _get_group_data_with_hue(self, group_name: tuple):
        cat = group_name[0]
        group_data, index = self._get_group_data_from_plotter(cat)

        hue_level = group_name[1]
        hue_mask = self._plotter.plot_hues[index] == hue_level

        return group_data[hue_mask]

    def _get_group_data_without_hue(self, group_name):
        group_data, _ = self._get_group_data_from_plotter(group_name)

        return group_data

    def get_group_data(self, group_name):
        """Get the data for the (group[, hue]) tuple.

        group_name can be either a tuple ("cat",) or a tuple ("cat", "hue")

        Almost a duplicate of seaborn's code, because there is not
        direct access to the group_data in the respective Plotter class.
        """
        if isinstance(self._plotter, _CategoricalPlotterNew):
            return self._get_group_data_new(group_name)

        if not isinstance(group_name, tuple) or len(group_name) == 1:
            data = self._get_group_data_without_hue(group_name)
        else:
            data = self._get_group_data_with_hue(group_name)

        group_data = remove_null(data)

        return group_data

    def _get_group_data_new(self, group_name: TupleGroup) -> pd.Series:
        """Get the data for the (group[, hue]) tuple.

        group_name can be either a "cat" or a tuple ("cat", "hue")

        Almost a duplicate of seaborn's code, because there is not
        direct access to the group_data in the respective Plotter class.
        """
        if not isinstance(self._plotter, _CategoricalPlotterNew):
            msg = '`self._plotter` should be a `_CategoricalPlotterNew` instance.'
            raise TypeError(msg)

        # the value variable is the one that is no the group axis
        tgroup = group_name if isinstance(group_name, tuple) else (group_name,)
        # 'x' if vertical or 'y' if horizontal
        cat_var = self._plotter.cat_axis
        # opposite: 'y' if vertical or 'x' if horizontal
        value_var = {'x': 'y', 'y': 'x'}[cat_var]
        group = tgroup[0]
        hue = None
        iter_vars = [cat_var]
        if self.has_hue and len(tgroup) > 1:
            iter_vars.append('hue')
            hue = tgroup[1]

        for sub_vars, sub_data in self._plotter.iter_data(iter_vars):
            if sub_vars[cat_var] != group:
                continue
            if hue is not None and sub_vars['hue'] != hue:
                continue

            # Found a matching group, return the data
            group_data = remove_null(sub_data[value_var])
            return group_data


class CategoricalPlotterWrapper_v13(Wrapper):
    """Compatibility wrapper for seaborn v11."""

    _plotter: _CategoricalPlotter

    def __init__(self,
        plot_type: str,
        *,
        x,
        y,
        hue,
        data,
        order,
        **plot_params,
    ) -> None:
        super().__init__(plot_type, hue=hue, **plot_params)
        self._group_names = None

        variables = {'x': x, 'y': y, 'hue': hue}
        kwargs = {
            'data': data,
            'variables': variables,
            'order': order,
            'orient': plot_params.get('orient'),
            'color': None,
            'legend': plot_params.get('legend'),
        }
        self._plotter = _CategoricalPlotter(**kwargs)

        # Order the group variables
        native_scale = plot_params.get('native_scale', False)
        formatter = plot_params.get('formatter')
        self._order_variable(order=order, native_scale=native_scale, formatter=formatter)

        # Native scaling of the group variable
        if native_scale:
            self.native_group_offsets = self._plotter.plot_data[self.axis]

    @property
    def has_hue(self) -> bool:
        # Get variables mapping
        if self.is_redundant_hue:
            return False
        if self._plotter.variables.get('hue') is None:
            return False
        return True

    @property
    def axis(self) -> str:
        return {'x': 'x', 'v': 'x', 'y': 'y', 'h': 'y'}[self._plotter.orient]

    @property
    def group_names(self) -> list:
        if self._group_names is not None:
            return self._group_names
        return self._plotter.var_levels[self.axis]

    @property
    def hue_names(self):
        if not self.has_hue or 'hue' not in self._plotter.var_levels:
            return []
        return self._plotter.var_levels['hue']

    def _order_variable(
        self,
        *,
        order,
        native_scale: bool = False,
        formatter: Callable | None = None,
        raw_groups: bool = False,
    ) -> None:
        if raw_groups:
            # Save the group names before formatting, because they are transformed to str
            self._group_names = list(self._plotter.var_levels[self.axis])

        # Do not order if not categorical and native scale
        if self._plotter.var_types.get(self.axis) != 'categorical' and native_scale:
            return

        # Order the group variable
        self._plotter.scale_categorical(self.axis, order=order, formatter=formatter)

        if raw_groups:
            # Reorder group names
            formatter = formatter if callable(formatter) else str
            ordered_group_names = list(self._plotter.var_levels[self.axis])
            formatted_group_names = [formatter(v) for v in self._group_names]

            # find permutation indices
            indices = [ordered_group_names.index(val) for val in formatted_group_names]
            # Reorder raw group names
            self._group_names = [self._group_names[i] for i in indices]

    def get_group_data(self, group_name: TupleGroup) -> pd.Series:
        """Get the data for the (group[, hue]) tuple.

        group_name can be either a "cat" or a tuple ("cat", "hue")

        Almost a duplicate of seaborn's code, because there is not
        direct access to the group_data in the respective Plotter class.
        """
        # the value variable is the one that is not the group axis
        value_var = {'x': 'y', 'y': 'x'}[self.axis]
        tgroup = group_name if isinstance(group_name, tuple) else (group_name,)
        group = tgroup[0]
        hue = None
        iter_vars = [self.axis]
        if self.has_hue and len(tgroup) > 1:
            iter_vars.append('hue')
            hue = tgroup[1]

        for sub_vars, sub_data in self._plotter.iter_data(iter_vars):
            if sub_vars[self.axis] != group:
                continue
            if hue is not None and sub_vars['hue'] != hue:
                continue

            # Found a matching group, return the data
            group_data = remove_null(sub_data[value_var])
            return group_data

    def parse_pairs(
        self,
        pairs: list[tuple],
        structs: list[Struct],
        *,
        formatter: Callable | None = None,
    ) -> list[tuple[TupleGroup, TupleGroup]]:
        struct_groups = [struct['group'] for struct in structs]
        ret: list[tuple[TupleGroup, TupleGroup]] = []
        formatter = formatter if callable(formatter) else str

        def format_group(value, formatter) -> TupleGroup:
            """Format the group (but not the optional hue)."""
            tvalue = value if isinstance(value, tuple) else (value,)
            group = tvalue[0]
            return tuple([formatter(group), *tvalue[1:]])

        for pair in pairs:
            if not isinstance(pair, Sequence) or len(pair) != 2:
                msg = f'pair {pair} is not a 2-tuple, skipping.'
                warnings.warn(msg)
                continue
            # Format the groups
            new_pair = tuple(format_group(v, formatter) for v in pair)

            # Check that the groups are valid group names
            valid_group = True
            for i, group in enumerate(new_pair):
                if group not in struct_groups:
                    msg = (
                        f'cannot find group{i} of pair in the group tuples: '
                        f'{group} not in {struct_groups}'
                    )
                    warnings.warn(msg)
                    valid_group = False
            if not valid_group:
                continue

            ret.append(new_pair)

        if len(ret) == 0:
            msg = (
                f'pairs are empty after parsing: original_pairs={pairs}'
            )
            raise ValueError(msg)
        return ret



# Define CategoricalPlotterWrapper depending on seaborn version
if sns_version >= (0, 13, 0):
    CategoricalPlotterWrapper = CategoricalPlotterWrapper_v13
elif sns_version >= (0, 12, 0):
    CategoricalPlotterWrapper = CategoricalPlotterWrapper_v12
    # # CategoricalPlotterWrapper = CategoricalPlotterWrapper_v12
    # msg = f'Seaborn version {sns_version} is not supported.'
    # raise NotImplementedError(msg)
else:
    CategoricalPlotterWrapper = CategoricalPlotterWrapper_v11


def get_plotter(plot_type: str, **kwargs) -> CategoricalPlotterWrapper:
    return CategoricalPlotterWrapper(plot_type, **kwargs)
