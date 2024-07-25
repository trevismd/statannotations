#! /usr/bin/python
from __future__ import annotations

import argparse
import logging
import os
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pylab as plt
import seaborn as sns

from statannotations.Annotator import Annotator
from statannotations._Plotter import IMPLEMENTED_PLOTTERS

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import NotRequired, TypeAlias, TypedDict

    Pairs: TypeAlias = list[tuple[Any, Any]]

    class DataType(TypedDict):
        params: dict
        pairs: Pairs
        skip_plotters: NotRequired[list[str]]
        v0_13: NotRequired[bool]


logger = logging.getLogger(__name__)

seaborn_v0_11 = int(sns.__version__.split(".")[1]) <= 11
seaborn_v0_12 = int(sns.__version__.split(".")[1]) <= 12


def day_formatter(abbrev: str) -> str:
    """Format an abbreviated week day to full name."""
    if abbrev == "Thur":
        return "thursday"
    if abbrev == "Fri":
        return "friday"
    if abbrev == "Sat":
        return "saturday"
    if abbrev == "Sun":
        return "sunday"
    return abbrev.lower()


def make_plot(
    plotter: str,
    pairs: Pairs,
    params: Mapping[str, Any] | None = None,
    title: str = "",
    folder: os.PathLike | str = ".",
    extension: str = ".png",
    show: bool = False,
):
    """Make the Seaborn plot with parameters and annotate it."""
    params = params or {}
    fig, ax = plt.subplots()
    plot_func = getattr(sns, plotter)
    plot_func(ax=ax, **params)
    ax.set_title(title)
    annot = Annotator(ax, plot=plotter, pairs=pairs, **params)
    annot.configure(test='Mann-Whitney', text_format='star', loc='inside')
    annot.apply_and_annotate()

    positions = annot._plotter.groups_positions._data
    print(positions)
    if show:
        plt.show()
    else:
        filename = title + extension
        fig.savefig(os.path.join(folder, filename))


def make_data_types() -> dict[str, DataType]:
    """Import the dataframe and create the data types."""
    # Import dataset
    df = sns.load_dataset("tips")

    filtered_df = df.query("size in [2, 3, 5]")

    reduced_df = df.loc[:, ["total_bill", "day"]]

    # Params
    params_df = {
        "data": df,
        "x": "day",
        "y": "total_bill",
        "order": ["Sun", "Thur", "Fri", "Sat"],
        "hue_order": ["Male", "Female"],
        "hue": "sex",
        "dodge": True,
    }

    params_arrays = {
        "data": None,
        "x": df["day"],
        "y": df["total_bill"],
        "order": ["Sun", "Thur", "Fri", "Sat"],
        "hue_order": ["Male", "Female"],
        "hue": df["sex"],
        "dodge": True,
    }

    reduced_params_df_no_hue = {
        "data": reduced_df,
        "x": "day",
        "y": "total_bill",
        "order": ["Sun", "Thur", "Fri", "Sat"],
        "dodge": False,
    }

    reduced_params_df_with_hue = {
        "data": reduced_df,
        "x": "day",
        "y": "total_bill",
        "order": ["Sun", "Thur", "Fri", "Sat"],
        "hue": "day",
        "dodge": False,
    }

    native_scale_df = {
        "data": filtered_df,
        "x": "size",
        "y": "total_bill",
        "dodge": False,
    }

    native_scale_df_with_hue = {
        "data": filtered_df,
        "x": "size",
        "y": "total_bill",
        "hue": "sex",
        "dodge": True,
    }

    # Grouping
    group_hue_pairs = [
        (("Thur", "Male"), ("Thur", "Female")),
        (("Sun", "Male"), ("Sun", "Female")),
        (("Thur", "Male"), ("Fri", "Female")),
        (("Fri", "Male"), ("Fri", "Female")),
        (("Sun", "Male"), ("Fri", "Female")),
    ]

    hue_pairs = [
        (("Thur", "Male"), ("Thur", "Female")),
    ]

    group_pairs = [
        ("Thur", "Fri"),
        ("Thur", "Sun"),
        ("Sat", "Sun"),
    ]

    tuple_group_pairs = [
        (("Thur",), ("Fri",)),
        (("Thur",), ("Sun",)),
        (("Sat",), ("Sun",)),
    ]

    native_pairs = [
        (2, 3),
        (2, 5),
    ]

    native_pairs_tuples = [
        ((2,), (3,)),
        ((2,), (5,)),
    ]

    native_hue_pairs = [
        ((3, "Male"), (5, "Male")),
        ((2, "Male"), (2, "Female")),
    ]

    data_types: dict[str, DataType] = {
        "df_with_group_and_hue": {
            "params": params_df,
            "pairs": group_hue_pairs,
        },
        "arrays_with_group_and_hue": {
            "params": params_arrays,
            "pairs": group_hue_pairs,
        },
        "df_with_hue": {
            "params": params_df,
            "pairs": hue_pairs,
        },
        "df_with_group_and_hue_with_gap": {
            "params": {**params_df, "gap": 0.2},
            "pairs": group_hue_pairs,
            "skip_plotters": ["stripplot", "swarmplot"],
            "v0_13": True,
        },
        ## Error when plotting with Seaborn 0.13
        # "df_with_group_and_hue_with_formatter": {
        #     "params": {
        #         **params_df,
        #         "formatter": day_formatter,
        #     },
        #     "pairs": group_hue_pairs,
        #     "v0_13": True,
        # },
        "reduced_df_no_hue": {
            "params": reduced_params_df_no_hue,
            "pairs": group_pairs,
        },
        "reduced_df_no_hue_with_formatter": {
            "params": {
                **reduced_params_df_no_hue,
                "formatter": day_formatter,
            },
            "pairs": group_pairs,
            "v0_13": True,
        },
        "reduced_df_with_hue": {
            "params": reduced_params_df_with_hue,
            "pairs": group_pairs,
        },
        "reduced_df_no_hue_tuple_pairs": {
            "params": reduced_params_df_no_hue,
            "pairs": tuple_group_pairs,
        },
        "df_log_scale_with_group_and_hue": {
            "params": {**params_df, "log_scale": True},
            "pairs": group_hue_pairs,
            "v0_13": True,
        },
        "df_no_native_scale": {
            "params": {**native_scale_df, "order": [5, 3, 2]},
            "pairs": native_pairs,
        },
        "df_no_native_scale_tuples": {
            "params": native_scale_df,
            "pairs": native_pairs_tuples,
        },
        "df_no_native_scale_with_formatter": {
            "params": {
                **native_scale_df,
                "formatter": lambda x: str(float(x)),
            },
            "pairs": native_pairs,
            "v0_13": True,
        },
        "df_no_native_scale_with_hue": {
            "params": native_scale_df_with_hue,
            "pairs": native_hue_pairs,
        },
        "df_native_scale": {
            "params": {**native_scale_df, "native_scale": True},
            "pairs": native_pairs,
            "v0_13": True,
        },
        "df_native_scale_tuples": {
            "params": {**native_scale_df, "native_scale": True},
            "pairs": native_pairs_tuples,
            "v0_13": True,
        },
        "df_native_scale_with_formatter": {
            "params": {
                **native_scale_df,
                "native_scale": True,
                "formatter": lambda x: str(float(x)),
            },
            "pairs": native_pairs,
            "v0_13": True,
        },
        "df_native_scale_with_hue": {
            "params": {**native_scale_df_with_hue, "native_scale": True},
            "pairs": native_hue_pairs,
            "v0_13": True,
        },
    }
    return data_types


if __name__ == "__main__":
    # All plot types
    plot_types = IMPLEMENTED_PLOTTERS["seaborn"]

    # All data types
    data_types = make_data_types()

    # Command line argument parser
    parser = argparse.ArgumentParser(
        prog="start-video-annotator",
        description="GUI for annotation videos",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plots and don't save them to './figures/'.",
    )
    parser.add_argument(
        "--figure-extension",
        type=str,
        default=".png",
        help="Save the figures with the given extension.",
    )
    parser.add_argument(
        "-p",
        "--plot-type",
        action="append",
        type=str,
        choices=list(plot_types),
        help="Plot type to test.",
    )
    parser.add_argument(
        "-d",
        "--data-type",
        action="append",
        type=str,
        choices=list(data_types.keys()),
        help="Kind of data to test.",
    )

    args = parser.parse_args()
    print("Command line arguments:")
    print(vars(args))

    # Figure path
    figures_folder = Path(__file__).parent / "figures"
    if not os.path.isdir(figures_folder):
        os.mkdir(figures_folder)

    # Filter data types
    if args.data_type:
        data_types = {k: v for k, v in data_types.items() if k in args.data_type}

    # Filter plot types
    if args.plot_type:
        plot_types = [p for p in plot_types if p in args.plot_type]

    # Loop and make plots
    for groups_type, data in data_types.items():
        raw_params = data["params"]
        pairs = data["pairs"]
        only_seaborn_v0_13 = data.get("v0_13", False)
        skip_plotters = data.get("skip_plotters", [])
        if seaborn_v0_12 and only_seaborn_v0_13:
            continue

        for orient in ("v", "h", "x", "y"):
            if seaborn_v0_12 and orient in ("x", "y"):
                continue

            params = raw_params.copy()
            if seaborn_v0_12:
                # Make sure unsupported arguments are pruned
                params.pop("log_scale", None)
                params.pop("native_scale", None)

            if orient in ("h", "y"):
                # x = params["x"]
                # params["x"] = params["y"]
                # params["y"] = x
                params["x"], params["y"] = params["y"], params["x"]
            # update the parameters
            final_params = {**params, "orient": orient}

            for plotter in plot_types:
                if plotter in skip_plotters:
                    continue
                # Title
                title = f"Seaborn v{sns.__version__} - {plotter} ({orient}) - {groups_type}"

                print()
                print("+--------------------------------------------------------+")
                print(title)
                print("+--------------------------------------------------------+")
                print()

                try:
                    # Make plot
                    make_plot(
                        plotter,
                        pairs,
                        params=final_params,
                        title=title,
                        folder=figures_folder,
                        extension=args.figure_extension,
                        show=args.show,
                    )
                except Exception as e:  # noqa: E722
                    msg = f"Error plotting with the params:\n{final_params}"
                    logger.exception(msg)
                    # write error log to file
                    if not args.show:
                        filename = title + ".error.log"
                        filepath = os.path.join(figures_folder, filename)
                        with open(filepath, "w") as f:
                            f.write(msg)
                            f.write(str(e))
                            f.write(traceback.format_exc())
