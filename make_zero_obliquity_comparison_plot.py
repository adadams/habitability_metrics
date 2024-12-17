from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv
from plotting_customizations import plot_filetypes
from plotting_functions import set_rotation_xaxis
from user_filepaths import LOCAL_REPOSITORY_DIRECTORY
from xarray import load_dataset

JANSEN_RESULTS = np.array([0.76, 0.78, 0.80, 0.81, 0.92, 0.87, 0.78, 0.74, 0.71])
JANSEN_ROTATION_PERIODS = 2 ** np.arange(len(JANSEN_RESULTS))

PLOT_COLORS = {"all": "darkgreen", "ocean": "steelblue", "land": "palevioletred"}

CIRCULAR_METRIC_FILEPATH = LOCAL_REPOSITORY_DIRECTORY / "circular_metrics.nc"


def make_zero_obliquity_comparison_plot(
    circular_metric_filepath: Path,
    plot_output_directory: Path = LOCAL_REPOSITORY_DIRECTORY,
) -> list[plt.figure, plt.axis]:
    circular_metrics = load_dataset(circular_metric_filepath)

    zero_obl_rotation_periods = (
        2 ** circular_metrics.rotation_period[circular_metrics.obliquity == 0.0]
    )
    zero_obl_fTs = {
        "all": circular_metrics.fT_global[circular_metrics.obliquity == 0.0],
        "land": circular_metrics.fT_land[circular_metrics.obliquity == 0.0],
        "ocean": circular_metrics.fT_ocean[circular_metrics.obliquity == 0.0],
    }

    He_results = read_csv(LOCAL_REPOSITORY_DIRECTORY / "ROCKE_global_mean_data_all.csv")
    He_obliquities = He_results["obliquity"][1:].to_numpy().astype(int)
    He_zero_obl_rotation_periods = (
        (He_results["rotation period"][1:]).to_numpy().astype(float)
    )[He_obliquities == 0]
    He_zero_obl_fT = (
        (He_results["temperature habitability metric"][1:]).to_numpy().astype(float)
    )[He_obliquities == 0]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        JANSEN_ROTATION_PERIODS,
        JANSEN_RESULTS,
        color="coral",
        marker="s",
        s=64,
        label="Jansen et. al. (2019)",
    )

    ax.scatter(
        He_zero_obl_rotation_periods,
        He_zero_obl_fT,
        color="gold",
        marker="x",
        s=64,
        label="He et. al. (2022)",
        zorder=2,
    )

    for subset, values in zero_obl_fTs.items():
        ax.scatter(
            zero_obl_rotation_periods,
            values,
            color=PLOT_COLORS[subset],
            s=64,
            label=subset.capitalize(),
        )

    set_rotation_xaxis(ax)
    ax.set_xlabel("Rotation Period (days)")
    ax.set_ylabel(r"$f_\mathrm{T}$")
    ax.legend(loc="lower left", fontsize=13)

    fig.tight_layout()
    for filetype in plot_filetypes:
        plt.savefig(
            plot_output_directory / f"zero_obliquity_comparison_plus_He.{filetype}"
        )

    return fig, ax


if __name__ == "__main__":
    make_zero_obliquity_comparison_plot(CIRCULAR_METRIC_FILEPATH)
