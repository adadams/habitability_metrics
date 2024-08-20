from pathlib import Path

import numpy as np
from emulator import build_emulator
from matplotlib import pyplot as plt
from plotting_customizations import (
    cmap_habitability,
    cmap_precipitation,
    cmap_temperature,
    plot_filetypes,
)
from plotting_functions import plot_noneccentric_emulator_grid
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from xarray import load_dataset

RBF_KERNEL_PARAMETERS = dict(
    length_scale=[1, 15], length_scale_bounds=[[1e-10, 1e10], [1e-10, 1e10]]
)
WHITE_KERNEL_PARAMETERS = dict(noise_level=0.01, noise_level_bounds=[1e-6, 1e6])
GAUSSIAN_PROCESS_REGRESSION_PARAMETERS = dict(alpha=1e-3, normalize_y=True)
GPR_KWARGS = {
    **dict(
        kernel=RBF(**RBF_KERNEL_PARAMETERS) + WhiteKernel(**WHITE_KERNEL_PARAMETERS)
    ),
    **GAUSSIAN_PROCESS_REGRESSION_PARAMETERS,
}

CIRCULAR_METRIC_FILEPATH = Path("circular_metrics.nc")


def build_circular_emulators(circular_metrics, emulated_variable_names, **GPR_kwargs):
    circular_coordinates = {
        coordinate_name: circular_metrics[coordinate_name].to_numpy()
        for coordinate_name in ["rotation_period", "obliquity"]
    }

    emulators = []
    for emulated_variable_name in emulated_variable_names:
        emulator = build_emulator(
            dimensions=circular_coordinates,
            emulated_variable=circular_metrics.get(emulated_variable_name),
            use_polar=False,
            **GPR_kwargs,
        )
        print(f"{emulated_variable_name}: {emulator.regressor.kernel_=}")
        emulators.append(emulator)

    return *emulators, circular_coordinates


def plot_emulated_grid(
    fT_emulator,
    fprec_emulator,
    habitability_emulator,
    grid_dimensions,
    grid_resolution=100,
):
    dimension_bounds = {
        dimension_name: [np.min(dimension), np.max(dimension)]
        for dimension_name, dimension in grid_dimensions.items()
    }

    grid_spacings = {
        dimension_name: np.linspace(*dimension_bound, num=grid_resolution)
        for dimension_name, dimension_bound in dimension_bounds.items()
    }

    circular_shared_grid_kwargs = dict(
        dimensions=grid_dimensions,
        grid_spacings=list(grid_spacings.values()),
        subplot_kwargs=dict(figsize=(6, 6)),
    )

    circular_fT_grid_kwargs = dict(
        emulator=fT_emulator,
        emulated_print_name=r"$f_\mathrm{T}$",
        emulated_save_name="fT_circular",
        colormap=cmap_temperature,
    )

    circular_fprec_grid_kwargs = dict(
        emulator=fprec_emulator,
        emulated_print_name=r"$f_\mathrm{prec}$",
        emulated_save_name="fprec_circular",
        colormap=cmap_precipitation,
    )

    circular_habitability_grid_kwargs = dict(
        emulator=habitability_emulator,
        emulated_print_name="Habitability",
        emulated_save_name="habitability_circular",
        colormap=cmap_habitability,
    )

    circular_grid_kwargs = [
        circular_fT_grid_kwargs,
        circular_fprec_grid_kwargs,
        circular_habitability_grid_kwargs,
    ]

    fig, axes = plt.subplots(1, len(circular_grid_kwargs), figsize=(18, 7), sharey=True)

    for i, (ax, grid_kwargs) in enumerate(zip(axes, circular_grid_kwargs)):
        plot_noneccentric_emulator_grid(
            **circular_shared_grid_kwargs, **grid_kwargs, fig=fig, ax=ax
        )
        if i > 0:
            ax.set_ylabel("")

    fig.tight_layout()

    for filetype in plot_filetypes:
        plt.savefig(f"circular_emulated_grids.{filetype}", bbox_inches="tight", dpi=300)

    return fig, axes


def make_circular_emulator_and_plot(
    circular_metrics,
    emulated_variables=["fT_land", "fprec_land", "habitability_land"],
    **GPR_kwargs,
):
    return plot_emulated_grid(
        *build_circular_emulators(circular_metrics, emulated_variables, **GPR_kwargs)
    )


def run_for_circular_metrics():
    circular_metrics = load_dataset(CIRCULAR_METRIC_FILEPATH)
    make_circular_emulator_and_plot(circular_metrics, **GPR_KWARGS)

    return circular_metrics


if __name__ == "__main__":
    run_for_circular_metrics()
