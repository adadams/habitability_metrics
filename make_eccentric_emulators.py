import numpy as np
from dataset import recast_eccentricity_dimensions_in_polar
from eccentric_configuration import (
    COORDINATE_NAMES,
    COORDINATE_NAMES_FOR_GRID,
    COORDINATE_PRINT_NAMES,
    GPR_KWARGS,
    MAXIMUM_ECCENTRICITY,
    TEST_METRIC_FILEPATH,
    TRAINING_METRIC_FILEPATH,
    USE_POLAR,
)
from emulator import build_emulator
from plotting_customizations import (
    cmap_habitability,
    cmap_precipitation,
    cmap_temperature,
)
from plotting_functions import plot_grid_of_grids
from xarray import concat, load_dataset

NUMBER_OF_ROWS = 5
NUMBER_OF_COLUMNS = 5

GRID_RESOLUTION = 100


def get_eccentric_coordinates_as_dict(
    eccentric_metrics, coordinate_names=COORDINATE_NAMES
):
    return {
        coordinate_name: eccentric_metrics[coordinate_name].to_numpy()
        for coordinate_name in coordinate_names
    }


def build_eccentric_emulator(
    eccentric_metrics, emulated_variable_name, use_polar, **GPR_kwargs
):
    eccentric_coordinates = get_eccentric_coordinates_as_dict(eccentric_metrics)

    emulator = build_emulator(
        dimensions=eccentric_coordinates,
        emulated_variable=eccentric_metrics.get(emulated_variable_name).to_numpy(),
        use_polar=use_polar,
        **GPR_kwargs,
    )
    print(f"{emulated_variable_name}: {emulator.regressor.kernel_=}")

    return emulator


def build_eccentric_emulators(
    eccentric_metrics, emulated_variable_names, use_polar, **GPR_kwargs
):
    return [
        build_eccentric_emulator(
            eccentric_metrics, emulated_variable_name, use_polar, **GPR_kwargs
        )
        for emulated_variable_name in emulated_variable_names
    ]


def plot_emulated_grid(
    fT_emulator,
    fprec_emulator,
    habitability_emulator,
    grid_dimensions,
    filename_suffix,
    grid_resolution=GRID_RESOLUTION,
):
    dimension_bounds = {
        dimension_name: [np.min(dimension), np.max(dimension)]
        for dimension_name, dimension in grid_dimensions.items()
    }

    if "ecc_cos_lon" in dimension_bounds and "ecc_sin_lon" in dimension_bounds:
        eccentricity_bounds = [-MAXIMUM_ECCENTRICITY, MAXIMUM_ECCENTRICITY]
        dimension_bounds["ecc_cos_lon"] = eccentricity_bounds
        dimension_bounds["ecc_sin_lon"] = eccentricity_bounds

    grid_spacings = {
        dimension_name: np.linspace(*dimension_bound, num=grid_resolution)
        for dimension_name, dimension_bound in dimension_bounds.items()
    }

    eccentric_shared_grid_kwargs = dict(
        dimensions=grid_dimensions,
        plotted_dimension_names=COORDINATE_NAMES_FOR_GRID[
            :2
        ],  # rotation period, obliquity
        plotted_print_labels=COORDINATE_PRINT_NAMES[:2],
        fixed_dimension_names=COORDINATE_NAMES_FOR_GRID[
            2:
        ],  # e cos phi, e sin phi OR eccentricity, longitude at periapse
        fixed_print_labels=COORDINATE_PRINT_NAMES[2:],
        grid_spacings=grid_spacings,
        subplot_kwargs={
            "nrows": NUMBER_OF_ROWS,
            "ncols": NUMBER_OF_COLUMNS,
            "figsize": (3 * NUMBER_OF_ROWS, 3 * NUMBER_OF_COLUMNS),
        },
    )

    eccentric_fT_grid_kwargs = dict(
        emulator=fT_emulator,
        actual_values=fT_emulator.emulated,
        emulated_print_name=r"$f_\mathrm{T}$",
        emulated_save_name=f"fT_{filename_suffix}",
        colormap=cmap_temperature,
    )

    eccentric_fprec_grid_kwargs = dict(
        emulator=fprec_emulator,
        actual_values=fprec_emulator.emulated,
        emulated_print_name=r"$f_\mathrm{prec}$",
        emulated_save_name=f"fprec_{filename_suffix}",
        colormap=cmap_precipitation,
    )

    eccentric_habitability_grid_kwargs = dict(
        emulator=habitability_emulator,
        actual_values=habitability_emulator.emulated,
        emulated_print_name="Habitability",
        emulated_save_name=f"habitability_{filename_suffix}",
        colormap=cmap_habitability,
    )

    eccentric_grid_kwargs = [
        eccentric_fT_grid_kwargs,
        eccentric_fprec_grid_kwargs,
        eccentric_habitability_grid_kwargs,
    ]
    contour_kwargs = dict(levels=0.1 * np.arange(0, 11))
    scatter_kwargs = dict(vmin=0, vmax=1)

    for grid_kwargs in eccentric_grid_kwargs:
        fig, axes = plot_grid_of_grids(
            **eccentric_shared_grid_kwargs,
            **grid_kwargs,
            contour_kwargs=contour_kwargs,
            scatter_kwargs=scatter_kwargs,
        )
        errors_fig, errors_axes = plot_grid_of_grids(
            **eccentric_shared_grid_kwargs,
            **grid_kwargs,
            plot_errors=True,
            scatter_kwargs=scatter_kwargs,
        )

    return fig, axes, errors_fig, errors_axes


def make_eccentric_emulators_and_plot(
    eccentric_metrics,
    filename_suffix,
    emulated_variable_names=["fT_land", "fprec_land", "habitability_land"],
    use_polar=USE_POLAR,
):
    eccentric_coordinates = get_eccentric_coordinates_as_dict(eccentric_metrics)
    eccentric_coordinates_for_grid = (
        recast_eccentricity_dimensions_in_polar(eccentric_coordinates)
        if use_polar
        else eccentric_coordinates
    )

    return plot_emulated_grid(
        *build_eccentric_emulators(
            eccentric_metrics, emulated_variable_names, use_polar, **GPR_KWARGS
        ),
        eccentric_coordinates_for_grid,
        filename_suffix,
    )


def run_for_training_and_all_metrics():
    training_metrics = load_dataset(TRAINING_METRIC_FILEPATH)
    test_metrics = load_dataset(TEST_METRIC_FILEPATH)

    all_metrics = concat((training_metrics, test_metrics), dim="case")
    all_metrics = all_metrics.sortby(all_metrics.rotation_period)

    make_eccentric_emulators_and_plot(training_metrics, filename_suffix="training")
    make_eccentric_emulators_and_plot(all_metrics, filename_suffix="all")


if __name__ == "__main__":
    run_for_training_and_all_metrics()
