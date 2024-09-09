import numpy as np
from dataset import concatenate_datasets
from eccentric_configuration import (
    COORDINATE_NAMES,
    COORDINATE_PRINT_NAMES,
    TEST_METRIC_FILEPATH,
    TRAINING_METRIC_FILEPATH,
)
from matplotlib import pyplot as plt
from plotting_customizations import plot_filetypes
from plotting_functions import plot_scatterplot, set_rotation_xaxis


def plot_metric_scatterplots(
    *metrics_filepaths,
    dimension_list: list[str] = COORDINATE_NAMES,
    temperature_variable_name: str = "tsurf_land_average",
    precipitation_variable_name: str = "prec_land_average",
) -> dict[str, plt.Figure]:
    dataset = concatenate_datasets(*metrics_filepaths)

    dimensions = {
        dimension_name: dataset.get(dimension_name).to_numpy()
        for dimension_name in dimension_list
    }

    temperature_scatterplots = plot_scatterplot(
        dimensions,
        COORDINATE_PRINT_NAMES,
        dataset.get(temperature_variable_name),
        [r"Surface Temperature (${}^\circ$ C)"],
        scatter_kwargs=dict(s=64, c="#F5425A"),
        filetypes=plot_filetypes,
    )

    precipitation_scatterplots = plot_scatterplot(
        dimensions,
        COORDINATE_PRINT_NAMES,
        dataset.get(precipitation_variable_name),
        ["Precipitation (mm/day)"],
        scatter_kwargs=dict(s=64, c="#0A89B8"),
        filetypes=plot_filetypes,
    )

    return {
        "tsurf_scatterplots": temperature_scatterplots,
        "prec_scatterplots": precipitation_scatterplots,
    }


def plot_combined_scatterplots(
    *metrics_filepaths,
    dimension_list=COORDINATE_NAMES,
    temperature_variable_name="tsurf_land_average",
    precipitation_variable_name="prec_land_average",
    temperature_habitability_variable_name="fT_land",
    precipitation_habitability_variable_name="fprec_land",
    climate_habitability_variable_name="habitability_land",
):
    dataset = concatenate_datasets(*metrics_filepaths)

    all_dimensions = {
        dimension_name: dataset.get(dimension_name).to_numpy()
        for dimension_name in dimension_list
    }

    fig, axes = plt.subplots(2, 3, figsize=(20, 8))

    quantities = [
        dataset.get(temperature_variable_name),
        dataset.get(precipitation_variable_name),
        np.zeros_like(dataset.get(temperature_variable_name)),
        dataset.get(temperature_habitability_variable_name),
        dataset.get(precipitation_habitability_variable_name),
        dataset.get(climate_habitability_variable_name),
    ]
    quantity_plot_names = [
        r"Surface Temperature (${}^\circ$ C)",
        "Precipitation (mm/day)",
        None,
        r"$f_\mathrm{T}$",
        r"$f_\mathrm{prec}$",
        "Habitability",
    ]
    y_min = np.min(np.concatenate(quantities[3:]))
    y_max = np.max(np.concatenate(quantities[3:]))

    for i, (ax, quantity, quantity_plot_name) in enumerate(
        zip(axes.flatten(), quantities, quantity_plot_names)
    ):
        if quantity_plot_name is None:
            continue
        scatterplot = ax.scatter(
            2 ** all_dimensions["rotation_period"],
            quantity,
            c=all_dimensions["obliquity"],
            s=64,
            cmap=plt.cm.viridis,
        )
        set_rotation_xaxis(
            ax,
            rotation_period_limits=2
            ** np.array(
                [
                    np.min(all_dimensions["rotation_period"]),
                    np.max(all_dimensions["rotation_period"]),
                ]
            ),
        )
        if i > 2:
            y_buffer = 0.05 * (y_max - y_min)
            ax.set_ylim([y_min - y_buffer, y_max + y_buffer])
            ax.set_xlabel("Rotation Period (days)")
        ax.set_ylabel(quantity_plot_name)

    fig.colorbar(
        scatterplot,
        ax=axes[0, 2],
        location="top",
        orientation="horizontal",
        fraction=0.7,
        aspect=6,
        label=r"Obliquity $\left(^\circ\right)$",
    )
    axes[0, 2].axis("off")

    for filetype in plot_filetypes:
        plt.savefig(
            f"scatterplots_all-data_obliquity-colored.{filetype}",
            bbox_inches="tight",
            dpi=150,
        )

    return fig, axes


if __name__ == "__main__":
    plot_metric_scatterplots(TRAINING_METRIC_FILEPATH, TEST_METRIC_FILEPATH)
    plot_combined_scatterplots(TRAINING_METRIC_FILEPATH, TEST_METRIC_FILEPATH)
