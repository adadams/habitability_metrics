import numpy as np
from eccentric_configuration import (
    GPR_KWARGS,
    TEST_METRIC_FILEPATH,
    TRAINING_METRIC_FILEPATH,
    USE_POLAR,
)
from emulator import recast_eccentricity_dimensions_in_polar
from make_eccentric_emulators import (
    build_eccentric_emulator,
    get_eccentric_coordinates_as_dict,
)
from matplotlib import pyplot as plt
from plotting_functions import set_rotation_xaxis
from xarray import load_dataset


def unpack_dict_into_array(dict: dict):
    return np.asarray(list(dict.values())).T


def format_coordinates_for_emulator(coordinates):
    return np.vstack([coordinate for coordinate in coordinates.values()]).T


def predict_metrics_from_emulator(emulator, input_coordinates):
    predicted_test_metric, predicted_test_metric_uncertainty = (
        emulator.regressor.predict(
            format_coordinates_for_emulator(input_coordinates), return_std=True
        )
    )

    return dict(
        metric=predicted_test_metric,
        metric_uncertainty=predicted_test_metric_uncertainty,
    )


def calculate_residuals(predicted_value, directly_modeled_value, predicted_uncertainty):
    return (predicted_value - directly_modeled_value) / predicted_uncertainty


def calculate_normalized_polar_RMS_distances(training_coordinates, test_coordinates):
    polar_training_coordinates = recast_eccentricity_dimensions_in_polar(
        training_coordinates
    )
    polar_test_coordinates = recast_eccentricity_dimensions_in_polar(test_coordinates)

    edges_of_training_domain = unpack_dict_into_array(
        {
            dimension_name: np.array([np.min(dimension), np.max(dimension)])
            for dimension_name, dimension in polar_training_coordinates.items()
        }
    )

    normalized_training_coordinates = unpack_dict_into_array(
        polar_training_coordinates
    ) / np.ptp(edges_of_training_domain, axis=0)
    normalized_test_coordinates = unpack_dict_into_array(
        polar_test_coordinates
    ) / np.ptp(edges_of_training_domain, axis=0)

    return np.sqrt(
        np.sum(
            (
                normalized_training_coordinates[:, np.newaxis, :]
                - normalized_test_coordinates
            )
            ** 2,
            axis=(0, 2),
        )
        / len(normalized_test_coordinates)
    )


def plot_test_habitabilities_versus_emulator_predictions(
    predicted_habitabilities, test_coordinates, test_habitabilities
):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(
        2 ** test_coordinates["rotation_period"],
        predicted_habitabilities["metric"],
        predicted_habitabilities["metric_uncertainty"],
        linestyle="none",
        fmt="o",
        mfc="white",
        c="#388F52",
        label="Value predicted at test location",
    )
    ax.scatter(
        2 ** test_coordinates["rotation_period"],
        test_habitabilities,
        c="#388F52",
        label="Value of test model",
    )
    ax.set_xscale("log", base=2)
    set_rotation_xaxis(
        ax,
        rotation_period_limits=[
            np.min(2 ** test_coordinates["rotation_period"]),
            np.max(2 ** test_coordinates["rotation_period"]),
        ],
    )

    plt.legend(loc="lower left", fontsize=13)

    ax.set_xlabel(r"Rotation Period (days)")
    ax.set_ylabel("Habitability")

    fig.tight_layout()
    plt.savefig("habitability_test-vs-training_rotation_white-kernel.pdf")

    return fig, ax


def plot_residuals_versus_RMS_distances(
    training_coordinates,
    test_coordinates,
    predicted_habitabilities,
    test_habitabilities,
):
    fig = plt.figure(figsize=(6, 6))
    RMS_ax = fig.add_subplot(111)

    normalized_polar_RMS_distances = calculate_normalized_polar_RMS_distances(
        training_coordinates, test_coordinates
    )
    residuals = calculate_residuals(
        predicted_habitabilities["metric"],
        test_habitabilities,
        predicted_habitabilities["metric_uncertainty"],
    )
    print(
        f"There are {np.sum(np.abs(residuals.values)>1)} points more than 1 sigma away from their predictions, ",
        f"and {np.sum(np.abs(residuals.values)>2)} points more than 2 sigma away.",
    )

    RMS_scatter = RMS_ax.scatter(
        normalized_polar_RMS_distances,
        residuals,
        c=test_coordinates["rotation_period"],
        cmap=plt.cm.plasma,
        s=64,
    )

    RMS_ax.axhspan(-1, 1, color="#444444", alpha=0.33, zorder=0)
    # RMS_ax.set_ylim(-(ymax := RMS_ax.get_ylim()[1]), ymax)

    RMS_ax.set_xlabel(r"$d_\mathrm{RMS}$")
    RMS_ax.set_ylabel(
        r"$\left(H_\mathrm{pred} - H_\mathrm{GCM}\right) / \sigma_\mathrm{pred}$"
    )

    colorbar_ax = RMS_ax.inset_axes([1.05, 0, 0.05, 1], transform=RMS_ax.transAxes)

    colorbar = fig.colorbar(RMS_scatter, cax=colorbar_ax)
    colorbar_ax.set_yticks(np.arange(9))
    colorbar.ax.set_yticklabels(2 ** np.arange(9))
    colorbar.ax.set_ylabel("Rotation Period (days)")

    plt.savefig("residuals_versus_RMS_distance.pdf", bbox_inches="tight")

    return fig, (RMS_ax, colorbar_ax)


def run_plotting_routines(habitability_variable_name: str = "habitability_land"):
    training_metrics = load_dataset(TRAINING_METRIC_FILEPATH)
    test_metrics = load_dataset(TEST_METRIC_FILEPATH)

    training_habitability_emulator = build_eccentric_emulator(
        training_metrics, habitability_variable_name, use_polar=USE_POLAR, **GPR_KWARGS
    )

    training_coordinates = get_eccentric_coordinates_as_dict(training_metrics)
    test_coordinates = get_eccentric_coordinates_as_dict(test_metrics)

    predicted_test_habitabilities = predict_metrics_from_emulator(
        training_habitability_emulator, test_coordinates
    )
    test_habitabilities = test_metrics.get(habitability_variable_name)

    fig_1, ax_1 = plot_test_habitabilities_versus_emulator_predictions(
        predicted_test_habitabilities, test_coordinates, test_habitabilities
    )

    fig_2, axes_2 = plot_residuals_versus_RMS_distances(
        training_coordinates,
        test_coordinates,
        predicted_test_habitabilities,
        test_habitabilities,
    )

    return {
        "test_vs_training_plot": (fig_1, ax_1),
        "residuals_vs_RMS_plot": (fig_2, axes_2),
    }


if __name__ == "__main__":
    run_plotting_routines()
