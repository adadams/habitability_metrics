import numpy as np
from dataset import add_attributes, load_dataset_and_add_calculated_variables
from model_variables import VariableAttrs, model_variables
from user_filepaths import LOCAL_PROCESSED_DATASET_DIRECTORY, LOCAL_REPOSITORY_DIRECTORY
from variable_averages import (
    variable_averages,
    variable_averages_attrs,
)
from xarray import merge

from habitability_metrics import habitability_attrs, habitability_metrics

TRAINING_INPUT_FILEPATH = LOCAL_PROCESSED_DATASET_DIRECTORY / "LHSTR_data.nc"
TRAINING_OUTPUT_FILEPATH = LOCAL_REPOSITORY_DIRECTORY / "training_metrics.nc"

TEST_INPUT_FILEPATH = LOCAL_PROCESSED_DATASET_DIRECTORY / "LHSTE_data.nc"
TEST_OUTPUT_FILEPATH = LOCAL_REPOSITORY_DIRECTORY / "test_metrics.nc"

ECCENTRIC_COORDINATES = {
    "rotation_period": VariableAttrs(
        long_name="Rotation Period",
        plot_name="Rotation Period (days)",
        units="",
        colormap="",
    ),
    "obliquity": VariableAttrs(
        long_name="Obliquity",
        plot_name=r"Obliquity ($^\circ$)",
        units="deg",
        colormap="",
    ),
    "eccentricity": VariableAttrs(
        long_name="Eccentricity", plot_name=r"$e$", units="", colormap=""
    ),
    "longitude_at_periapse": VariableAttrs(
        long_name="Longitude at Periapse",
        plot_name=r"$\phi_\mathrm{peri}$ $\left(^\circ\right)$",
        units="deg",
        colormap="",
    ),
}


def create_eccentric_metrics_file(input_filepath, output_filepath, overwrite=True):
    eccentric_data = load_dataset_and_add_calculated_variables(
        filepath=input_filepath,
        calculators={**variable_averages, **habitability_metrics},
        attributes={
            **model_variables,
            **variable_averages_attrs,
            **habitability_attrs,
        },
    )
    eccentric_data = eccentric_data.set_coords(ECCENTRIC_COORDINATES)
    eccentric_data["rotation_period"] = np.log2(eccentric_data["rotation_period"])

    eccentric_metrics = merge(
        [
            *[eccentric_data.get(variable) for variable in variable_averages],
            *[eccentric_data.get(metric) for metric in habitability_metrics],
        ]
    )
    eccentric_metrics = add_attributes(eccentric_metrics, ECCENTRIC_COORDINATES)

    if overwrite or not output_filepath.is_file():
        eccentric_metrics.to_netcdf(output_filepath)

    return eccentric_metrics


if __name__ == "__main__":
    training_metrics = create_eccentric_metrics_file(
        TRAINING_INPUT_FILEPATH, TRAINING_OUTPUT_FILEPATH
    )
    test_metrics = create_eccentric_metrics_file(
        TEST_INPUT_FILEPATH, TEST_OUTPUT_FILEPATH
    )

    print(training_metrics)
    print(test_metrics)
