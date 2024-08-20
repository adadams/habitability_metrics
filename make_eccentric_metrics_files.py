from pathlib import Path

import numpy as np
from dataset import (
    add_attributes,
    apply_to_variable,
    average_dataset_over_globe_and_time,
    average_dataset_over_land_and_time,
    average_dataset_over_ocean_and_time,
    load_dataset_and_add_calculated_variables,
)
from model_variables import VariableAttrs, model_variables
from user_filepaths import LOCAL_PROCESSED_DATASET_DIRECTORY
from xarray import merge

from habitability_metrics import habitability_attrs, habitability_metrics

TRAINING_INPUT_FILEPATH = LOCAL_PROCESSED_DATASET_DIRECTORY / "LHSTR_data.nc"
TRAINING_OUTPUT_FILEPATH = Path("training_metrics.nc")

TEST_INPUT_FILEPATH = LOCAL_PROCESSED_DATASET_DIRECTORY / "LHSTE_data.nc"
TEST_OUTPUT_FILEPATH = Path("test_metrics.nc")

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
        input_filepath, habitability_metrics, {**model_variables, **habitability_attrs}
    )
    eccentric_data = eccentric_data.set_coords(ECCENTRIC_COORDINATES)
    eccentric_data["rotation_period"] = np.log2(eccentric_data["rotation_period"])

    global_average_T = apply_to_variable(
        average_dataset_over_globe_and_time, "tsurf", eccentric_data
    ).rename("tsurf_global_average")
    land_average_T = apply_to_variable(
        average_dataset_over_land_and_time, "tsurf", eccentric_data
    ).rename("tsurf_land_average")
    ocean_average_T = apply_to_variable(
        average_dataset_over_ocean_and_time, "tsurf", eccentric_data
    ).rename("tsurf_ocean_average")

    global_average_prec = apply_to_variable(
        average_dataset_over_globe_and_time, "prec", eccentric_data
    ).rename("prec_global_average")
    land_average_prec = apply_to_variable(
        average_dataset_over_land_and_time, "prec", eccentric_data
    ).rename("prec_land_average")
    ocean_average_prec = apply_to_variable(
        average_dataset_over_ocean_and_time, "prec", eccentric_data
    ).rename("prec_ocean_average")

    eccentric_metrics = merge(
        [
            global_average_T,
            land_average_T,
            ocean_average_T,
            global_average_prec,
            land_average_prec,
            ocean_average_prec,
            eccentric_data.fT_global,
            eccentric_data.fT_land,
            eccentric_data.fT_ocean,
            eccentric_data.fprec_global,
            eccentric_data.fprec_land,
            eccentric_data.fprec_ocean,
            eccentric_data.habitability_global,
            eccentric_data.habitability_land,
            eccentric_data.habitability_ocean,
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
