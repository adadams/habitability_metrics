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

INPUT_FILEPATH = LOCAL_PROCESSED_DATASET_DIRECTORY / "He_data.nc"
OUTPUT_FILEPATH = Path("circular_metrics.nc")

CIRCULAR_COORDINATES = {
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
}


def create_circular_metrics_file(
    input_filepath=INPUT_FILEPATH, output_filepath=OUTPUT_FILEPATH, overwrite=True
):
    circular_data = load_dataset_and_add_calculated_variables(
        input_filepath, habitability_metrics, {**model_variables, **habitability_attrs}
    )
    circular_data = circular_data.set_coords(CIRCULAR_COORDINATES)
    circular_data["rotation_period"] = np.log2(circular_data["rotation_period"])

    global_average_T = apply_to_variable(
        average_dataset_over_globe_and_time, "tsurf", circular_data
    ).rename("tsurf_global_average")
    land_average_T = apply_to_variable(
        average_dataset_over_land_and_time, "tsurf", circular_data
    ).rename("tsurf_land_average")
    ocean_average_T = apply_to_variable(
        average_dataset_over_ocean_and_time, "tsurf", circular_data
    ).rename("tsurf_ocean_average")

    global_average_prec = apply_to_variable(
        average_dataset_over_globe_and_time, "prec", circular_data
    ).rename("prec_global_average")
    land_average_prec = apply_to_variable(
        average_dataset_over_land_and_time, "prec", circular_data
    ).rename("prec_land_average")
    ocean_average_prec = apply_to_variable(
        average_dataset_over_ocean_and_time, "prec", circular_data
    ).rename("prec_ocean_average")

    circular_metrics = merge(
        [
            global_average_T,
            land_average_T,
            ocean_average_T,
            global_average_prec,
            land_average_prec,
            ocean_average_prec,
            circular_data.fT_global,
            circular_data.fT_land,
            circular_data.fT_ocean,
            circular_data.fprec_global,
            circular_data.fprec_land,
            circular_data.fprec_ocean,
            circular_data.habitability_global,
            circular_data.habitability_land,
            circular_data.habitability_ocean,
        ]
    )
    circular_metrics = add_attributes(circular_metrics, CIRCULAR_COORDINATES)

    if overwrite or not output_filepath.is_file():
        circular_metrics.to_netcdf(output_filepath)

    return circular_metrics


if __name__ == "__main__":
    circular_metrics = create_circular_metrics_file()
    print(circular_metrics)
