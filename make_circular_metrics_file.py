import numpy as np
from dataset import add_attributes, load_dataset_and_add_calculated_variables
from model_variables import VariableAttrs, model_variables
from user_filepaths import LOCAL_PROCESSED_DATASET_DIRECTORY, LOCAL_REPOSITORY_DIRECTORY
from variable_averages import variable_averages, variable_averages_attrs
from xarray import merge

from habitability_metrics import habitability_attrs, habitability_metrics

INPUT_FILEPATH = LOCAL_PROCESSED_DATASET_DIRECTORY / "He_data.nc"
OUTPUT_FILEPATH = LOCAL_REPOSITORY_DIRECTORY / "circular_metrics.nc"

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
        filepath=input_filepath,
        calculators={**variable_averages, **habitability_metrics},
        attributes={
            **variable_averages_attrs,
            **model_variables,
            **habitability_attrs,
        },
    )
    circular_data = circular_data.set_coords(CIRCULAR_COORDINATES)
    circular_data["rotation_period"] = np.log2(circular_data["rotation_period"])

    circular_metrics = merge(
        [
            *[circular_data.get(variable) for variable in variable_averages],
            *[circular_data.get(metric) for metric in habitability_metrics],
        ]
    )
    circular_metrics = add_attributes(circular_metrics, CIRCULAR_COORDINATES)

    if overwrite or not output_filepath.is_file():
        circular_metrics.to_netcdf(output_filepath)

    return circular_metrics


if __name__ == "__main__":
    circular_metrics = create_circular_metrics_file()
    print(circular_metrics)
