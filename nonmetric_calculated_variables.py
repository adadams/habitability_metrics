from collections.abc import Callable

from dataset import map_function_arguments_to_dataset_variables
from model_variables import VariableAttrs
from xarray import DataArray, Dataset


# ------------------------------ Runoff ----------------------------
def calculate_runoff_from_precipitation_and_evaporation(
    precipitation: DataArray, evaporation: DataArray
) -> DataArray:
    return precipitation - evaporation


calculate_runoff_from_dataset: Callable[[Dataset], DataArray] = (
    map_function_arguments_to_dataset_variables(
        function=calculate_runoff_from_precipitation_and_evaporation,
        variable_mapping={"precipitation": "prec", "evaporation": "evap"},
    )
)
# ------------------------------------------------------------------

nonmetric_calculated_variables: dict[str, Callable[[Dataset], DataArray]] = {
    "runoff": calculate_runoff_from_dataset
}

nonmetric_variable_attrs: dict[str, VariableAttrs] = {
    "runoff": VariableAttrs(
        long_name="Runoff",
        plot_name="Runoff (mm/day)",
        units="mm/day",
        colormap="summer",
    ),
}
