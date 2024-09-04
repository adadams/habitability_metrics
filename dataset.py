from collections.abc import Callable, Sequence
from functools import partial, reduce
from inspect import signature
from pathlib import Path
from typing import Final

import numpy as np
import xarray as xr
from model_variables import VariableAttrs
from numpy.typing import NDArray
from pandas import MultiIndex

ESSENTIAL_VARIABLES: Final[list] = [
    "axyp",
    "ocnfr",
]  # always need cell area and ocean fraction


def compose(*functions: Sequence[Callable]) -> Callable:
    return reduce(lambda f, g: lambda x: g(f(x)), functions)


def build_dataset_from_multiple_files(
    model_output_filepaths: Sequence[Path],
    dimensions: xr.Dataset,
    variables: list[str],
    outputs_per_year: int = 12,
    month_length_weights: NDArray[np.float_] | None = None,
    case_name: str = "file_index",
    month_name: str = "record",
) -> xr.Dataset:
    def select_variables(
        dataset: xr.Dataset,
        selected_variables: list[str] = variables + ESSENTIAL_VARIABLES,
    ):
        return dataset.get(selected_variables)

    def rename_dimensions(
        dataset: xr.Dataset,
        possible_dimension_renames: dict[str, str] = {
            case_name: "case",
            month_name: "month",
        },
    ) -> xr.Dataset:
        dimension_renames: dict[str, str] = {
            original_dimension_name: new_dimension_name
            for original_dimension_name, new_dimension_name in possible_dimension_renames.items()
            if original_dimension_name in dataset.dims
        }

        return dataset.rename_dims(dimension_renames)

    def select_time_indices(
        dataset: xr.Dataset,
        time_unit: str = "month",
        time_slice: slice = slice(None, 360),
    ) -> xr.Dataset:
        return dataset.sel({time_unit: time_slice})

    apply_preprocessing_steps: Callable[[xr.Dataset], xr.Dataset] = compose(
        select_variables,
        rename_dimensions,
        select_time_indices,
    )

    def assign_dimensions(
        dataset: xr.Dataset, dimensions: xr.Dataset = dimensions
    ) -> xr.Dataset:
        return dataset.assign(dimensions)

    apply_postprocessing_steps: Callable[[xr.Dataset], xr.Dataset] = compose(
        assign_dimensions,
        partial(
            reshape_time_dimensions,
            outputs_per_year=outputs_per_year,
            month_length_weights=month_length_weights,
        ),
    )

    dataset: xr.Dataset = xr.open_mfdataset(
        model_output_filepaths,
        combine="nested",
        coords="different",
        concat_dim="case",
        preprocess=apply_preprocessing_steps,  # important because it pares down each file before concatenating
    )

    return apply_postprocessing_steps(dataset)


def build_dataset_from_single_file(
    variables: Sequence[str],
    model_output_filepaths: Path,
    dimensions: xr.Dataset,
    outputs_per_year: int = 12,
    month_length_weights: NDArray[np.float_] | None = None,
    case_name: str = "file_index",
    month_name: str = "record",
):
    dimension_renames = {case_name: "case", month_name: "month"}

    dataset = xr.open_dataset(model_output_filepaths)
    dataset = dataset.get(ESSENTIAL_VARIABLES + list(variables))

    dimension_renames: dict[str, str] = {
        original_dimension_name: new_dimension_name
        for original_dimension_name, new_dimension_name in dimension_renames.items()
        if original_dimension_name in dataset.dims
    }
    if dimension_renames:
        dataset = dataset.rename_dims(dimension_renames)

    dataset = dataset.assign(dimensions)
    dataset = dataset.sel(month=slice(None, 360))
    dataset = reshape_time_dimensions(dataset, outputs_per_year, month_length_weights)

    return dataset


def reshape_time_dimensions(data, outputs_per_year=12, month_length_weights=None):
    if ("month" in data.dims) or (("time" in data.dims) and outputs_per_year == 12):
        if "month" in data.dims:
            month_label = "month"
        else:
            month_label = "time"
        month = range(outputs_per_year)
        year = range(data.dims[month_label] // outputs_per_year)

        new_time_indices = MultiIndex.from_product(
            (year, month), names=("year", "month_in_year")
        )
        if "month" in data.dims:
            data = (
                data.assign(month=new_time_indices)
                .unstack(month_label)
                .rename({"month_in_year": "month"})
            )
        else:
            data = (
                data.assign(time=new_time_indices)
                .unstack(month_label)
                .rename({"month_in_year": "month"})
            )

    if month_length_weights is not None:
        month_length_weights = xr.DataArray(
            month_length_weights,
            coords={
                coord_name: coord_value
                for coord_name, coord_value in data.coords.items()
                if coord_name in ["case", "month"]
            },
            dims=["case", "month"],
        ).to_dataset(name="month_length_weights")

    return xr.merge([data, month_length_weights])


def add_variables_to_dataset(dataset: xr.Dataset, calculators: dict[str, Callable]):
    new_variables: dict[str, xr.DataArray] = {
        variable: calculator(dataset=dataset)
        for variable, calculator in calculators.items()
    }

    return dataset.assign(**new_variables)


def add_attributes(dataset: xr.Dataset, attributes: dict[str, VariableAttrs]):
    for variable, variable_attributes in attributes.items():
        try:
            dataset[variable] = dataset.get(variable).assign_attrs(
                **variable_attributes._asdict()
            )
        except AttributeError:
            print(f"Variable {variable} is not in the dataset. Skipping...")
            continue

    return dataset


def add_variables_and_attributes(
    dataset: xr.Dataset,
    calculators: dict[str, Callable],
    attributes: dict[str, VariableAttrs],
):
    return add_attributes(add_variables_to_dataset(dataset, calculators), attributes)


def load_dataset_and_add_calculated_variables(
    filepath: str,
    calculators: dict[str, Callable],
    attributes: dict[str, VariableAttrs],
):
    dataset = xr.load_dataset(filepath)

    return add_variables_and_attributes(dataset, calculators, attributes)


def get_getter_from_dataset(variable_name: str):
    return lambda dataset: dataset[variable_name]


def apply_to_variable(function: Callable, variable_name: str, dataset):
    return (function(get_getter_from_dataset(variable_name)))(dataset=dataset)


def map_function_of_function_arguments_to_dataset_variables(
    second_order_dataset_function, *dataset_functions
):
    """
    xr.Dataset functions are functions that take dataset variables as arguments.
    Sometimes they also take arguments that are functions of dataset variables themselves.
    That's what I mean by "second order".

    For dataset variables d1, d2, d3, ...
    and dataset functions f1(d1, d2, ...), f2(d1, d2, ...), ...,
    second_order_dataset_function = function(f1(...), f2(...), ..., d1, d2, ...)

    This will produce a function that is equivalent to the original second-order function
    but only needs the dataset as a single argument.
    """

    def apply_all_functions(*dataset_functions, second_order_dataset_function, dataset):
        purely_second_order_dataset_function = (
            partial(second_order_dataset_function, dataset=dataset)
            if "dataset" in signature(second_order_dataset_function).parameters.keys()
            else second_order_dataset_function
        )

        evaluated_dataset_functions = [
            dataset_function(dataset=dataset) for dataset_function in dataset_functions
        ]

        return purely_second_order_dataset_function(*evaluated_dataset_functions)

    return partial(
        apply_all_functions,
        *dataset_functions,
        second_order_dataset_function=second_order_dataset_function,
    )


def map_function_arguments_to_dataset_variables(function, variable_mapping):
    def function_using_dataset(*non_dataset_args, dataset, function, variable_mapping):
        dataset_kwargs = {
            kwarg: dataset.get(label) for kwarg, label in variable_mapping.items()
        }
        return function(*non_dataset_args, **dataset_kwargs)

    return partial(
        function_using_dataset, function=function, variable_mapping=variable_mapping
    )


def concatenate_datasets(*metrics_filepaths):
    datasets = (
        xr.load_dataset(metrics_filepath) for metrics_filepath in metrics_filepaths
    )

    combined_dataset = xr.concat(datasets, dim="case")
    combined_dataset = combined_dataset.sortby(combined_dataset.rotation_period)

    return combined_dataset


def get_global_area(dataset):
    cell_area_has_time_dimensions: bool = ("year" in dataset.axyp.dims) or (
        "month" in dataset.axyp.dims
    )
    cell_area = (
        take_time_average(
            dataset.axyp, month_length_weights=dataset.month_length_weights
        )
        if cell_area_has_time_dimensions
        else dataset.axyp
    )

    return cell_area


def get_ocean_percentage(dataset):
    ocean_fraction_has_time_dimensions = ("year" in dataset.ocnfr.dims) or (
        "month" in dataset.ocnfr.dims
    )
    ocean_fraction = (
        take_time_average(
            dataset.ocnfr, month_length_weights=dataset.month_length_weights
        )
        if ocean_fraction_has_time_dimensions
        else dataset.ocnfr
    )

    return ocean_fraction / 100


def get_land_area(dataset):
    ocean_percentage = get_ocean_percentage(dataset)
    cell_area = get_global_area(dataset)

    land_percentage = 1 - ocean_percentage
    return land_percentage * cell_area


def get_ocean_area(dataset):
    ocean_percentage = get_ocean_percentage(dataset)
    cell_area = get_global_area(dataset)

    return ocean_percentage * cell_area


def take_every_average(quantity, area_weights, month_length_weights):
    if "month" in area_weights.dims:
        area_weights = area_weights.weighted(month_length_weights).mean("month")
    if "year" in area_weights.dims:
        area_weights = area_weights.mean("year")

    return take_global_average(
        take_time_average(quantity, month_length_weights), area_weights
    )


def take_global_average(quantity, area_weights=None):
    quantity_weighted_by_area = (
        quantity.weighted(area_weights) if area_weights is not None else quantity
    )

    return quantity_weighted_by_area.mean(["lat", "lon"])


def take_annual_average(quantity, month_length_weights):
    quantity_weighted_by_month_length = quantity.weighted(month_length_weights)

    return quantity_weighted_by_month_length.mean("month")


def take_annual_sum(quantity, month_length_weights):
    quantity_weighted_by_month_length = quantity.weighted(month_length_weights)

    return quantity_weighted_by_month_length.sum("month")


def take_average_across_all_years(quantity: xr.DataArray):
    """That is, the average January across all years, the average Feb..."""
    return quantity.mean("year")


def take_time_average(quantity, month_length_weights):
    """A composition of the above time averages."""
    return take_annual_average(
        take_average_across_all_years(quantity), month_length_weights
    )


def average_dataset_over_globe(quantity_mapper):
    return map_function_of_function_arguments_to_dataset_variables(
        take_global_average, quantity_mapper, get_global_area
    )


def average_dataset_over_globe_and_time(quantity_mapper):
    return map_function_of_function_arguments_to_dataset_variables(
        map_function_arguments_to_dataset_variables(
            take_every_average, dict(month_length_weights="month_length_weights")
        ),
        quantity_mapper,
        get_global_area,
    )


def average_dataset_over_land(quantity_mapper):
    return map_function_of_function_arguments_to_dataset_variables(
        take_global_average, quantity_mapper, get_land_area
    )


def average_dataset_over_land_and_time(quantity_mapper):
    return map_function_of_function_arguments_to_dataset_variables(
        map_function_arguments_to_dataset_variables(
            take_every_average, dict(month_length_weights="month_length_weights")
        ),
        quantity_mapper,
        get_land_area,
    )


def average_dataset_over_ocean(quantity_mapper):
    return map_function_of_function_arguments_to_dataset_variables(
        take_global_average, quantity_mapper, get_ocean_area
    )


def average_dataset_over_ocean_and_time(quantity_mapper):
    return map_function_of_function_arguments_to_dataset_variables(
        map_function_arguments_to_dataset_variables(
            take_every_average, dict(month_length_weights="month_length_weights")
        ),
        quantity_mapper,
        get_ocean_area,
    )


def recast_eccentricity_dimensions_in_polar(dimensions):
    ecc_sin_lon = dimensions["eccentricity"] * np.sin(
        np.deg2rad(dimensions["longitude_at_periapse"])
    )
    ecc_cos_lon = dimensions["eccentricity"] * np.cos(
        np.deg2rad(dimensions["longitude_at_periapse"])
    )
    return {
        **{
            key: value
            for key, value in dimensions.items()
            if key not in ["eccentricity", "longitude_at_periapse"]
        },
        "ecc_sin_lon": ecc_sin_lon,
        "ecc_cos_lon": ecc_cos_lon,
    }
