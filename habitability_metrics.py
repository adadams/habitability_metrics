from functools import partial, reduce
from typing import Protocol

from astropy.units import W, deg_C, m, mm
from numpy import logical_and
from xarray import Dataset, apply_ufunc, broadcast

xarray_and = partial(apply_ufunc, logical_and)

from dataset import (  # noqa: E402
    average_dataset_over_globe,
    average_dataset_over_globe_and_time,
    average_dataset_over_land,
    average_dataset_over_land_and_time,
    average_dataset_over_ocean,
    average_dataset_over_ocean_and_time,
    map_function_arguments_to_dataset_variables,
    take_annual_sum,
    take_average_across_all_years,
)
from model_variables import VariableAttrs  # noqa: E402


class OperatesOnDataset(Protocol):
    def __call__(self, dataset: Dataset) -> Dataset: ...


#######################################################
"""These functions return indicator variables (i.e. 0 or 1) depending on some condition."""


def is_temperate(
    temperature, units=deg_C, lower_limit_C=0 * deg_C, upper_limit_C=100 * deg_C
):
    above_freezing = temperature > lower_limit_C.to(units).value
    below_boiling = temperature < upper_limit_C.to(units).value
    temperate = logical_and(above_freezing, below_boiling)

    return temperate


def accumulates_enough_rainfall(
    precipitation,  # mm/day
    units=mm,
    minimum_precipitation=300 * mm,
    number_of_days_in_a_year=365,
):
    enough_rainfall = (
        precipitation * number_of_days_in_a_year
        >= minimum_precipitation.to(units).value
    )

    return enough_rainfall


def accumulates_enough_runoff_moisture(
    precipitation,  # mm/day
    evaporation,  # mm/day
    units=mm,
    minimum_runoff=300 * mm,
    number_of_days_in_a_year=365,
):
    enough_runoff = (
        precipitation - evaporation
    ) * number_of_days_in_a_year >= minimum_runoff.to(units).value

    return enough_runoff


def gets_enough_sunlight(insolation, units=W / m**2, minimum_insolation=50 * W / m**2):
    enough_sunlight = insolation > minimum_insolation.to(units).value

    return enough_sunlight


def compose_indicators(*indicator_functions):
    def evaluate_composed_indicator(dataset, indicator_functions):
        indicators = broadcast(
            *[
                indicator_function(dataset=dataset)
                for indicator_function in indicator_functions
            ]
        )
        return reduce(xarray_and, indicators)

    return partial(evaluate_composed_indicator, indicator_functions=indicator_functions)


#######################################################

#######################################################
# ------------------ Temperature Metric ------------------
get_fT_indicators = map_function_arguments_to_dataset_variables(
    is_temperate, dict(temperature="tsurf")
)

get_fT_over_globe: OperatesOnDataset = average_dataset_over_globe_and_time(
    get_fT_indicators
)
get_fT_over_land: OperatesOnDataset = average_dataset_over_land_and_time(
    get_fT_indicators
)
get_fT_over_ocean: OperatesOnDataset = average_dataset_over_ocean_and_time(
    get_fT_indicators
)
# --------------------------------------------------------

# ----------------- Precipitation Metric -----------------
get_annual_precipitation = map_function_arguments_to_dataset_variables(
    take_annual_sum, dict(quantity="prec", month_length_weights="month_length_weights")
)


def get_fprec_indicators(dataset):
    return accumulates_enough_rainfall(get_annual_precipitation(dataset=dataset))


def get_time_averaged_fprec(dataset):
    return take_average_across_all_years(get_fprec_indicators(dataset=dataset))


get_fprec_over_globe: OperatesOnDataset = average_dataset_over_globe(
    get_time_averaged_fprec
)
get_fprec_over_land: OperatesOnDataset = average_dataset_over_land(
    get_time_averaged_fprec
)
get_fprec_over_ocean: OperatesOnDataset = average_dataset_over_ocean(
    get_time_averaged_fprec
)
# --------------------------------------------------------

# --------------------- Runoff Metric --------------------
get_annual_evaporation = map_function_arguments_to_dataset_variables(
    take_annual_sum, dict(quantity="evap", month_length_weights="month_length_weights")
)


def get_frunoff_indicators(dataset):
    return accumulates_enough_runoff_moisture(
        get_annual_precipitation(dataset=dataset),
        get_annual_evaporation(dataset=dataset),
    )


def get_time_averaged_frunoff(dataset):
    return take_average_across_all_years(get_frunoff_indicators(dataset=dataset))


get_frunoff_over_globe: OperatesOnDataset = average_dataset_over_globe(
    get_time_averaged_frunoff
)
get_frunoff_over_land: OperatesOnDataset = average_dataset_over_land(
    get_time_averaged_frunoff
)
get_frunoff_over_ocean: OperatesOnDataset = average_dataset_over_ocean(
    get_time_averaged_frunoff
)
# --------------------------------------------------------

# ----------- Temperature+Precipitation Metric -----------
get_habitability_indicators: OperatesOnDataset = compose_indicators(
    get_fT_indicators, get_fprec_indicators
)
get_habitability_over_globe: OperatesOnDataset = average_dataset_over_globe_and_time(
    get_habitability_indicators
)
get_habitability_over_land: OperatesOnDataset = average_dataset_over_land_and_time(
    get_habitability_indicators
)
get_habitability_over_ocean: OperatesOnDataset = average_dataset_over_ocean_and_time(
    get_habitability_indicators
)
# --------------------------------------------------------

# --------------- Temperature+Runoff Metric --------------
get_static_habitability_indicators: OperatesOnDataset = compose_indicators(
    get_fT_indicators, get_frunoff_indicators
)
get_static_habitability_over_globe: OperatesOnDataset = (
    average_dataset_over_globe_and_time(get_static_habitability_indicators)
)
get_static_habitability_over_land: OperatesOnDataset = (
    average_dataset_over_land_and_time(get_static_habitability_indicators)
)
get_static_habitability_over_ocean: OperatesOnDataset = (
    average_dataset_over_ocean_and_time(get_static_habitability_indicators)
)
# --------------------------------------------------------


habitability_metrics: dict[str, OperatesOnDataset] = {
    "fT_global": get_fT_over_globe,
    "fT_land": get_fT_over_land,
    "fT_ocean": get_fT_over_ocean,
    "fprec_global": get_fprec_over_globe,
    "fprec_land": get_fprec_over_land,
    "fprec_ocean": get_fprec_over_ocean,
    "habitability_global": get_habitability_over_globe,
    "habitability_land": get_habitability_over_land,
    "habitability_ocean": get_habitability_over_ocean,
    "static_habitability_global": get_static_habitability_over_globe,
    "static_habitability_land": get_static_habitability_over_land,
    "static_habitability_ocean": get_static_habitability_over_ocean,
}
#######################################################

#######################################################
habitability_attrs: dict[str, VariableAttrs] = {
    "fT_global": VariableAttrs(
        long_name="Temperature Habitability",
        plot_name=r"$f_\mathrm{T}$",
        units="",
        colormap="",
    ),
    "fT_land": VariableAttrs(
        long_name="Temperature Habitability over Land",
        plot_name=r"$f_\mathrm{T}$ (land)",
        units="",
        colormap="",
    ),
    "fT_ocean": VariableAttrs(
        long_name="Temperature Habitability over Ocean",
        plot_name=r"$f_\mathrm{T}$ (ocean)",
        units="",
        colormap="",
    ),
    "fprec_global": VariableAttrs(
        long_name="Precipitation Habitability",
        plot_name=r"$f_\mathrm{prec}$",
        units="",
        colormap="",
    ),
    "fprec_land": VariableAttrs(
        long_name="Precipitation Habitability over Land",
        plot_name=r"$f_\mathrm{prec}$ (land)",
        units="",
        colormap="",
    ),
    "fprec_ocean": VariableAttrs(
        long_name="Precipitation Habitability over Ocean",
        plot_name=r"$f_\mathrm{prec}$ (ocean)",
        units="",
        colormap="",
    ),
    "habitability_global": VariableAttrs(
        long_name="Climate Habitability",
        plot_name="Habitability",
        units="",
        colormap="",
    ),
    "habitability_land": VariableAttrs(
        long_name="Climate Habitability over Land",
        plot_name="Habitability (land)",
        units="",
        colormap="",
    ),
    "habitability_ocean": VariableAttrs(
        long_name="Climate Habitability over Ocean",
        plot_name="Habitability (ocean)",
        units="",
        colormap="",
    ),
    "static_habitability_global": VariableAttrs(
        long_name="Climate Habitability",
        plot_name="Habitability",
        units="",
        colormap="",
    ),
    "static_habitability_land": VariableAttrs(
        long_name="Climate Habitability over Land",
        plot_name="Habitability (land)",
        units="",
        colormap="",
    ),
    "static_habitability_ocean": VariableAttrs(
        long_name="Climate Habitability over Ocean",
        plot_name="Habitability (ocean)",
        units="",
        colormap="",
    ),
}
#######################################################
