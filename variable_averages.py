from collections.abc import Callable
from functools import partial

from dataset import (
    apply_to_variable,
    average_dataset_over_globe_and_time,
    average_dataset_over_land_and_time,
    average_dataset_over_ocean_and_time,
)
from model_variables import VariableAttrs
from xarray import DataArray, Dataset

# ------------------ Temperature Spatial Averages ------------------
global_average_T = partial(
    apply_to_variable,
    function=average_dataset_over_globe_and_time,
    variable_name="tsurf",
    result_variable_name="tsurf_global_average",
)

land_average_T = partial(
    apply_to_variable,
    function=average_dataset_over_land_and_time,
    variable_name="tsurf",
    result_variable_name="tsurf_land_average",
)

ocean_average_T = partial(
    apply_to_variable,
    function=average_dataset_over_ocean_and_time,
    variable_name="tsurf",
    result_variable_name="tsurf_ocean_average",
)
# ------------------------------------------------------------------

# ----------------- Precipitation Spatial Averages -----------------
global_average_prec = partial(
    apply_to_variable,
    function=average_dataset_over_globe_and_time,
    variable_name="prec",
    result_variable_name="prec_global_average",
)

land_average_prec = partial(
    apply_to_variable,
    function=average_dataset_over_land_and_time,
    variable_name="prec",
    result_variable_name="prec_land_average",
)

ocean_average_prec = partial(
    apply_to_variable,
    function=average_dataset_over_ocean_and_time,
    variable_name="prec",
    result_variable_name="prec_ocean_average",
)
# ------------------------------------------------------------------


# --------------------- Runoff Spatial Averages --------------------
global_average_runoff = partial(
    apply_to_variable,
    function=average_dataset_over_globe_and_time,
    variable_name="runoff",
    result_variable_name="runoff_global_average",
)

land_average_runoff = partial(
    apply_to_variable,
    function=average_dataset_over_land_and_time,
    variable_name="runoff",
    result_variable_name="runoff_land_average",
)

ocean_average_runoff = partial(
    apply_to_variable,
    function=average_dataset_over_ocean_and_time,
    variable_name="runoff",
    result_variable_name="runoff_ocean_average",
)
# ------------------------------------------------------------------

variable_averages: dict[str, Callable[[Dataset], DataArray]] = {
    "tsurf_global_average": global_average_T,
    "tsurf_land_average": land_average_T,
    "tsurf_ocean_average": ocean_average_T,
    "prec_global_average": global_average_prec,
    "prec_land_average": land_average_prec,
    "prec_ocean_average": ocean_average_prec,
    "runoff_global_average": global_average_runoff,
    "runoff_land_average": land_average_runoff,
    "runoff_ocean_average": ocean_average_runoff,
}

variable_averages_attrs: dict[str, VariableAttrs] = {
    "tsurf_global_average": VariableAttrs(
        long_name="Global average temperature",
        plot_name=r"$\bar{T}_\mathrm{surf}$ (${}^\circ$ C)",
        units="K",
        colormap="cmap_temperature",
    ),
    "tsurf_land_average": VariableAttrs(
        long_name="Land average temperature",
        plot_name=r"$\bar{T}_\mathrm{surf}$ (${}^\circ$ C)",
        units="K",
        colormap="cmap_temperature",
    ),
    "tsurf_ocean_average": VariableAttrs(
        long_name="Ocean average temperature",
        plot_name=r"$\bar{T}_\mathrm{surf}$ (${}^\circ$ C)",
        units="K",
        colormap="cmap_temperature",
    ),
    "prec_global_average": VariableAttrs(
        long_name="Global average precipitation",
        plot_name=r"$\bar{prec.}$ (${}^\circ$ C)",
        units="mm/day",
        colormap="cmap_precipitation",
    ),
    "prec_land_average": VariableAttrs(
        long_name="Land average precipitation",
        plot_name=r"$\bar{prec.}$ (${}^\circ$ C)",
        units="mm/day",
        colormap="cmap_precipitation",
    ),
    "prec_ocean_average": VariableAttrs(
        long_name="Ocean average precipitation",
        plot_name=r"$\bar{prec.}$ (${}^\circ$ C)",
        units="mm/day",
        colormap="cmap_precipitation",
    ),
    "runoff_global_average": VariableAttrs(
        long_name="Global average runoff",
        plot_name=r"$\bar{runoff}$ (mm/day)",
        units="mm/day",
        colormap="summer",
    ),
    "runoff_land_average": VariableAttrs(
        long_name="Land average runoff",
        plot_name=r"$\bar{runoff}$ (mm/day)",
        units="mm/day",
        colormap="summer",
    ),
    "runoff_ocean_average": VariableAttrs(
        long_name="Ocean average runoff",
        plot_name=r"$\bar{runoff}$ (mm/day)",
        units="mm/day",
        colormap="summer",
    ),
}
