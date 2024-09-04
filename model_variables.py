from typing import NamedTuple, Optional

from matplotlib.colors import Colormap


class VariableAttrs(NamedTuple):
    long_name: str
    plot_name: str
    units: str
    colormap: Optional[Colormap]


#######################################################
model_variables = {
    "tsurf": VariableAttrs(
        long_name="Surface Temperature",
        plot_name=r"$T_\mathrm{surf}^\circ$C",
        units="C",
        colormap="cmap_temperature",
    ),
    "prec": VariableAttrs(
        long_name="Rainfall",
        plot_name="Precipitation (mm/day)",
        units="mm/day",
        colormap="cmap_precipitation",
    ),
    "evap": VariableAttrs(
        long_name="Evaporation",
        plot_name="Evaporation (mm/day)",
        units="mm/day",
        colormap="summer",
    ),
    "snowicefr": VariableAttrs(
        long_name="Snow/Ice Cover",
        plot_name="Snow/Ice Cover (\%)",
        units="\%",
        colormap="blues_r",
    ),
    "pcldt": VariableAttrs(
        long_name="Cloud Cover",
        plot_name="Cloud Cover (\%)",
        units="\%",
        colormap="gray",
    ),
    "incsw_toa": VariableAttrs(
        long_name="Insolation",
        plot_name="Insolation (W/m$^2$)",
        units="W/m^2",
        colormap="bmy",
    ),
}
#######################################################
