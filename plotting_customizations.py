from colorcet import cm
from matplotlib.pyplot import rc
from visualization_functions import create_linear_colormap

rc("text", usetex=True)
rc("font", family="serif")
rc("axes", labelsize=22)
rc("xtick", labelsize=20)
rc("ytick", labelsize=20)

plot_filetypes = ["pdf", "png"]

cmap_temperature = cm["coolwarm"]
cmap_precipitation = cm["bwy_r"]
cmap_kwargs = {
    "lightness_minimum": 0.1,
    "lightness_maximum": 0.9,
    "saturation_minimum": 0.05,
    "saturation_maximum": 0.5,
}
cmap_habitability = create_linear_colormap(["#388F75", "#388F52"], **cmap_kwargs)
