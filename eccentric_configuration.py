from pathlib import Path

from sklearn.gaussian_process.kernels import RBF, WhiteKernel

TRAINING_METRIC_FILEPATH = Path("training_metrics.nc")
TEST_METRIC_FILEPATH = Path("test_metrics.nc")

RBF_KERNEL_PARAMETERS = dict(
    length_scale=[1, 15, 0.05, 0.05],
    length_scale_bounds=[[1e-10, 1e10], [1e-10, 1e10], [1e-10, 1e10], [1e-10, 1e10]],
)
WHITE_KERNEL_PARAMETERS = dict(noise_level=0.01, noise_level_bounds=[1e-30, 1e30])
GAUSSIAN_PROCESS_REGRESSION_PARAMETERS = dict(alpha=2e-2, normalize_y=True)
GPR_KWARGS = {
    **dict(
        kernel=RBF(**RBF_KERNEL_PARAMETERS) + WhiteKernel(**WHITE_KERNEL_PARAMETERS)
    ),
    **GAUSSIAN_PROCESS_REGRESSION_PARAMETERS,
}

USE_POLAR = True
MAXIMUM_ECCENTRICITY = 0.225

COORDINATE_NAMES = [
    "rotation_period",
    "obliquity",
    "eccentricity",
    "longitude_at_periapse",
]
COORDINATE_NAMES_FOR_GRID = (
    ["rotation_period", "obliquity"] + ["ecc_cos_lon", "ecc_sin_lon"]
    if USE_POLAR
    else ["eccentricity", "longitude_at_periapse"]
)
COORDINATE_PRINT_NAMES = [
    r"Rotation Period (days)",
    r"Obliquity $\left(^\circ\right)$",
] + (
    [r"$e \cos\phi_\mathrm{peri}$", r"$e \sin\phi_\mathrm{peri}$"]
    if USE_POLAR
    else ["$e$", r"$\phi_\mathrm{peri}$ $\left(^\circ\right)$"]
)
