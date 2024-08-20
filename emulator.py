from typing import NamedTuple

import numpy as np
from dataset import recast_eccentricity_dimensions_in_polar
from numpy.typing import ArrayLike
from sklearn.gaussian_process import GaussianProcessRegressor


class Emulator(NamedTuple):
    regressor: GaussianProcessRegressor
    dimensions: ArrayLike
    emulated: list[float]
    emulated_mean: float


def build_emulator(dimensions, emulated_variable, use_polar, **GPR_kwargs):
    """Builds an emulator from a training data set and a kernel."""
    gpr = GaussianProcessRegressor(**GPR_kwargs)

    emulation_dimensions = (
        recast_eccentricity_dimensions_in_polar(dimensions) if use_polar else dimensions
    )

    X = np.vstack([dimension for dimension in emulation_dimensions.values()]).T
    y = emulated_variable
    y_mean = y.mean()

    gpr.fit(X, y)

    return Emulator(gpr, X, y, y_mean)


def construct_emulator_set(
    emulator, parameter_values, predicted_min=0, predicted_max=1
):
    pred_grid, pred_grid_stdev = emulator.predict(parameter_values, return_std=True)

    return {
        "locations": parameter_values,
        "emulated_values": np.clip(pred_grid, a_min=predicted_min, a_max=predicted_max),
        "emulated_error": pred_grid_stdev,
    }


def construct_emulator_grid(training_fit, grid_spacings, indexing="xy"):
    grid = np.meshgrid(*grid_spacings, indexing=indexing)

    X_grid = np.vstack([dimension.flatten() for dimension in grid]).T

    emulator_set = construct_emulator_set(training_fit, X_grid)

    return {
        "locations": [np.squeeze(dimension) for dimension in grid],
        "emulated_values": np.squeeze(
            emulator_set["emulated_values"].reshape(np.shape(grid[0]))
        ),
        "emulated_error": np.squeeze(
            emulator_set["emulated_error"].reshape(np.shape(grid[0]))
        ),
    }


def construct_lowerD_grid(
    emulator, grid_spacings, list_of_dimensions, nongrid_positions=None
):
    if nongrid_positions is None:
        nongrid_positions = {dimension: [0] for dimension in grid_spacings}

    return construct_emulator_grid(
        emulator,
        [
            dimension if name in list_of_dimensions else np.atleast_1d(fixed_dimension)
            for (name, dimension), fixed_dimension in zip(
                grid_spacings.items(), nongrid_positions.values()
            )
        ],
    )
