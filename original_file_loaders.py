from glob import glob
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from user_filepaths import ORIGINAL_MODEL_STORAGE_DIRECTORY
from xarray import Dataset


def read_in_dataset_specifications(
    dimensions_filepath: str | Path,
    filepath_header_length: int,
    run_identifiers: list[str],
    month_weights_filepath: str | Path,
    case_name: str = "case",
) -> dict[str, Any]:
    print(f"Loading dimensions from {dimensions_filepath}.")

    grid_parameters = pd.read_csv(dimensions_filepath, index_col=0)
    grid_parameters.index = grid_parameters.index.str.strip()
    grid_parameters.columns = grid_parameters.columns.str.strip()

    rotation_periods: list[float] = [
        grid_parameters.loc[id]["sidereal.rot.day"] for id in run_identifiers
    ]
    obliquities: list[float] = [
        grid_parameters.loc[id]["obliquity"] for id in run_identifiers
    ]
    eccentricities: list[float] = [
        grid_parameters.loc[id]["eccentricity"] for id in run_identifiers
    ]
    longitudes: list[float] = [
        grid_parameters.loc[id]["LongitudeAtPeriapsis"] for id in run_identifiers
    ]

    month_length_table: pd.DataFrame = pd.read_csv(month_weights_filepath)
    month_length_weights: list[float] = [
        month_length_table[id[filepath_header_length:].lstrip("0")]
        / np.sum(month_length_table[id[filepath_header_length:].lstrip("0")])
        for id in run_identifiers
    ]
    month_length_weights: pd.DataFrame = pd.DataFrame(
        month_length_weights / np.sum(month_length_weights, axis=1)[:, np.newaxis]
    )

    case_ids: list[str] = [id[filepath_header_length:] for id in run_identifiers]
    if "Earth" in case_ids:
        case_ids[case_ids.index("Earth")] = "000"
    case_ids: NDArray[np.int_] = np.array([int(id) for id in case_ids])

    dimensions: Dataset = Dataset(
        {
            "rotation_period": (case_name, rotation_periods),
            "obliquity": (case_name, obliquities),
            "eccentricity": (case_name, eccentricities),
            "longitude_at_periapse": (case_name, longitudes),
        },
        coords={case_name: case_ids},
    )

    return {
        "dimensions": dimensions,
        "outputs_per_year": 12,
        "month_length_weights": month_length_weights,
    }


class DatasetSpecs(TypedDict):
    model_output_filepaths: Path | list[Path]
    dimensions: Dataset
    outputs_per_year: int
    month_length_weights: NDArray[np.float64]


class DatasetReadinSpecs(TypedDict):
    dimensions_filepath: Path
    run_identifiers: list[str]
    filepath_header_length: int
    month_weights_filepath: Path
    case_name: str = "case"


def get_model_output_specs(
    model_output_filepaths: Path,
    dimensions_filepath: Path,
    month_weights_filepath: Path,
    filepath_header_length: int,
    run_identifiers: list[str],
) -> DatasetSpecs:
    dataset_readin_kwargs: DatasetReadinSpecs = DatasetReadinSpecs(
        dimensions_filepath=dimensions_filepath,
        run_identifiers=run_identifiers,
        filepath_header_length=filepath_header_length,
        month_weights_filepath=month_weights_filepath,
    )

    return DatasetSpecs(
        model_output_filepaths=model_output_filepaths,
        **read_in_dataset_specifications(**dataset_readin_kwargs),
    )


def get_LHSTR_specs(
    directory: Path = ORIGINAL_MODEL_STORAGE_DIRECTORY / "LHSTR4/",
    filepath_header: str = "LHSTR4_",
    number_of_runs: int = 46,
) -> DatasetSpecs:
    model_output_filepaths: list[Path] = [
        directory / "monthly/LHSTR_combined.nc",
        directory / "monthly/outEarth.nc",
    ]
    dimensions_filepath: Path = directory / "turnbull_rundecks_train4.csv"
    month_weights_filepath: Path = directory / "monweights.csv"

    run_identifiers: list[str] = [
        *[filepath_header + f"{index:03d}" for index in range(1, number_of_runs + 1)],
        filepath_header + "Earth",
    ]

    return get_model_output_specs(
        model_output_filepaths=model_output_filepaths,
        dimensions_filepath=dimensions_filepath,
        month_weights_filepath=month_weights_filepath,
        filepath_header_length=len(filepath_header),
        run_identifiers=run_identifiers,
    )


def get_LHSTE_specs(
    directory: Path = ORIGINAL_MODEL_STORAGE_DIRECTORY / "LHSTE4/",
    filepath_header: str = "LHSTE4_",
    number_of_runs: int = 46,
) -> DatasetSpecs:
    model_output_filepath: Path = directory / "monthly/LHSTE_combined.nc"
    dimensions_filepath: Path = directory / "turnbull_rundecks_test4.csv"
    month_weights_filepath: Path = directory / "monweights_test.csv"

    run_identifiers: list[str] = [
        filepath_header + f"{index:03d}" for index in range(1, number_of_runs + 1)
    ]

    return get_model_output_specs(
        model_output_filepaths=model_output_filepath,
        dimensions_filepath=dimensions_filepath,
        month_weights_filepath=month_weights_filepath,
        filepath_header_length=len(filepath_header),
        run_identifiers=run_identifiers,
    )


def get_He2022_specs(
    directory=ORIGINAL_MODEL_STORAGE_DIRECTORY / "ROCKE_timeseries_monthly_last50/",
) -> DatasetSpecs:
    model_output_filepaths: list[str] = glob(
        str(directory / "mcMONtseries50.aijROCKE.*")
    )

    # File names contain rotation rate in days after "X"
    # and obliquity in degrees after "OBL".
    run_parameters: list[str] = [
        filepath[filepath.index(".X") + 1 : filepath.index(".nc")]
        for filepath in model_output_filepaths
    ]

    rotation_periods: NDArray[np.float64] = np.asarray(
        [
            parameter[parameter.index("X") + 1 : parameter.index("OBL")]
            for parameter in run_parameters
        ],
        dtype=float,
    )

    obliquities: NDArray[np.float64] = np.asarray(
        [parameter[parameter.index("OBL") + 3 :] for parameter in run_parameters],
        dtype=float,
    )

    eccentricities: NDArray[np.float64] = np.zeros_like(rotation_periods)

    longitudes: NDArray[np.float64] = np.zeros_like(rotation_periods)

    month_length_weights: NDArray[np.float64] = np.ones((len(rotation_periods), 12))

    dimensions: Dataset = Dataset(
        {
            "rotation_period": ("case", rotation_periods),
            "obliquity": ("case", obliquities),
            "eccentricity": ("case", eccentricities),
            "longitude_at_periapse": ("case", longitudes),
        },
        coords={"case": np.arange(len(rotation_periods))},
    )

    return DatasetSpecs(
        model_output_filepaths=model_output_filepaths,
        dimensions=dimensions,
        month_length_weights=month_length_weights,
        outputs_per_year=12,
    )
