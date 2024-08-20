import numpy as np
from dataset import (
    build_dataset_from_compiled_file,
    build_dataset_from_multiple_files,
)
from model_variables import model_variables
from original_file_loaders import He_monthly_kwargs as CIRCULAR_DATA_KWARGS
from original_file_loaders import LHSTE_merged_monthly_kwargs as MERGED_TEST_DATA_KWARGS
from original_file_loaders import (
    LHSTR_merged_monthly_kwargs as MERGED_TRAINING_DATA_KWARGS,
)
from original_file_loaders import LHSTR_monthly_kwargs as TRAINING_DATA_KWARGS
from user_filepaths import LOCAL_PROCESSED_DATASET_DIRECTORY
from xarray import concat

LIST_OF_VARIABLES = list(model_variables.keys())

# TRAINING DATA (still use Earth case from original separate-file set)
original_training_data = build_dataset_from_multiple_files(
    LIST_OF_VARIABLES, **TRAINING_DATA_KWARGS
)
Earth_case = original_training_data.sel(
    case=original_training_data.case[
        np.where(original_training_data.obliquity == 23.44)
    ]
)
Earth_case["case"] = np.array([0])

merged_training_data = build_dataset_from_compiled_file(
    LIST_OF_VARIABLES, **MERGED_TRAINING_DATA_KWARGS
)
training_data = concat([merged_training_data, Earth_case], dim="case")
training_data = training_data.sortby("rotation_period")
training_data.to_netcdf(LOCAL_PROCESSED_DATASET_DIRECTORY / "LHSTR_data.nc")

# TEST DATA
merged_test_data = build_dataset_from_compiled_file(
    LIST_OF_VARIABLES, **MERGED_TEST_DATA_KWARGS
)
merged_test_data = merged_test_data.sortby("rotation_period")
merged_test_data.to_netcdf(LOCAL_PROCESSED_DATASET_DIRECTORY / "LHSTE_data.nc")

# CIRCULAR DATA
circular_data = build_dataset_from_multiple_files(
    LIST_OF_VARIABLES, **CIRCULAR_DATA_KWARGS
)
circular_data.to_netcdf(LOCAL_PROCESSED_DATASET_DIRECTORY / "He_data.nc")
