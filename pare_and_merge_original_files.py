from dataset import (
    add_variables_and_attributes,
    build_dataset_from_multiple_files,
    build_dataset_from_single_file,
)
from model_variables import model_variables
from nonmetric_calculated_variables import (
    nonmetric_calculated_variables,
    nonmetric_variable_attrs,
)
from original_file_loaders import (
    DatasetSpecs,
    get_He2022_specs,
    get_LHSTE_specs,
    get_LHSTR_specs,
)
from user_filepaths import LOCAL_PROCESSED_DATASET_DIRECTORY

LIST_OF_VARIABLES: list[str] = list(model_variables.keys())

# TRAINING DATA
LHSTR_specs: DatasetSpecs = get_LHSTR_specs()

training_data = build_dataset_from_multiple_files(
    variables=LIST_OF_VARIABLES, **LHSTR_specs
)
training_data = training_data.sortby("rotation_period")
training_data = add_variables_and_attributes(
    dataset=training_data,
    calculators=nonmetric_calculated_variables,
    attributes=nonmetric_variable_attrs,
)
training_data.to_netcdf(LOCAL_PROCESSED_DATASET_DIRECTORY / "LHSTR_data.nc")

# TEST DATA
LHSTE_specs: DatasetSpecs = get_LHSTE_specs()

test_data = build_dataset_from_single_file(variables=LIST_OF_VARIABLES, **LHSTE_specs)
test_data = test_data.sortby("rotation_period")
test_data = add_variables_and_attributes(
    dataset=test_data,
    calculators=nonmetric_calculated_variables,
    attributes=nonmetric_variable_attrs,
)
test_data.to_netcdf(LOCAL_PROCESSED_DATASET_DIRECTORY / "LHSTE_data.nc")

# CIRCULAR DATA
He2022_specs: DatasetSpecs = get_He2022_specs()

circular_data = build_dataset_from_multiple_files(
    variables=LIST_OF_VARIABLES, **He2022_specs, month_name="time"
)
# circular_data = circular_data.sortby("rotation_period")
circular_data = add_variables_and_attributes(
    dataset=circular_data,
    calculators=nonmetric_calculated_variables,
    attributes=nonmetric_variable_attrs,
)
circular_data.to_netcdf(LOCAL_PROCESSED_DATASET_DIRECTORY / "He_data.nc")
