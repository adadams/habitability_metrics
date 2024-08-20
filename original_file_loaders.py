from glob import glob

import numpy as np
import pandas as pd
from user_filepaths import ORIGINAL_MODEL_STORAGE_DIRECTORY
from xarray import Dataset


def read_dimensions_from_csv(
    dimensions_filepath, run_identifiers, month_weights_filepath, case_name="case"
):
    grid_parameters = pd.read_csv(dimensions_filepath, index_col=0)

    rotation_periods = [
        grid_parameters.loc[id]["sidereal.rot.day"] for id in run_identifiers
    ]
    obliquities = [grid_parameters.loc[id]["obliquity"] for id in run_identifiers]
    eccentricities = [grid_parameters.loc[id]["eccentricity"] for id in run_identifiers]
    longitudes = [
        grid_parameters.loc[id]["LongitudeAtPeriapsis"] for id in run_identifiers
    ]

    month_length_table = pd.read_csv(month_weights_filepath)
    month_length_weights = [
        month_length_table[id[7:].lstrip("0")]
        / np.sum(month_length_table[id[7:].lstrip("0")])
        for id in run_identifiers
    ]
    month_length_weights = pd.DataFrame(
        month_length_weights / np.sum(month_length_weights, axis=1)[:, np.newaxis]
    )

    case_ids = [id[len(filepath_header) :] for id in run_identifiers]
    if "Earth" in case_ids:
        case_ids[case_ids.index("Earth")] = "000"
    case_ids = np.array([int(id) for id in case_ids])

    dimensions = Dataset(
        dict(
            rotation_period=(case_name, rotation_periods),
            obliquity=(case_name, obliquities),
            eccentricity=(case_name, eccentricities),
            longitude_at_periapse=(case_name, longitudes),
        ),
        coords={case_name: case_ids},
    )

    return dict(
        dimensions=dimensions,
        outputs_per_year=12,
        month_length_weights=month_length_weights,
    )


###############################################################################
################ Latin Hypercube Sampled Training Run, Monthly ################
###############################################################################

directory = ORIGINAL_MODEL_STORAGE_DIRECTORY / "LHSTR4/"

filepaths = glob(str(directory / "monthly/outLHSTR*.nc"))
filepaths.sort()

filepath_header = "LHSTR4_"
run_identifiers = [
    filepath_header
    + filepath[
        filepath.index(filepath_header) + len(filepath_header) : filepath.index(".nc")
    ]
    for filepath in filepaths
]
# run_positions = np.argsort(run_identifiers)
sorted_filepaths = np.asarray(filepaths)  # [run_positions]

LHSTR_dimensions_kwargs = dict(
    dimensions_filepath=directory + "turnbull_rundecks_train4.csv",
    run_identifiers=run_identifiers,
    month_weights_filepath=directory + "monweights.csv",
)

LHSTR_monthly_kwargs = dict(filepaths=sorted_filepaths) | read_dimensions_from_csv(
    **LHSTR_dimensions_kwargs
)
###############################################################################


###############################################################################
############ Latin Hypercube Sampled Training Run, Monthly (merged) ###########
###############################################################################

directory = ORIGINAL_MODEL_STORAGE_DIRECTORY / "LHSTR4/"

filepath = directory + "monthly/LHSTR_combined.nc"

filepath_header = "LHSTR4_"
run_identifiers = [filepath_header + f"{index:03d}" for index in range(1, 47)]

LHSTR_merged_dimensions_kwargs = dict(
    dimensions_filepath=directory + "turnbull_rundecks_train4.csv",
    run_identifiers=run_identifiers,
    month_weights_filepath=directory + "monweights.csv",
    case_name="file_index",
)

LHSTR_merged_monthly_kwargs = dict(filepath=filepath) | read_dimensions_from_csv(
    **LHSTR_merged_dimensions_kwargs
)
###############################################################################


###############################################################################
################# Latin Hypercube Sampled Test Run, Monthly ###################
###############################################################################

directory = ORIGINAL_MODEL_STORAGE_DIRECTORY / "LHSTE4/"

filepaths = glob(str(directory / "monthly/outLHSTE*.nc"))

filepath_header = "LHSTE4_"
run_identifiers = [
    filepath_header
    + filepath[
        filepath.index(filepath_header) + len(filepath_header) : filepath.index(".nc")
    ]
    for filepath in filepaths
]
run_positions = np.argsort(run_identifiers)
sorted_filepaths = np.asarray(filepaths)[run_positions]

LHSTE_dimensions_kwargs = dict(
    dimensions_filepath=directory + "turnbull_rundecks_test4.csv",
    run_identifiers=run_identifiers,
    month_weights_filepath=directory + "monweights_test.csv",
)

LHSTE_monthly_kwargs = dict(filepaths=sorted_filepaths) | read_dimensions_from_csv(
    **LHSTE_dimensions_kwargs
)
###############################################################################

###############################################################################
############## Latin Hypercube Sampled Test Run, Monthly (merged) #############
###############################################################################
directory = ORIGINAL_MODEL_STORAGE_DIRECTORY / "LHSTE4/"

filepath = directory + "monthly/LHSTE_combined.nc"

filepath_header = "LHSTE4_"
run_identifiers = [filepath_header + f"{index:03d}" for index in range(1, 47)]

LHSTE_merged_dimensions_kwargs = dict(
    dimensions_filepath=directory + "turnbull_rundecks_test4.csv",
    run_identifiers=run_identifiers,
    month_weights_filepath=directory + "monweights_test.csv",
    case_name="file_index",
)

LHSTE_merged_monthly_kwargs = dict(filepath=filepath) | read_dimensions_from_csv(
    **LHSTE_merged_dimensions_kwargs
)
###############################################################################


"""
###############################################################################
################ Latin Hypercube Sampled Training Run, Annual #################
###############################################################################

directory = ORIGINAL_MODEL_STORAGE_DIRECTORY+'LHSTR/'

filepaths = glob(directory+'annual/*.nc')
run_identifiers = [
    filepath[filepath.index('aij')+3:filepath.index('.nc')]
    for filepath in filepaths
    ]
run_positions = np.argsort(run_identifiers)
sorted_filepaths = np.asarray(filepaths)[run_positions]

training_grid_parameters = pd.read_excel(directory+'turnbull_rundecks_train1.xlsx', index_col=0)

rotation_periods = [
    training_grid_parameters.loc[id]['sidereal.rot.day']
    for id in run_identifiers
    ]
obliquities = [
    training_grid_parameters.loc[id]['obliquity']
    for id in run_identifiers
    ]
eccentricities = [
    training_grid_parameters.loc[id]['eccentricity']
    for id in run_identifiers
    ]
longitudes = [
    training_grid_parameters.loc[id]['LongitudeAtPeriapsis']
    for id in run_identifiers
    ]

dimensions = {
    'obliquity': obliquities,
    'rotation_period': rotation_periods,
    'eccentricity': eccentricities,
    'longitude_at_periapse': longitudes
    }

LHSTR_annual_kwargs = grid_kwargs(sorted_filepaths, dimensions, 1, None)

###############################################################################

###############################################################################
############################ He+ 2022 Grid, Annual ############################
###############################################################################

directory = ORIGINAL_MODEL_STORAGE_DIRECTORY + 'ROCKE_timeseries_annual_last50/'

filepaths = glob(directory+'mcANNtseries50.aijROCKE.*')

# File names contain rotation rate in days after 'X'
# and obliquity in degrees after 'OBL'.
run_parameters = [filepath[filepath.index('.X')+1:filepath.index('.nc')]
                  for filepath in filepaths]

rotation_periods = np.asarray([parameter[parameter.index('X')+1:parameter.index('OBL')]
                               for parameter in run_parameters], dtype=float)

obliquities = np.asarray([parameter[parameter.index('OBL')+3:]
                          for parameter in run_parameters], dtype=float)

eccentricities = np.zeros_like(rotation_periods)

longitudes_at_periapse = np.zeros_like(rotation_periods)

dimensions = {
    'obliquity': obliquities,
    'rotation_period': rotation_periods,
    'eccentricity': eccentricities,
    'longitude_at_periapse': longitudes_at_periapse
    }

He_annual_kwargs = grid_kwargs(filepaths, dimensions, 1, None)

###############################################################################
"""

###############################################################################
########################### He+ 2022 Grid, Monthly ############################
###############################################################################

directory = ORIGINAL_MODEL_STORAGE_DIRECTORY / "ROCKE_timeseries_monthly_last50/"

filepaths = glob(str(directory / "mcMONtseries50.aijROCKE.*"))

# File names contain rotation rate in days after "X"
# and obliquity in degrees after "OBL".
run_parameters = [
    filepath[filepath.index(".X") + 1 : filepath.index(".nc")] for filepath in filepaths
]

rotation_periods = np.asarray(
    [
        parameter[parameter.index("X") + 1 : parameter.index("OBL")]
        for parameter in run_parameters
    ],
    dtype=float,
)

obliquities = np.asarray(
    [parameter[parameter.index("OBL") + 3 :] for parameter in run_parameters],
    dtype=float,
)

eccentricities = np.zeros_like(rotation_periods)

longitudes = np.zeros_like(rotation_periods)

month_length_weights = np.ones((len(rotation_periods), 12))

dimensions = Dataset(
    {
        "rotation_period": ("case", rotation_periods),
        "obliquity": ("case", obliquities),
        "eccentricity": ("case", eccentricities),
        "longitude_at_periapse": ("case", longitudes),
    },
    coords={"case": np.arange(len(rotation_periods))},
)

He_monthly_kwargs = dict(
    filepaths=filepaths,
    dimensions=dimensions,
    outputs_per_year=12,
    month_length_weights=month_length_weights,
)

###############################################################################
