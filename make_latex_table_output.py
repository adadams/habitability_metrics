import re
from pathlib import Path

import numpy as np
from eccentric_configuration import TEST_METRIC_FILEPATH, TRAINING_METRIC_FILEPATH
from xarray import load_dataset


def view_row_as_dict(dataset, index):
    dataset_as_dict = dataset.sel(case=index).to_dict("array")
    coordinates = {
        key: value["data"] if key != "rotation_period" else 2 ** value["data"]
        for key, value in dataset_as_dict["coords"].items()
    }
    data_variables = {
        key: value["data"] for key, value in dataset_as_dict["data_vars"].items()
    }

    return {**coordinates, **data_variables}


def make_latex_table_output(
    training_metric_filepath: Path, test_metric_filepath: Path
) -> str:
    training_metrics = load_dataset(training_metric_filepath)
    test_metrics = load_dataset(test_metric_filepath)

    case_indices = np.unique(np.concatenate((training_metrics.case, test_metrics.case)))

    rotation_row_template = "{rotation_period:.3g}"
    case_row_template = (
        "{obliquity:.0f} & {eccentricity:.3f} & {longitude_at_periapse:.0f}"
    )
    case_row_metric_template = "{tsurf_land_average:.1f} & {prec_land_average:.2f} & {fT_land:.2f} & {fprec_land:.2f} & {habitability_land:.2f}"
    case_row_full_template = f"{case_row_template} & {case_row_metric_template}"
    number_of_entries_per_case = case_row_full_template.count("{")

    full_rows = []
    for case_index in case_indices:
        index_print = case_index if (case_index != 0) else "Earth"

        rotation_row = rotation_row_template.format(
            **view_row_as_dict(training_metrics, case_index)
        )

        training_row = (
            case_row_full_template.format(
                **view_row_as_dict(training_metrics, case_index)
            )
            if case_index in training_metrics.case
            else "&".join([" -- "] * number_of_entries_per_case)
        )

        test_row = (
            case_row_full_template.format(**view_row_as_dict(test_metrics, case_index))
            if case_index in test_metrics.case
            else "&".join([" -- "] * number_of_entries_per_case)
        )

        full_row = f"\\textbf{{{index_print}}} & {rotation_row} & {training_row} & {test_row} \\\\"

        is_negative_number = re.compile(r"-\d+\.?\d+?")
        for negative_number in is_negative_number.findall(full_row):
            full_row = full_row.replace(negative_number, rf"${negative_number}$")

        full_rows.append(full_row)

    return full_rows


if __name__ == "__main__":
    for row in make_latex_table_output(TRAINING_METRIC_FILEPATH, TEST_METRIC_FILEPATH):
        print(row)
