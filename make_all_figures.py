from eccentric_configuration import TEST_METRIC_FILEPATH, TRAINING_METRIC_FILEPATH
from make_circular_emulators import CIRCULAR_METRIC_FILEPATH, run_for_circular_metrics
from make_circular_metrics_file import create_circular_metrics_file
from make_eccentric_emulators import run_for_training_and_all_metrics
from make_eccentric_metrics_files import (
    TEST_INPUT_FILEPATH,
    TEST_OUTPUT_FILEPATH,
    TRAINING_INPUT_FILEPATH,
    TRAINING_OUTPUT_FILEPATH,
    create_eccentric_metrics_file,
)
from make_latex_table_output import make_latex_table_output
from make_scatterplots import plot_combined_scatterplots, plot_metric_scatterplots
from make_test_vs_emulated_plot import run_plotting_routines
from make_zero_obliquity_comparison_plot import make_zero_obliquity_comparison_plot


def make_all_figures():
    create_circular_metrics_file()
    make_zero_obliquity_comparison_plot(CIRCULAR_METRIC_FILEPATH)
    run_for_circular_metrics(
        emulated_variables=["fT_land", "frunoff_land", "static_habitability_land"]
    )

    create_eccentric_metrics_file(TRAINING_INPUT_FILEPATH, TRAINING_OUTPUT_FILEPATH)
    create_eccentric_metrics_file(TEST_INPUT_FILEPATH, TEST_OUTPUT_FILEPATH)
    run_for_training_and_all_metrics(
        emulated_variable_names=["fT_land", "frunoff_land", "static_habitability_land"]
    )
    plot_metric_scatterplots(
        TRAINING_METRIC_FILEPATH,
        TEST_METRIC_FILEPATH,
        temperature_variable_name="tsurf_land_average",
        precipitation_variable_name="prec_land_average",
    )
    plot_combined_scatterplots(
        TRAINING_METRIC_FILEPATH,
        TEST_METRIC_FILEPATH,
        temperature_variable_name="tsurf_land_average",
        precipitation_variable_name="runoff_land_average",
        temperature_habitability_variable_name="fT_land",
        precipitation_habitability_variable_name="frunoff_land",
        climate_habitability_variable_name="static_habitability_land",
    )
    run_plotting_routines(habitability_variable_name="static_habitability_land")

    latex_table_rows: list[str] = make_latex_table_output(
        TRAINING_METRIC_FILEPATH, TEST_METRIC_FILEPATH
    )
    for row in latex_table_rows:
        print(row)


if __name__ == "__main__":
    make_all_figures()
