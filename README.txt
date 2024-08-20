Habitability Metric Emulator Files

There are three levels of files:

(1) are the original files Chris provides, which are pared and merged (using "pare_and_merge_original_files.py")
    into intermediate NetCDF files that have only a handful of the most useful climate variables.
    (These are not uploaded with the code.)

(2) are the intermediate NetCDF files that are used to calculate time+spatial averages of temperature and precipitation
    as well as the metrics derived from them. We compile the averages and metrics in "make_*_metrics_file.py" scripts.

(3) are the metric files ("*_metrics.nc"), which are used for pretty much all other calculations and scripts.