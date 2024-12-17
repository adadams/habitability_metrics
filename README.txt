Habitability Metric Emulator Files

There are three levels of files:

(1) are the original files from the data portal (https://portal.nccs.nasa.gov/GISS_modelE/ROCKE-3D/data-share/hab4D/),
    which can be pared and merged using "pare_and_merge_original_files.py" into intermediate NetCDF files that have
    only a handful of the most useful climate variables.

(2) are the intermediate NetCDF files that are used to calculate time+spatial averages of temperature and precipitation
    as well as the metrics derived from them. We compile the averages and metrics in "make_*_metrics_file.py" scripts.

(3) are the metric files ("*_metrics.nc"), which are used for pretty much all other calculations and scripts.