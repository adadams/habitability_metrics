Habitability metric code for Adams, Colose, Merrelli, Turnbull, & Kane (2024)

This repository serves as an archive for the Python scripts used to generate most of the figures in the code. "make_all_figures.py" can be run as-is. Each "make_*.py" file can be run as its own script.

"user_filepaths.py" may need to be edited depending on your use case.

There are three levels of files used for this work. Only the third level is published with the repository due to space, but the second level data contains the processed versions of all the original model outputs and will be available in the near future (currently upon request):

(1) are the original files from the data portal (https://portal.nccs.nasa.gov/GISS_modelE/ROCKE-3D/data-share/hab4D/),
    which can be pared and merged using "pare_and_merge_original_files.py" into intermediate NetCDF files that have
    only a handful of the most useful climate variables.

(2) are the intermediate NetCDF files that are used to calculate time+spatial averages of temperature and precipitation
    as well as the metrics derived from them. We compile the averages and metrics in "make_*_metrics_file.py" scripts.

(3) are the metric files ("*_metrics.nc"), which are used for pretty much all other calculations and scripts.