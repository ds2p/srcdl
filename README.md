Super-resolution Convolutional Dictionary Learning
==============================

Convolutional sparse coding (CSC) and convolutional dictionary learning (CDL) for off-the-grid events. This code is for the manuscript

Song, A., Flores, F., and Ba D., **Convolutional Dictionary Learning with Grid Refinement**, *IEEE Transaction on Signal Processing*, 2020

- 1-D example for spike sorting application is available.
- 2-D example for Single-molecule-localization-microscopy (SMLM) will be made available soon.

Please email Andrew Song (andrew90@mit.edu) for any questions/suggestions

<h3>Getting started</h3>
To get started, clone the repository and run

```
pip install -r requirements.txt
```

To run CDL without interpolation on spikesorting application run the following,

```
cd src/run_experiments
python run_experiments.py train --folder_name=spikesorting_no_interp
```

If you want to run CDL with interpolation, change "spikesorting_no_interp"  to "spikesorting_interp".

To predict with the learned dictionary, run the following (change the folder name for interpolated dictionary as above),

```
cd src/run_experiments
python run_experiments.py predict --folder_name=spikesorting_no_interp
```

To generate the error curve (along with pre-computed baseline error curves), run (Depending on the platform, you might have to do display_spikesorting_errorcurve with underscore, not '-')

```
python extract_results.py display-spikesorting-errorcurve
```

This command will generate the error curve and save it.
