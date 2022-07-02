# adiabatic-tides
Code for evaluating the tidal stripping of dark matter (sub)haloes in the adiabatic limit

![alt text](https://github.com/jstuecker/adiabatic-tides/blob/main/img/tidal_experiment.png)

## Directories:

* adiabatic_tides: Contains all the important python module files. This is the main part of this repository
* notebooks: Contains example notebooks to show how to use the model
    - [notebooks/tutorials](notebooks/tutorials): Tutorials for introducing the the code. Best to start here!
    - [notebooks/paper](notebooks/paper): Notebooks that contain many of the plots presented in the paper. Here is a good place to look for examples!
    - [notebooks/verification](notebooks/verification): Some very specific notebooks to test numerical validity of the code. Not recommended for most users
* caches: Contains some .hdf5 files to store precalculated results. In principle, you can delete this at the cost of needing to recalculate results
* img: Just an image for enhancing the presentation

## Usage:
Easiest would be inside python...

```
import sys
sys.path.append("path/to/adiabatic-tides")
import adiabatic_tides
[...]
```

## Getting Started:
I'd recommend to start with the jupyter notebook under [notebooks/tutorials/inroduction.ipynb](notebooks/tutorials/inroduction.ipynb)
Afterwards you might have a look at the notebooks under [notebooks/paper](notebooks/paper) for some additional examples!

Also don't for get to read the paper!

## Requirements:
* numpy
* scipy
* matplotlib

## Acknowledgement:
If you use this code, please cite the paper
