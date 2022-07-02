# adiabatic-tides
Code for evaluating the tidal stripping of dark matter (sub)haloes in the adiabatic limit

![alt text](https://github.com/jstuecker/adiabatic-tides/blob/main/img/tidal_experiment.png)

## Directories:

* adiabatic_tides: Contains all the important python module files. This is the main part of this repository
* notebooks: Contains example notebooks to show how to use the model
    - notebooks/paper: Notebooks that contain many of the plots presented in the paper. Here is a good place to look for examples!
    - notebooks/verification: Some very specific notebooks to test numerical validity of the code. Not recommended for most users
* caches: Contains some .hdf5 files to store precalculated results. In principle, you can delete this at the cost of needing to recalculate results
* img: Just an image for enhancing the presentation

# Usage:
Easiest would be inside python...

```
import sys
sys.path.append("path/to/adiabatic-tides")
import adiabatic_tides
```

# Requirements:
* numpy
* scipy
* matplotlib
