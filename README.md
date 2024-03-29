# adiabatic-tides
Code for evaluating the tidal stripping of dark matter (sub)haloes in the adiabatic limit. For details please read the paper [arXiv:2207.00604](https://arxiv.org/abs/2207.00604)

![alt text](https://github.com/jstuecker/adiabatic-tides/blob/main/img/tidal_experiment.png)

## Directories:

* [adiabatic_tides](adiabatic_tides): Contains all the important python module files. This is the main part of this repository
* [notebooks](notebooks): Contains example notebooks to show how to use the model
    - [notebooks/tutorials](notebooks/tutorials): Tutorials for introducing the the code. Best to start here!
    - [notebooks/paper](notebooks/paper): Notebooks that contain many of the plots presented in the paper. Here is a good place to look for examples!
    - [notebooks/verification](notebooks/verification): Some very specific notebooks to test numerical validity of the code. Not recommended for most users
* [caches](caches): Contains some .hdf5 files to store precalculated results. In principle, you can delete this at the cost of needing to recalculate results
* [img](img): Just an image for enhancing the presentation

## Usage:
Inside of python:

```
import sys
sys.path.append("path/to/adiabatic-tides")
import adiabatic_tides
[...]
```

## Getting Started:
I'd recommend to start with the jupyter notebook under [notebooks/tutorials/introduction.ipynb](notebooks/tutorials/introduction.ipynb).
Afterwards you might have a look at the notebooks under [notebooks/paper](notebooks/paper) for some additional examples!

Also don't forget to read [the paper](https://arxiv.org/abs/2207.00604)!

## Requirements:
* numpy
* scipy
* matplotlib

## More:
For some intuition you might want to watch these videos, showing the potential landscapes and the massloss of subhaloes on a circular and a non-circular orbit:
* https://youtu.be/RK1cRlplO0k
* https://youtu.be/3MUma-22yr4

## Acknowledgement:
If you use this code, please cite the paper as [Stücker et al. (2022)](https://arxiv.org/abs/2207.00604)

## References:
A related code -- treating the effect of stellar encounters on prompt cusps -- can be found here [https://github.com/jstuecker/cusp-encounters](https://github.com/jstuecker/cusp-encounters)
