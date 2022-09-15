# Astronavigation

Here we briefly illustrate the purpose of the programs in this repository. We are able to reach three main goals which are:
1. Rewriting in *Python* language the astronavigation code from *Guido Fiorillo* stage project
2. Compare different formulas for light deflection and apply them to the planets of the Solar System and also to extrasolar bodies such as exoplanets, black holes and wormholes.
3. Simulate the *Compound shell* method used to identify gravitational effects of invisible isolated objects in space.

## Code description
Here I briefly describe the Python package I developed in my internship. The
main standard Python libraries that I used are Numpy and SciPy for mathemati-
cal implementation, Pandas for dataset manipulation, Matplotlib for plots and
Astropy for astrophysical parameters.

The main formulas and the physical information about the Solar System and
the exosystems are collected in a package called [astronavigation](astronavigation) divided into:
* [deflection.py](astronavigation/deflection.py): contains the main formulas for the evaluation of light
deflection;
* [planets.py](astronavigation/planets.py): contains the information about the Solar System and compute
the position of its bodies from the ephemerides of a given epoch;
* [read_exo.py](astronavigation/read_exo.py): contains user-friendly methods to read the information of
exosystem from the Nasa Exoplanet archive.

In order to solve the specific tasks I wrote the following Python scripts and
Jupyter Notebook:
* planets_deflection0.py;
* planets_deflection90.py;
* Compound_shell;
* pointing_max_error.py;
* pointing_date.py;
* Exosystems;
* comparison.py;
* ellis_test.py.
