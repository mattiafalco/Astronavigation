# Astronavigation

Here we briefly illustrate the purpose of the programs in this repository. We are able to reach three main goals which are:
1. Rewriting in *Python* language the astronavigation code from *Guido Fiorillo* stage project
2. Compare different formulas for light deflection and apply them to the planets of the Solar System and also to extrasolar bodies such as exoplanets, black holes and wormholes.
3. Simulate the *Compound shell* method used to identify gravitational effects of invisible isolated objects in space.

## Code description
In the directory [astronavigation](astronavigation) are written the functions that are used to evaluate the light deflection and to read all the information about the Solar system and the exoplanets.

The main programs are:
* [error_point.py](error_point.py) evaluates the maximum deflection due to the planets of the Solar System.
* [pointing_date.py](pointing_date.py) evalutes the deflection due to planets of the Solar System for the given exoplanets and in a particular range of given dates.
* [comparison.py](comparison.py) makes a comparison between different formulas and different
contributions of light deflection. For different types of bodies in the Solar System
and also exo bodies. The quantity taken in consideration are:
  - deflection w/ null velocity
  - deflection w/ velocity
  - Erez-Rosen deflection
  - Erez-Rosen monopole correction
  - quadrupole deflection
  - Erez-Rosen quadrupole deflection
  - Erez-Rosen quadrupole correction
  - Erez-Rosen centroid shift
  
  an Ellis wormhole is also considered.
* 