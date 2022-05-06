""" Description

Creator: mattia
Date: 06/05/22.
"""
import numpy as np
import pandas as pd
from read_exo import getExo, getExoAll
from planets import SolarSystem, Body
from astropy import constants
from deflection import *

# Define constants
pc = constants.pc.to('km').value
AU = constants.au.to('km').value
c = constants.c.to('km/s').value
eps = 1/c

# create solar system and jupiter
ss = SolarSystem()
jupiter = ss.getPlanet('jupiter')

# target
dist = 1 * pc
x = np.array([0, 1, 0])*dist

# exoplanet
orb_rad = 1 * AU
pl = Body(mass=5*jupiter.mass,
          pos=(dist - orb_rad)*np.array([0, 1, 0]),
          radius=jupiter.radius)

# observer
x_obs = AU*np.array([0, -1, 0])


# impact angle
chi = einstein_ring(pl.mass, eps, pl.pos, x)
print(np.rad2deg(chi)*3600*1e6)
print(chi*dist/AU)

# direction
l0 = -np.array([np.sin(chi), np.cos(chi), 0])
x = -np.linalg.norm(x - x_obs) * l0 + x_obs

# centroid shift
dte = centroid_shift(x, pl.pos, x_obs, eps, pl.mass, 0, 0)


print('\n-------------------------------')
print(f'centroid shift: {np.rad2deg(dte)*3600*1e6} muas')
print(f'{np.rad2deg(chi)/3*3600*1e6}')


# correct this problem because beta is set to zero by numerical precision
def cs_mod(beta):

    return beta/(beta**2 + 2) * chi

print(f'{np.rad2deg(cs_mod(1.5))*3600*1e6}')
