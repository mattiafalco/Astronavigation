""" Description

Creator: mattiafalco
Date: 13/04/22 
"""

import numpy as np
import pandas as pd
from read_exo import getExo, getExoAll
from planets import SolarSystem, Body
from astropy import constants
from deflection import deflection_mod

# Define constants
pc = constants.pc.to('km').value
AU = constants.au.to('km').value
c = constants.c.to('km/s').value
eps = 1/c

# create solar system and jupiter
ss = SolarSystem()
jupiter = ss.getPlanet('jupiter')

# print mean values
path = 'exo_archive.csv'
catalogue = pd.read_csv(path)
mean_val = catalogue.mean(numeric_only=True)
print(mean_val)

# now we estimate the deflection generated by the mean planet
# we center the coordinate system into the star of the exosystem
pl = Body(mass=mean_val['pl_massj']*jupiter.mass,
          pos=mean_val['pl_orbsmax']*AU*np.array([0, 1, 0]),
          radius=mean_val['pl_radj']*jupiter.radius)

# observer
x_obs = AU*np.array([0, -1, 0])

# target
dist = mean_val['sy_dist'] * pc

# impact angle
chi = pl.radius / np.linalg.norm(x_obs - pl.pos)

# direction
l0 = -np.array([np.sin(chi), np.cos(chi), 0])
x = np.array([0, dist, 0])

# deflection
dls = deflection_mod(l0, x, pl.pos, x_obs, eps, pl.vel, pl.mass, chi)
dl1 = np.linalg.norm(dls)

print(f'deflection: {np.rad2deg(dl1)*3600*1e6} muas')
print(f'dr: {dist*dl1/AU} AU')

