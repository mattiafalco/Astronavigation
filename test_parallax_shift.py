""" Description

Creator: mattia
Date: 06/05/22.
"""

import numpy as np
from deflection import *
from astropy import constants
from planets import Body, SolarSystem
import pandas as pd
from read_exo import getExo
from save_df import save_df

# Define constants
pc = constants.pc.to('km').value
AU = constants.au.to('km').value
c = constants.c.to('km/s').value
eps = 1/c

# read exo catalogue
path = 'exo_archive.csv'
data = pd.read_csv(path)

# save parameter
save = False

######################################
#
# Write here the data of the problem
#
######################################

# angle of observation
obs = 'earth'
# masses
list_p = ['sun', 'jupiter', 'saturn', 'uranus', 'neptune']
# targets
dist = np.array([1, 10, 100])*pc

#################
#
# Algorithm
#
#################

# Create Solar System
ss = SolarSystem()

# observer
x_obs = ss.getPlanet(obs, anom=0).pos

# take bodies which generate grav. field
planets = []
for pl in list_p:

    if pl != 'earth':
        anom = np.pi/2
    else:
        anom = -np.pi/2

    planets.append(ss.getPlanet(pl, anom=anom))

v_null = np.array([0, 0, 0])

# targets
x_stars = [np.array([0, d, 0]) for d in dist]

# external loop on the targets
for x in x_stars:

    dl1 = []  # w/ null velocities
    dl2 = []
    dlq = []  # quadrupole
    par1 = []
    par2 = []
    parq = []

    # internal loop on the masses, evaluate deflections
    for pl in planets:

        # direction
        l0 = -(x - x_obs) / (np.linalg.norm(x - x_obs))

        # deflection w/ null velocities
        dls = deflection(l0, x, pl.pos, x_obs, eps, v_null, pl.mass)
        dl1.append(np.linalg.norm(dls))
        par1.append(parallax_shift(dl1[-1], l0, x_obs))
        # deflection
        dls = deflection(l0, x, pl.pos, x_obs, eps, pl.vel, pl.mass)
        dl2.append(np.linalg.norm(dls))
        par2.append(parallax_shift(dl2[-1], l0, x_obs))
        # quadrupole
        dls = deflection(l0, x, pl.pos, x_obs, eps, v_null, pl.mass, pl.s, pl.J2, pl.radius)
        dlq.append(np.linalg.norm(dls))
        parq.append(parallax_shift(dlq[-1], l0, x_obs))

    # transform into array
    dl1 = np.array(dl1)
    dl2 = np.array(dl2)
    dlq = np.array(dlq)
    par1 = np.array(par1) * pc
    par2 = np.array(par2) * pc
    parq = np.array(parq) * pc

    # distance error
    dd1 = par1 * (np.linalg.norm(x-x_obs)/pc)**2
    dd2 = par2 * (np.linalg.norm(x - x_obs)/pc) ** 2
    ddq = parq * (np.linalg.norm(x - x_obs)/pc) ** 2

    # point error
    dr1 = np.linalg.norm(x-x_obs)*dl1
    dr2 = np.linalg.norm(x-x_obs)*dl2
    drq = np.linalg.norm(x-x_obs)*dlq

    print(f'\n---------------------------------\nexoplanet at {np.linalg.norm(x)/pc} pc')
    print(f'angle errors v_null: {np.rad2deg(dl1)*3600*1e6} muas')
    print(f'angle errors: {np.rad2deg(dl2) * 3600 * 1e6} muas')
    print(f'quadrupole: {np.rad2deg(dlq) * 3600 * 1e6} muas')
    print(f'parallax shift: {par1*1e6} muas')
    print(f'distance error: {dd1*pc/AU} AU')
    print(f'point error: {dr1/AU} AU')

    # saving
    if save:
        rows = list_p
        columns = []
        data = []
        path = f''
        save_df(data, columns, rows, path)




