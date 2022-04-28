"""

Creator: mattiafalco
date: 27/04/22
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

# Create Solar System
ss = SolarSystem()

# save parameter
save = True

######################################
#
# Write here the data of the problem
#
######################################

# angle of observation
positions = [0, -np.pi]

# masses
list_p = ['sun', 'jupiter', 'saturn', 'uranus', 'neptune']

# targets
dist = 10000 * pc


#################
#
# Algorithm
#
#################
count = 1
for g in positions:

    # observer
    x_obs = AU*np.array([np.cos(g), np.sin(g), 0])

    # take bodies which generate grav. field
    planets = [ss.getPlanet(pl) for pl in list_p]

    v_null = np.array([0, 0, 0])

    # targets
    x = np.array([0, dist, 0])

    # internal loop on the masses, evaluate deflections
    dl1 = []
    dl2 = []  # w/ null velocities
    dlq = []  # standard quadrupole
    dl_er = []  # Erez-Rosen
    dl_er_c1 = []  # Erez-Rosen monopole correction
    dlq_er = []  # Erez-Rosen quadrupole
    dlq_er_c2 = []  # Erez-Rosen quadrupole correction
    cs = []  # centroid-shift

    for pl in planets:

        # print impact angle
        chi = np.arccos(np.dot(pl.pos - x_obs, x - x_obs) /
                        (np.linalg.norm(pl.pos - x_obs) * np.linalg.norm(x - x_obs)))
        print(f'chi: {chi}')

        # direction
        l0 = -(x-x_obs)/np.linalg.norm(x-x_obs)

        # deflection
        dls = deflection(l0, x, pl.pos, x_obs, eps, pl.vel, pl.mass)
        dl1.append(np.linalg.norm(dls))
        # deflection w/ null velocities
        dls = deflection(l0, x, pl.pos, x_obs, eps, v_null, pl.mass)
        dl2.append(np.linalg.norm(dls))
        print(dls)
        # deflection quadrupole
        dls = deflection(l0, x, pl.pos, x_obs, eps, v_null, pl.mass, pl.s, pl.J2, pl.radius)
        dlq.append(np.linalg.norm(dls))
        # deflection Erez-Rosen
        dls = er_deflection(l0, x, pl.pos, x_obs, eps, pl.mass, pl.J2, pl.radius, c1=False, quad=False)
        dl_er.append(np.linalg.norm(dls))
        # deflection Erez-Rosen monopole correction
        dls = er_deflection(l0, x, pl.pos, x_obs, eps, pl.mass, pl.J2, pl.radius, quad=False)
        dl_er_c1.append(np.linalg.norm(dls))
        # deflection quadrupole Erez-Rosen
        dls = er_deflection(l0, x, pl.pos, x_obs, eps, pl.mass, pl.J2, pl.radius, c2=False)
        dlq_er.append(np.linalg.norm(dls) - dl_er_c1[-1])  # subtract the monopole contribution
        # deflection quadrupole correction Erez-Rosen
        dls = er_deflection(l0, x, pl.pos, x_obs, eps, pl.mass, pl.J2, pl.radius)
        dlq_er_c2.append(np.linalg.norm(dls) - dl_er_c1[-1])  # subtract the monopole contribution
        # centroid shift
        delta = centroid_shift(x, pl.pos, x_obs, eps, pl.mass, pl.J2, pl.radius)
        cs.append(delta)

    print(f'\n---------------------------------\nexoplanet at {np.linalg.norm(x)/pc} pc')
    print(f'angle errors: {np.rad2deg(dl1)*3600*1e6} muas')
    print(f'angle errors v null: {np.rad2deg(dl2) * 3600 * 1e6} muas')
    print(f'angle errors er: {np.rad2deg(dl_er) * 3600 * 1e6} muas')
    print(f'angle errors er_c1: {np.rad2deg(dl_er_c1) * 3600 * 1e6} muas')
    print(f'quadrupole: {np.rad2deg(dlq) * 3600 * 1e6} muas')
    print(f'quadrupole er: {np.rad2deg(dlq_er) * 3600 * 1e6} muas')
    print(f'quadrupole er_c2: {np.rad2deg(dlq_er_c2) * 3600 * 1e6} muas')
    print(f'centroid shift: {np.rad2deg(cs) * 3600 * 1e6} muas')

    # saving
    if save:
        rows = list_p
        columns = ['dl_vn', 'dl', 'dl_er', 'dl_er_c1', 'dlq', 'dlq_er', 'dlq_er_c2', 'centroid']
        data = [np.rad2deg(dl2) * 3600 * 1e6,
                np.rad2deg(dl1) * 3600 * 1e6,
                np.rad2deg(dl_er) * 3600 * 1e6,
                np.rad2deg(dl_er_c1) * 3600 * 1e6,
                np.rad2deg(dlq) * 3600 * 1e6,
                np.rad2deg(dlq_er) * 3600 * 1e6,
                np.rad2deg(dlq_er_c2) * 3600 * 1e6,
                np.rad2deg(cs) * 3600 * 1e6]
        path = f'Data/parallax{count}'
        save_df(data, columns, rows, path)
        count += 1




