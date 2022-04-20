""" Description

Creator: Mattia Falco
Date: 20/04/2022
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
save = True

######################################
#
# Write here the data of the problem
#
######################################

# angle of observation
g = -np.pi/2

# masses
list_p = ['sun', 'jupiter', 'saturn', 'uranus', 'neptune']

# targets
targets = ['Proxima Cen b']
dist = np.array([getExo(pl, data).dist for pl in targets])

# Create Solar System
ss = SolarSystem()

# useful values
m_sun = ss.getSun().mass
r_sun = ss.getSun().radius
m_jup = ss.getPlanet('jupiter').mass
r_jup = ss.getPlanet('jupiter').radius
J2_jup = ss.getPlanet('jupiter').J2

# exo sources
list_exo = ['bh_7m']

bh_7m = Body(mass=7*m_sun,
             pos=np.array([0, 10, 0])*pc,
             radius=r_sun,
             J2=J2_jup)

exos = [bh_7m]

#################
#
# Algorithm
#
#################


# observer
x_obs = AU*np.array([np.cos(g), np.sin(g), 0])

# take bodies which generate grav. field
planets = []
for pl in list_p:

    if pl != 'earth':
        anom = np.pi/2
    else:
        anom = -np.pi/2

    planets.append(ss.getPlanet(pl, anom=anom))

v_null = np.array([0, 0, 0])

# add exos
planets += exos

# targets
x_stars = [np.array([0, d, 0]) for d in dist]

# external loop on the targets
for x in x_stars:

    # internal loop on the masses, evaluate deflections
    dl1 = []
    dl2 = []  # w/ null velocities
    dlq = []  # standard quadrupole
    dl_er = []  # Erez-Rosen
    dlq_er = []  # Erez-Rosen quadrupole
    cs = []  # centroid-shift

    for pl in planets:
        # impact angle
        chi = pl.radius / np.linalg.norm(x_obs - pl.pos)
        print(f'chi: {chi}')
        # direction
        l0 = -np.array([np.sin(chi), np.cos(chi), 0])
        x = -np.linalg.norm(x - x_obs) * l0 + x_obs
        # deflection
        dls = deflection_mod(l0, x, pl.pos, x_obs, eps, pl.vel, pl.mass, chi)
        dl1.append(np.linalg.norm(dls))
        # deflection w/ null velocities
        dls = deflection_mod(l0, x, pl.pos, x_obs, eps, v_null, pl.mass, chi)
        dl2.append(np.linalg.norm(dls))
        # deflection quadrupole
        dls = deflection_mod(l0, x, pl.pos, x_obs, eps, v_null, pl.mass, chi, pl.s, pl.J2, pl.radius)
        dlq.append(np.linalg.norm(dls))
        # deflection Erez-Rosen
        dls = er_deflection(l0, x, pl.pos, x_obs, eps, pl.mass, pl.J2, pl.radius, quad=False)
        dl_er.append(np.linalg.norm(dls))
        # deflection quadrupole Erez-Rosen
        dls = er_deflection(l0, x, pl.pos, x_obs, eps, pl.mass, pl.J2, pl.radius)
        dlq_er.append(np.linalg.norm(dls) - dl_er[-1])  # subtract the monopole contribution
        # centroid shift
        delta = centroid_shift(x, pl.pos, x_obs, eps, pl.mass, pl.J2, pl.radius)
        cs.append(delta)


    print(f'\n---------------------------------\nexoplanet at {np.linalg.norm(x)/pc} pc')
    print(f'angle errors: {np.rad2deg(dl1)*3600*1e6} muas')
    print(f'angle errors v null: {np.rad2deg(dl2) * 3600 * 1e6} muas')
    print(f'angle errors er: {np.rad2deg(dl_er) * 3600 * 1e6} muas')
    print(f'quadrupole: {np.rad2deg(dlq) * 3600 * 1e6} muas')
    print(f'quadrupole er: {np.rad2deg(dlq_er) * 3600 * 1e6} muas')
    print(f'centroid shift: {np.rad2deg(cs) * 3600 * 1e6} muas')


    # saving
    if save:
        rows = list_p + list_exo
        columns = ['dl_vn', 'dl', 'dl_er', 'dlq', 'dlq_er', 'centroid']
        data = [np.rad2deg(dl2) * 3600 * 1e6,
                np.rad2deg(dl1) * 3600 * 1e6,
                np.rad2deg(dl_er) * 3600 * 1e6,
                np.rad2deg(dlq) * 3600 * 1e6,
                np.rad2deg(dlq_er) * 3600 * 1e6,
                np.rad2deg(cs) * 3600 * 1e6]
        path = f'Data/comparison'
        save_df(data, columns, rows, path)



