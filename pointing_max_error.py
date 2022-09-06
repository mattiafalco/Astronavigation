"""

Creator: mattiafalco
date: 26/08/22
"""
import platform

from astronavigation.deflection import *
from astropy import constants
from astronavigation.planets import SolarSystem
import pandas as pd
import numpy.linalg as LA
from astronavigation.read_exo import getExo
from astronavigation.save_df import save_df
import matplotlib.pyplot as plt


# Define constants
pc = constants.pc.to('km').value
AU = constants.au.to('km').value
c = constants.c.to('km/s').value
eps = 1/c

# read exo catalogue
path = 'exo_archive.csv'
catalogue = pd.read_csv(path)

# save parameter
save = False
save_latex = True

######################################
#
# Write here the data of the problem
#
######################################

# observer
obs = 'earth'
# masses
list_p = ['sun', 'jupiter', 'saturn', 'uranus', 'neptune']
# targets
targets = ['Proxima Cen b', 'Kepler-847 b',
           'OGLE-2014-BLG-0124L b']

#################
#
# Algorithm
#
#################

ss = SolarSystem()

x_obs = ss.getPlanet(obs, anom=-np.pi/2).pos

# take bodies which generate grav. field
planets = []
for pl in list_p:

    if pl != 'earth':
        anom = np.pi/2
    else:
        anom = -np.pi/2

    planets.append(ss.getPlanet(pl, anom=anom))

dl = []  # deflection
dlq = []  # quadrupole deflection

# loop over planets
for pl in planets:

    # target
    x = np.array([0, 1, 0]) * 10 * pc
    # impact angle
    chi = pl.radius / np.linalg.norm(x_obs - pl.pos)
    # direction
    l0 = -np.array([np.sin(chi), np.cos(chi), 0])
    x = -np.linalg.norm(x - x_obs) * l0 + x_obs

    # monopole deflection
    dls = light_defl(l0, x, pl.pos, x_obs, eps, pl.mass, method='RAMOD')
    dl.append(LA.norm(dls))
    # quadrupole deflection
    dls = light_defl(l0, x, pl.pos, x_obs, eps, pl.mass, method='RAMOD', J2=pl.J2, R=pl.radius)
    dlq.append(LA.norm(dls))

dl = np.array(dl)
dlq = np.array(dlq)
print(rad2muas(dl), rad2muas(dlq))

point_errors = []
point_errors_q = []
for target in targets:
    dist = getExo(target, catalogue).dist
    print(target, dist/pc)

    dr = dl*dist
    point_errors.append(dr)
    print(f'dr: {dr/AU} AU')

    drq = dlq * dist
    point_errors_q.append(drq)
    print(f'drq: {drq / AU} AU')

if save_latex:
    columns = targets
    index = list_p
    path = f'Data/pointing_max_error_latex'
    save_df([np.round(point_errors[0]/AU, 4),
             np.round(point_errors[1] / AU, 4),
             np.round(point_errors[2]/AU, 4)],
            columns=columns,
            index=index,
            path=path)

    columns = targets
    index = list_p
    path = f'Data/pointing_max_error_q_latex'
    save_df([np.round(point_errors_q[0] / AU, 4),
             np.round(point_errors_q[1] / AU, 4),
             np.round(point_errors_q[2] / AU, 4)],
            columns=columns,
            index=index,
            path=path)



