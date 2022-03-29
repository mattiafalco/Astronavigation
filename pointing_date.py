import numpy as np
from deflection import *
from astropy import constants
from planets_2 import Body, SolarSystem

# Define constants
pc = constants.pc.to('km').value
AU = constants.au.to('km').value
c = constants.c.to('km/s').value
eps = 1/c

######################################
#
# Write here the data of the problem
#
######################################

# observer
obs = 'earth'
# masses
list_p = ['sun', 'jupiter', 'saturn', 'uranus', 'neptune']
# target
dist = 1.3012*pc  # Proxima Cen b
ra = 217.3934657  # deg
dec = -62.6761821  # deg
# date
date = '1996-02-25T12:00:00'

#################
#
# Algorithm
#
#################

# Create Solar System with ephemerides
ss = SolarSystem(ephemerides=True)

# observer
x_obs = ss.getPlanet(obs, date=date).pos

# target
x = dist*cartesian(ra, dec)
l0 = -(x - x_obs) / (np.linalg.norm(x - x_obs))

# take bodies which generate grav. field
planets = [ss.getPlanet(pl, date=date) for pl in list_p]

# Evaluate deflection for each body
dl1 = []
dpsi = []
dl1_q = []
dpsi_q = []
for pl in planets:
    # monopole deflection
    dls = deflection(l0, x, pl.pos, x_obs, eps, pl.vel, pl.mass)
    # quadrupole deflection
    dlq = deflection(l0, x, pl.pos, x_obs, eps, pl.vel, pl.mass, pl.s, pl.J2, pl.radius)

    # save norm values in lists
    dl1.append(np.linalg.norm(dls))
    dl1_q.append(np.linalg.norm(dlq))

    # projections orthogonal to l0
    dln = dls - l0*np.dot(dls, l0)
    dln_q = dlq - l0 * np.dot(dlq, l0)

    # save norm values
    dpsi.append(np.linalg.norm(dln))
    dpsi_q.append(np.linalg.norm(dln_q))

# transform lists into array
dl1 = np.array(dl1)
dpsi = np.array(dpsi)
dl1_q = np.array(dl1_q)
dpsi_q = np.array(dpsi_q)

# Take the cumulative sum
dlt = np.cumsum(dl1)
dpsi_tot = np.cumsum(dpsi)

#################
#
# Printing
#
#################
print(f'dl: {np.rad2deg(dlt)*3600*1e6} muas')
print(f'dl - sun: {np.rad2deg(dlt-dlt[0])*3600*1e6} muas')

# point error
dr = np.linalg.norm(x-x_obs)*dlt
print(f'pointing errors: {dr/AU} AU')
print(f'pointing errors - sun: {(dr-dr[0])/AU} AU')

# quadrupole
print(f'dl quadrupole: {np.rad2deg(dl1_q)*3600*1e6} muas')
