import numpy as np
from deflection import *
from astropy import constants
from planets import Body, SolarSystem

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

# angle of observation
g = -np.pi/2 + 6.955e5/AU  # constants.R_sun.to('km').value/AU
# masses
list_p = ['sun', 'jupiter', 'saturn', 'uranus', 'neptune']
# masses for quadrupole contribution
list_q = ['jupiter', 'saturn']
# target
dist = 1.3012*pc

#################
#
# Algorithm
#
#################

# Create Solar System
ss = SolarSystem()

# observer
x_obs = AU*np.array([np.cos(g), np.sin(g), 0])

# target
x = np.array([0, dist, 0])
l0 = -(x - x_obs) / (np.linalg.norm(x - x_obs))

# take bodies which generate grav. field
planets = [ss.getPlanet(pl) for pl in list_p]

# Evaluate deflection for each body
dl1 = []
dpsi = []
for pl in planets:
    dls = deflection(l0, x, pl.pos, x_obs, eps, pl.vel, pl.mass)
    dl1.append(np.linalg.norm(dls))
    dln = dls - l0*np.dot(dls, l0)
    dpsi.append(np.linalg.norm(dln))
dl1 = np.array(dl1)
dpsi = np.array(dpsi)

# Take the cumulative sum
dlt = np.cumsum(dl1)
dpsi_tot = np.cumsum(dpsi)


#################
#
# Printing
#
#################
# point error
dr = np.linalg.norm(x-x_obs)*dlt
print(f'angle errors: {np.rad2deg(dlt) * 3600 * 1e6} muas')
print(f'angle errors - sun: {np.rad2deg(dlt - dlt[0]) * 3600 * 1e6} muas')
print(f'pointing errors: {dr / AU} AU')
print(f'pointing errors - sun: {(dr - dr[0]) / AU} AU')

print(f'dpsi: {np.rad2deg(dpsi_tot) * 3600 * 1e6} muas')
print(f'dpsi - sun: {np.rad2deg(dpsi_tot - dpsi_tot[0]) * 3600 * 1e6} muas')


###########################
#
# Quadrupole contribution
#
###########################
planets_q = [ss.getPlanet(pl) for pl in list_q]
dlq = []
for pl in planets_q:
    dls = deflection(l0, x, pl.pos, x_obs, eps, pl.vel, pl.mass, pl.s, pl.J2, pl.radius)
    dlq.append(np.linalg.norm(dls))
dlq = np.cumsum(np.array(dlq))
drq = np.linalg.norm(x-x_obs)*dlq
print(f'quadrupole jup-sat: {drq/AU} AU')


