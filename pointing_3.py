import numpy as np
from deflection import *
from astropy import constants
from planets_2 import Body, SolarSystem

pc = constants.pc.to('km').value
AU = constants.au.to('km').value
c = constants.c.to('km/s').value
# AU = 149597870.691  # [km]
# pc = 3.0856775814672e13  # [km]
# c = 299792.458  # [km/s]
eps = 1/c

# angle of observation
g = -np.pi/2 + 6.955e5/AU  # constants.R_sun.to('km').value/AU
x_obs = AU*np.array([np.cos(g), np.sin(g), 0])

# star
dist = 2300*pc
x = np.array([0, dist, 0])
l0 = -(x - x_obs) / (np.linalg.norm(x - x_obs))

list_p = ['sun', 'jupiter', 'saturn', 'uranus', 'neptune']
ss = SolarSystem()
planets = [ss.getPlanet(pl) for pl in list_p]

dl1 = []
for pl in planets:
    dls = deflection(l0, x, pl.pos, x_obs, eps, pl.vel, pl.mass)
    dl1.append(np.linalg.norm(dls))
dl1 = np.array(dl1)

dlt = np.cumsum(dl1)
# dpsi_tot = [sum(dpsi[0:i+1]) for i in range(0, len(dpsi))]
# dll = dlt - dlt[0]*np.ones_like(dlt)
# dpsi_rel = dpsi_tot - dpsi_tot[0]
# print(f'dll = {np.rad2deg(dll)*3600*1e6} muas')
# print(dpsi_rel)

# point error
dr = np.linalg.norm(x-x_obs)*dlt
print(f'pointing errors: {dr/AU} AU')

# quadrupole Jupiter-Saturn
list_q = ['jupiter', 'saturn']
planets_q = [ss.getPlanet(pl) for pl in list_q]
dlq = []
for pl in planets_q:
    dls = deflection(l0, x, pl.pos, x_obs, eps, pl.vel, pl.mass, pl.s, pl.J2, pl.radius)
    dlq.append(np.linalg.norm(dls))
dlq = np.cumsum(np.array(dlq))
drq = np.linalg.norm(x-x_obs)*dlq
print(f'quadrupole jup-sat: {drq/AU} AU')


