import numpy as np
from deflection import *
from astropy import constants
from planets_2 import Body, SolarSystem

pc = constants.pc.to('km').value
AU = constants.au.to('km').value
c = constants.c.to('km/s').value
eps = 1/c

ss = SolarSystem(ephemerides=True)
date = '2003-10-31T12:00:00'

# observer
# x_obs = AU*np.array([-1, 0, 0])
x_obs = ss.getPlanet('earth', date).pos

# star
dist = 1.3012*pc
# x = dist*np.array([0, 1, 0])
x = dist*cartesian(217.3934657, -62.6761821)
l0 = -(x - x_obs) / (np.linalg.norm(x - x_obs))

list_p = ['sun', 'jupiter', 'saturn', 'uranus', 'neptune']
planets = [ss.getPlanet(pl, date=date) for pl in list_p]
print(f'planet1: {planets[1]}')

dl1 = []
dpsi = []
for pl in planets:
    dls = deflection(l0, x, pl.pos, x_obs, eps, pl.vel, pl.mass)
    dl1.append(np.linalg.norm(dls))
    dln = dls - l0*np.dot(dls, l0)
    dpsi.append(np.linalg.norm(dln))
dl1 = np.array(dl1)
dpsi = np.array(dpsi)

dlt = np.cumsum(dl1)
dpsi_tot = np.cumsum(dpsi)

print(f'dpsi: {np.rad2deg(dlt)*3600*1e6} muas')
print(f'dpsi - sun: {np.rad2deg(dlt-dlt[0])*3600*1e6} muas')

# point error
dr = np.linalg.norm(x-x_obs)*dlt
print(f'pointing errors: {dr/AU} AU')
print(f'pointing errors - sun: {(dr-dr[0])/AU} AU')

# # quadrupole Jupiter-Saturn
# list_q = ['jupiter', 'saturn']
# planets_q = [ss.getPlanet(pl) for pl in list_q]
# dlq = []
# for pl in planets_q:
#     dls = deflection(l0, x, pl.pos, x_obs, eps, pl.vel, pl.mass, pl.s, pl.J2, pl.radius)
#     dlq.append(np.linalg.norm(dls))
# dlq = np.cumsum(np.array(dlq))
# drq = np.linalg.norm(x-x_obs)*dlq
# print(f'quadrupole jup-sat: {drq/AU} AU')
