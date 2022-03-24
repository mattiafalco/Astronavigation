import numpy as np
from deflection import *
from astropy import constants
from planets_2 import Body, SolarSystem

# pc = constants.pc.to('km').value
#AU = constants.au.to('km').value
# c = constants.c.to('km/s').value
AU = 149597870.691  # [km]
pc = 3.0856775814672e13  # [km]
c = 299792.458  # [km/s]
eps = 1/c

# angle of observation
g = -np.pi/2
x_obs = AU*np.array([np.cos(g), np.sin(g), 0])

list_p = ['sun', 'jupiter', 'saturn', 'uranus', 'neptune']
ss = SolarSystem()
sun = ss.getSun()
planets = [ss.getPlanet(pl) for pl in list_p]

# planets data
x_p = [planets[i].pos for i in range(len(planets))]
v_p = [planets[i].vel for i in range(len(planets))]
m_p = [planets[i].mass for i in range(len(planets))]
r_p = [planets[i].radius for i in range(len(planets))]
chi_p = [r_p[i]/np.linalg.norm(x_obs - x_p[i]) for i in range(len(planets))]
print(f'chi: {chi_p}')

# star
dist = 1.3012*pc
x = np.array([0, dist, 0])
l0 = -(x - x_obs) / (np.linalg.norm(x - x_obs))

dl1 = np.zeros(len(x_p))
dpsi = np.zeros(len(x_p))
dr = np.zeros(len(x_p))
for i in range(len(x_p)):
    dls = deflection_mod(l0, x, x_p[i], x_obs, eps, v_p[i], m_p[i], chi_p[i])
    dl1[i] = np.linalg.norm(dls)
    # dln = dls - l0*np.dot(dls, l0)
    # dpsi[i] = np.rad2deg(np.linalg.norm(dln))*3600*1e6

dlt = np.cumsum(dl1)
# dpsi_tot = [sum(dpsi[0:i+1]) for i in range(0, len(dpsi))]
# dll = dlt - dlt[0]*np.ones_like(dlt)
# dpsi_rel = dpsi_tot - dpsi_tot[0]
# print(f'dll = {np.rad2deg(dll)*3600*1e6} muas')
# print(dpsi_rel)

# point error
dr = np.array([np.linalg.norm(x-x_obs)*dlt[i] for i in range(0, len(dlt))])
print(f'pointing errors: {dr/AU} AU')
print(np.rad2deg(dl1)*3600*1e6)


