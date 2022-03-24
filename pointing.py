import numpy as np
from deflection import *
from astropy import constants

pc = constants.pc.to('km').value
AU = constants.au.to('km').value
c = constants.c.to('km/s').value
eps = 1/c

# angle of observation
g = -np.pi/2 + constants.R_sun.to('km').value/AU
x_obs = AU*np.array([np.cos(g), np.sin(g), 0])

pos = [778412010, 1426725400, 2870972200, 4498252900]
vel = [13.0697, 9.6724, 6.8352, 5.4778]
# vel = [0, 0, 0, 0]

# Sun data
MS = 132712e6  # [km3/s2]
x_s = np.array([0, 0, 0])
v_s = np.array([0, 0, 0])

# Jupiter data
MJ = 126.687e6  # [km3/s2]
# x_j = np.array([0, 778.479e6, 0])
x_j = np.array([0, pos[0], 0])
v_j = np.array([-vel[0], 0, 0])

# Saturn data
MSa = 37.931e6  # [km3/s2]
# x_sa = np.array([0, 1432.041e6, 0])
x_sa = np.array([0, pos[1], 0])
v_sa = np.array([-vel[1], 0, 0])

# Uranus data
MU = 5.7940e6  # [km3/s2]
# x_u = np.array([0, 2867.043e6, 0])
x_u = np.array([0, pos[2], 0])
v_u = np.array([-vel[2], 0, 0])

# Neptune data
MN = 6.8351e6  # [km3/s2]
# x_n = np.array([0, 4514.953e6, 0])
x_n = np.array([0, pos[3], 0])
v_n = np.array([-vel[3], 0, 0])

sources_M = [MS, MJ, MSa, MU, MN]
sources_x = [x_s, x_j, x_sa, x_u, x_n]
sources_v = [v_s, v_j, v_sa, v_u, v_n]

# star
dist = 2300*pc
x = np.array([0, dist, 0])
l0 = -(x - x_obs) / (np.linalg.norm(x - x_obs))

dl1 = np.zeros(len(sources_x))
dpsi = np.zeros(len(sources_x))
dr = np.zeros(len(sources_x))
for i in range(len(sources_x)):
    dls = deflection(l0, x, sources_x[i], x_obs, eps, sources_v[i], sources_M[i])
    dl1[i] = np.linalg.norm(dls)
    # dln = dls - l0*np.dot(dls, l0)
    # dpsi[i] = np.rad2deg(np.linalg.norm(dln))*3600*1e6

dlt = np.cumsum(dl1)
# dpsi_tot = [sum(dpsi[0:i+1]) for i in range(0, len(dpsi))]
dll = dlt - dlt[0]*np.ones_like(dlt)
# dpsi_rel = dpsi_tot - dpsi_tot[0]
# print(np.rad2deg(dll)*3600*1e6)
# print(dpsi_rel)

# point error
dr = [np.linalg.norm(x-x_obs)*dlt[i] for i in range(0, len(dlt))]
print(f'pointing errors: {dr/AU} AU')
# print(np.rad2deg(dl1)*3600*1e6)
print(dlt)

# quadrupole
# Jupiter
sJ2 = quadru(1)
R = 71492  # [km]
dls = deflection(l0, x, x_j, x_obs, eps, v_j, MJ, sJ2[:3], sJ2[3], R)
dlq = np.linalg.norm(dls)
drq = np.linalg.norm(x-x_obs)*dlq
print(f'quadrupole_jup: {(drq)/AU} AU')

# Saturn
sJ2 = quadru(2)
R = 60268  # [km]
dls = deflection(l0, x, x_sa, x_obs, eps, v_sa, MSa, sJ2[:3], sJ2[3], R)
dlq = np.linalg.norm(dls)
drq = np.linalg.norm(x-x_obs)*dlq
print(f'quadrupole_sat: {(drq)/AU} AU')

print(f'dl1: {np.rad2deg(dl1)*3600*1e6}')
print(f'dlt: {np.rad2deg(dlt)*3600*1e6}')

