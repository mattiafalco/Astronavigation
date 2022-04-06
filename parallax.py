""" Description

Creator: Mattia Falco
Date: 06/04/2022 
"""

import numpy as np
from deflection import *
from astropy import constants
from planets import Body, SolarSystem
import matplotlib.pyplot as plt

# Define constants
pc = constants.pc.to('km').value
AU = constants.au.to('km').value
c = constants.c.to('km/s').value
GM_sun = constants.GM_sun.to('km3/s2').value
eps = 1/c

######################################
#
# Write here the data of the problem
#
######################################

# black hole
bh = Body(mass=7.1*GM_sun,
          pos=np.array([-4, -1580, -45])*pc,
          vel=np.array([3, 0, 40]))

# observer
obs = 'earth'
# masses

# target
dist = 3000*pc

# Time
t_span = np.arange(0, 365)  # day

#################
#
# Algorithm
#
#################

# Create Solar System
ss = SolarSystem()

# target
ll = bh.pos/bh.dist
the = 20*np.sqrt(4*bh.mass*(dist - bh.dist)/(dist * bh.dist))/c
l_tar = np.array([ll[0]*np.cos(the) - ll[1]*np.sin(the),
                  ll[0]*np.sin(the) + ll[1]*np.cos(the),
                  ll[2]])
x1 = l_tar*dist
x2 = ll*dist
print(np.rad2deg(the) * 3600 * 1e6)

x = x1

dl_time = []
for t in t_span:

    anom = ss.getPlanet(obs).speed/ss.getPlanet(obs).dist * t * 24 * 3600
    # print(f' anom: {anom}')

    # observer
    x_obs = ss.getPlanet(obs, anom=anom).pos
    # print(f'x_obs: {x_obs}')

    l0 = -(x - x_obs) / (np.linalg.norm(x - x_obs))

    dls = deflection(l0, x, bh.pos, x_obs, eps, bh.vel, bh.mass)
    # dls = deflection(l0, x, pl.pos, x_obs, eps, np.array([0,0,0]), pl.mass)
    dl1 = np.linalg.norm(dls)

    dl_time.append(dl1)

    #################
    #
    # Printing
    #
    #################

    #print(f'day = {t}')
    #print(f'angle errors: {np.rad2deg(dl1) * 3600 * 1e6} muas\n')

#################
#
# Plot
#
#################
dl_muas = np.rad2deg(np.array(dl_time)) * 3600 * 1e6

fig, ax = plt.subplots()
ax.plot(t_span, dl_muas)

fig2 = plt.figure()
ax2 = plt.axes(projection='3d')
ax2.plot(0.0, 0.0, 0.0, marker='^', color='blue')
#ax2.plot(x_obs[0], x_obs[1], x_obs[2], marker='o')
ax2.plot(x[0], x[1], x[2], marker='o', color='red')
ax2.plot(bh.pos[0], bh.pos[1], bh.pos[2], marker='o', color='black')
ax2.plot(x2[0], x2[1], x2[2], marker='o', color='green')


plt.show()




