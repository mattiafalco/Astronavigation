"""

Creator: mattiafalco
date: 19/07/22
"""

import numpy as np

from astronavigation.deflection import *
from astropy import constants
from astronavigation.planets import Body, SolarSystem
import pandas as pd
from astronavigation.save_df import save_df

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

grazing = False

# angle of observation
g = -np.pi/2 if grazing else -np.pi/2 + np.deg2rad(3600/3600)

# masses
list_p = ['jupiter']

# target
dist = 1.0*pc

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

# loop over planets
dl = []
dl_v = []
for pl in planets:

    if grazing:
        chi = 1*pl.radius / np.linalg.norm(x_obs - pl.pos)

        # direction
        l0 = -np.array([np.sin(chi), np.cos(chi), 0])
        x = -np.linalg.norm(x - x_obs) * l0 + x_obs
    else:
        chi = 0

    b = (x_obs - pl.pos) - l0 * np.dot(x_obs - pl.pos, l0)
    bv = np.cross(l0, np.cross(pl.vel, l0))
    sigma = np.dot(x-x_obs, l0)
    r_obs = x_obs - pl.pos
    r = x-pl.pos
    print(f'v.l0: {bv}\n'
          f'bv: {np.cross(l0, np.cross(pl.vel, l0))}\n'
          f'r_obs.v: {np.dot(x_obs - pl.pos, pl.vel)}\n'
          f'v.b: {np.dot(b, pl.vel)}\n'
          f'b: {b}\n'
          f'sigma: {sigma} km\n'
          f'p5/r: {(np.linalg.norm(r)- np.linalg.norm(r_obs)-sigma)/np.linalg.norm(r)}')


    vv = 2*pl.mass*eps**3 * pl.speed * np.linalg.norm(r_obs) / \
         (np.linalg.norm(b)**2 * np.linalg.norm(r)) * (np.linalg.norm(r) - np.linalg.norm(r_obs) - sigma)

    print(f'vv: {rad2muas(vv)}\n')

    # v_null
    dls = deflection(l0, x, pl.pos, x_obs, eps, pl.mass)
    dl.append(np.linalg.norm(dls))
    # vel
    dls = deflection(l0, x, pl.pos, x_obs, eps, pl.mass, pl.vel)
    dl_v.append(np.linalg.norm(dls))

dl = np.array(dl)
dl_v = np.array(dl_v)

print(f'chi: {rad2muas(chi)} muas\n'
      f'dl: {rad2muas(dl)} muas\n'
      f'dl_v: {rad2muas(dl_v - dl)} muas')

jupiter = ss.getPlanet('jupiter')
xx = rad2muas(4*jupiter.mass/(c**2 * jupiter.radius)*6*(AU/jupiter.radius)*(13.0697/c))
print(f'xx: {xx} muas')
# aa = rad2muas(4*jupiter.mass/(c**2 * jupiter.radius))
# bb = rad2muas(AU/jupiter.radius)
# cc = rad2muas(13.0697/c)
# print(f'aa: {aa} muas\n'
#       f'bb: {bb} muas\n'
#       f'cc: {cc} muas')

