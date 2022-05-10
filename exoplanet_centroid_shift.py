""" Description

Creator: mattia
Date: 06/05/22.
"""
from astronavigation.planets import SolarSystem, Body
from astropy import constants
from astronavigation.deflection import *

# Define constants
pc = constants.pc.to('km').value
AU = constants.au.to('km').value
c = constants.c.to('km/s').value
eps = 1/c

# create solar system and jupiter
ss = SolarSystem()
jupiter = ss.getPlanet('jupiter')

# target
dist = 1 * pc
x = np.array([0, 1, 0])*dist

# exoplanet
orb_rad = 1 * AU
pl = Body(mass=5*jupiter.mass,
          pos=(dist - orb_rad)*np.array([0, 1, 0]),
          radius=jupiter.radius)

# observer
x_obs = AU*np.array([0, -1, 0])


# impact angle
chi = einstein_ring(pl.mass, eps, pl.pos, x)
print(f'einstein ring: {np.rad2deg(chi)*3600*1e6} muas')
print(f'einstein ring at distance {dist/pc} pc: {chi*dist} km')

# direction
l0 = -np.array([np.sin(chi), np.cos(chi), 0])
x = -np.linalg.norm(x - x_obs) * l0 + x_obs

# centroid shift
# use cs_beta and not centroid_shift because numerical precision
# pushes the value of beta to zero
dte = cs_beta(3, np.linalg.norm(x - x_obs), 4 * np.linalg.norm(x - pl.pos), chi)

print('\n-------------------------------')
print(f'centroid shift: {np.rad2deg(dte)*3600*1e6} muas')

