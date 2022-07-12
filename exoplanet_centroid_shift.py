""" This program computes the centroid shift of a target star due to an exoplanet
orbiting around it.

Creator: mattia
Date: 06/05/22.
"""
from astronavigation.planets import SolarSystem, Body
from astropy import constants
from astronavigation.deflection import *
from astronavigation.read_exo import *


# Define constants
pc = constants.pc.to('km').value
AU = constants.au.to('km').value
c = constants.c.to('km/s').value
eps = 1/c

# create solar system and jupiter
ss = SolarSystem()
jupiter = ss.getPlanet('jupiter')

# target
dist = 10 * pc
x = np.array([0, 1, 0])*dist

for num in np.arange(1, 30, 5):

    # exoplanet
    orb_rad = num * AU
    pl = Body(mass=5*jupiter.mass,
              pos=(dist - orb_rad)*np.array([0, 1, 0]),
              radius=jupiter.radius)

    # observer
    x_obs = AU*np.array([0, -1, 0])


    # impact angle
    n_ring = 1.5
    theta_e = einstein_ring(pl.mass, eps, pl.pos, x, x_obs)
    chi = n_ring*theta_e
    print(f'einstein ring: {np.rad2deg(theta_e)*3600*1e6} muas')
    print(f'einstein ring at distance {dist/pc} pc: {chi*dist} km')

    # direction
    l0 = -np.array([np.sin(chi), np.cos(chi), 0])
    x = -np.linalg.norm(x - x_obs) * l0 + x_obs

    star_rad = np.rad2deg(ss.getSun().radius / dist) * 3600e6
    print(f'star radius from earth: {star_rad} muas')

    # centroid shift
    # use cs_beta and not centroid_shift because numerical precision
    # pushes the value of beta to zero
    dte = cs_beta(n_ring, np.linalg.norm(x - x_obs), np.linalg.norm(x - pl.pos), theta_e)

    print('\n-------------------------------')
    print(f'centroid shift: {np.rad2deg(dte)*3600*1e6} muas\n')
    print('\n-------------------------------')

