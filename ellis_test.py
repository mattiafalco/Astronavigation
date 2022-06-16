""" This program evaluates the Ellis deflection for wormholes of different radius

Creator: mattiafalco
date: 15/06/22
"""



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

# read exo catalogue
path = 'exo_archive.csv'
data = pd.read_csv(path)

# Create Solar System
ss = SolarSystem()

# useful values
m_sun = ss.getSun().mass
r_sun = ss.getSun().radius
m_jup = ss.getPlanet('jupiter').mass
r_jup = ss.getPlanet('jupiter').radius
J2_jup = ss.getPlanet('jupiter').J2

# save parameter
save_latex = True

######################################
#
# Write here the data of the problem
#
######################################

# angle of observation
g = -np.pi/2

# targets
dist = 10 * pc

radii = np.array([1, 5, 10, 15, 20])

worms = [Body(mass=0,
              pos=np.array([0, 3, 0])*pc,
              radius=n*r_jup) for n in radii]

#################
#
# Algorithm
#
#################

# observer
x_obs = AU*np.array([np.cos(g), np.sin(g), 0])

# targets
x = np.array([0, 1, 0]) * dist

dl = []
rings = []

# external loop on the targets
for body in worms:

    # ellis wormhole deflection
    chi = 1 * np.cbrt(np.pi / 4 * np.linalg.norm(x - body.pos) * body.radius ** 2
                  / (np.linalg.norm(x - x_obs) * np.linalg.norm(body.pos - x_obs) ** 2))
    l0 = -np.array([np.sin(chi), np.cos(chi), 0])
    x_el = body.dist * l0 + x_obs
    dls = ellis_deflection(l0, x_el, body.pos, x_obs, body.radius)
    dl.append(dls)
    rings.append(chi)
    # print(f'd: {chi*np.linalg.norm(worm_1j.pos - x_obs)} km')


    print(f'\nellis chi: {np.rad2deg(chi) * 3600 * 1e6} muas')
    print(f'ellis wormhole: {np.rad2deg(dls)*3600*1e6} muas\n')

dl = np.array(dl)

# saving
if save_latex:
    rows = radii
    columns = ['chi', 'defl']
    data = [np.round(np.rad2deg(rings) * 3600 * 1e6, 2), np.round(np.rad2deg(dl) * 3600 * 1e6, 2)]
    path = f'Data/ellis_latex'
    save_df(data, columns, rows, path)
