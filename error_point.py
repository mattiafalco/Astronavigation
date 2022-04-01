import numpy as np
from deflection import *
from astropy import constants
from planets import Body, SolarSystem
import pandas as pd
from read_exo import getExo
from save_df import save_df

# Define constants
pc = constants.pc.to('km').value
AU = constants.au.to('km').value
c = constants.c.to('km/s').value
eps = 1/c

# read exo catalogue
path = 'exo_archive.csv'
data = pd.read_csv(path)

# save parameter
save = True

######################################
#
# Write here the data of the problem
#
######################################

# angle of observation
g = -np.pi/2
# masses
list_p = ['sun', 'jupiter', 'saturn', 'uranus', 'neptune']
# targets
targets = ['Proxima Cen b', 'K2-100 b', 'Kepler-943 b']
dist = np.array([getExo(pl, data).dist for pl in targets])
# dist = np.array([1.3012, 188, 2300])*pc

#################
#
# Algorithm
#
#################

# Create Solar System
ss = SolarSystem()

# observer
x_obs = AU*np.array([np.cos(g), np.sin(g), 0])

# take bodies which generate grav. field
planets = [ss.getPlanet(pl) for pl in list_p]

# planets data
x_p = [planet.pos for planet in planets]
v_p = [planet.vel for planet in planets]
m_p = [planet.mass for planet in planets]
r_p = [planet.radius for planet in planets]
# evaluate impact angles
chi_p = [r_p[i]/np.linalg.norm(x_obs - x_p[i]) for i in range(len(planets))]
print(f'chi: {chi_p}')

# targets
x_stars = [np.array([0, d, 0]) for d in dist]
l0 = [-np.array([np.sin(chi), np.cos(chi), 0]) for chi in chi_p]

# external loop on the targets
dl1 = np.zeros(len(x_p))
for x in x_stars:
    # internal loop on the masses, evaluate deflections
    for i in range(len(x_p)):
        dls = deflection_mod(l0[i], x, x_p[i], x_obs, eps, v_p[i], m_p[i], chi_p[i])
        dl1[i] = np.linalg.norm(dls)

    dlt = np.cumsum(dl1)

    # point error
    dr = np.linalg.norm(x-x_obs)*dlt
    print(f'\n---------------------------------\nexoplanet at {np.linalg.norm(x)/pc} pc')
    print(f'angle errors: {np.rad2deg(dlt)*3600*1e6} muas')
    print(f'angle errors - sun: {np.rad2deg(dlt-dlt[0]) * 3600 * 1e6} muas')
    print(f'pointing errors: {dr/AU} AU')
    print(f'pointing errors - sun: {(dr-dr[0])/AU} AU')

    # saving
    if save:
        rows = list_p
        columns = ['dl', 'dlt', 'dlt - sun', 'dr', 'dr - sun']
        data = [np.rad2deg(dl1)*3600*1e6,
                np.rad2deg(dlt)*3600*1e6,
                np.rad2deg(dlt-dlt[0])*3600*1e6,
                dr/AU,
                (dr-dr[0])/AU]
        path = f'Data/max_errors_{np.round(np.linalg.norm(x) / pc, 4)}_pc'
        save_df(data, columns, rows, path)

    # quadrupole jupiter-saturn
    list_q = ['jupiter', 'saturn', 'uranus', 'neptune']
    planets_q = [ss.getPlanet(pl) for pl in list_q]

    dl_q = []
    # internal loop on the masses, evaluate quadrupole deflections
    for i in range(len(planets_q)):
        chi = planets_q[i].radius/np.linalg.norm(x_obs-planets_q[i].pos)
        l0q = -np.array([np.sin(chi), np.cos(chi), 0])
        dls = deflection_mod(l0q, x, planets_q[i].pos, x_obs, eps, planets_q[i].vel, planets_q[i].mass, chi,
                             planets_q[i].s, planets_q[i].J2, planets_q[i].radius)
        dl_q.append(np.linalg.norm(dls))
    dlt_q = np.cumsum(np.array(dl_q))
    dr_q = np.linalg.norm(x-x_obs)*dlt_q
    print(f'quadrupole jup-sat: {dr_q/AU} AU')

    # saving
    if save:
        rows = list_q
        columns = ['dl', 'dlt', 'dr']
        data = [np.rad2deg(dl_q) * 3600 * 1e6,
                np.rad2deg(dlt_q) * 3600 * 1e6,
                dr_q / AU]
        path = f'Data/max_errors_quad_{np.round(np.linalg.norm(x) / pc, 4)}_pc'
        save_df(data, columns, rows, path)
