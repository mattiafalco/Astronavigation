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
list_p = ['sun', 'earth', 'jupiter', 'saturn', 'uranus', 'neptune']
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
x_obs = (AU+380000)*np.array([np.cos(g), np.sin(g), 0])

# take bodies which generate grav. field
# planets = [ss.getPlanet(pl) for pl in list_p]
planets = []
for pl in list_p:

    if pl != 'earth':
        anom = np.pi/2
    else:
        anom = -np.pi/2

    planets.append(ss.getPlanet(pl, anom=anom))

v_null = np.array([0, 0, 0])

# targets
x_stars = [np.array([0, d, 0]) for d in dist]

# external loop on the targets
for x in x_stars:
    # internal loop on the masses, evaluate deflections
    dl1 = []
    dl2 = []  # w/ null velocities
    for pl in planets:
        # impact angle
        chi = pl.radius / np.linalg.norm(x_obs - pl.pos)
        # direction
        l0 = -np.array([np.sin(chi), np.cos(chi), 0])
        # deflection
        dls = deflection_mod(l0, x, pl.pos, x_obs, eps, pl.vel, pl.mass, chi)
        dl1.append(np.linalg.norm(dls))
        # deflection w/ null velocities
        dls = deflection_mod(l0, x, pl.pos, x_obs, eps, v_null, pl.mass, chi)
        dl2.append(np.linalg.norm(dls))

    dlt = np.cumsum(dl1)
    dlt_vn = np.cumsum(dl2)

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
        columns = ['dl_vn', 'dl', 'dlt', 'dlt - sun', 'dr', 'dr - sun']
        data = [np.rad2deg(dl2)*3600*1e6,
                np.rad2deg(dl1)*3600*1e6,
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
    for pl in planets_q:
        # impact angle
        chi = pl.radius/np.linalg.norm(x_obs-pl.pos)
        # direction
        l0q = -np.array([np.sin(chi), np.cos(chi), 0])
        # deflection
        dls = deflection_mod(l0q, x, pl.pos, x_obs, eps, pl.vel, pl.mass, chi,
                             pl.s, pl.J2, pl.radius)
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


print(np.rad2deg(4*ss.getPlanet('jupiter').mass/c**2/ss.getPlanet('jupiter').radius)*3600*1e6)
