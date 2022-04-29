""" Description

Creator: Mattia Falco
Date: 21/04/2022
"""

import numpy as np
from deflection import *
from astropy import constants
from planets import Body, SolarSystem
import pandas as pd
from save_df import save_df

# save parameter
save = True

# Define constants
pc = constants.pc.to('km').value
AU = constants.au.to('km').value
c = constants.c.to('km/s').value
eps = 1/c

# create solar system
ss = SolarSystem()

# useful values
m_sun = ss.getSun().mass
r_sun = ss.getSun().radius
m_jup = ss.getPlanet('jupiter').mass
r_jup = ss.getPlanet('jupiter').radius
J2_jup = ss.getPlanet('jupiter').J2
v_null = np.array([0, 0, 0])

path = 'comp_obs_70.csv'

df_targets = pd.read_csv(path)
print(df_targets.columns)

x_star = []
for i in df_targets.index:

    star = df_targets.iloc[i]

    ra = star['ra_s_new']
    dec = star['dec_s_new']
    dist = 1 * pc

    x_star.append(dist * cartesian(ra, dec))

mean_star = np.mean(x_star, axis=0)

################################################################
#
# evaluating deflection for each star due to different bodies
#
################################################################

list_p = ['jupiter', 'bh_7m', 'jupiter4']

jup = ss.getPlanet('jupiter')
jup.pos = mean_star/2
jup.s = np.array([0, 0, 1])

bh_7m = Body(mass=7*m_sun,
             pos=mean_star/2,
             radius=r_jup,
             J2=J2_jup)

jup4 = Body(mass=4*m_jup,
            pos=mean_star/2,
            radius=r_jup,
            J2=J2_jup)

bodies = [jup, bh_7m, jup4]

worm_1j = Body(mass=0,
               pos=mean_star/2,
               radius=r_jup)

# observer on earth
x_obs = mean_star * AU / pc

for pl_name, pl in zip(list_p, bodies):

    print(f'\n{pl_name}\n')

    dl1 = []  # deflection w/ null velocity
    for x in x_star:

        # print impact angle
        chi = np.arccos(np.dot(pl.pos - x_obs, x - x_obs) /
                        (np.linalg.norm(pl.pos - x_obs) * np.linalg.norm(x - x_obs)))
        print(f'chi: {np.rad2deg(chi)*3600} arcs')

        # direction
        l0 = -(x - x_obs) / np.linalg.norm(x - x_obs)

        # deflection
        dls = deflection(l0, x, pl.pos, x_obs, eps, v_null, pl.mass)
        dl1.append(np.linalg.norm(dls))

    if save:
        rows = range(len(x_star))
        columns = ['dl_vn']
        data = [np.rad2deg(dl1) * 3600 * 1e6]
        path = f'Data/{pl_name}_deflections'
        save_df(data, columns, rows, path)

# ellis wormhole
dl1 = []
for x in x_star:

    # direction
    l0 = -(x - x_obs) / np.linalg.norm(x - x_obs)

    # deflection
    dls = ellis_deflection(l0, x, worm_1j.pos, x_obs, worm_1j.radius)
    dl1.append(np.linalg.norm(dls))

if save:
    rows = range(len(x_star))
    columns = ['dl_vn']
    data = [np.rad2deg(dl1) * 3600 * 1e6]
    path = f'Data/ellis_deflections'
    save_df(data, columns, rows, path)

