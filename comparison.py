""" This program makes a comparison between different formulas and different
contributions of light deflection. For different types of bodies in the Solar System
and also exo bodies. The quantity taken in consideration are:
- deflection w/ null velocity
- deflection w/ velocity
- Erez-Rosen deflection
- Erez-Rosen monopole correction
- quadrupole deflection
- Erez-Rosen quadrupole deflection
- Erez-Rosen quadrupole correction
- Erez-Rosen centroid shift
an Ellis wormhole is also considered.

Creator: Mattia Falco
Date: 20/04/2022
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
save = False
save_latex = False

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
dist = np.array([8000])*pc  # 500, 1000, 2000]) * pc

# exo sources
list_exo = ['bh_7m', 'bh_20m', 'pl_3j', 'pl_7j']

bh_7m = Body(mass=7*m_sun,
             pos=np.array([0, 4000, 0])*pc,
             radius=r_jup,
             J2=20*r_sun/r_jup*J2_jup)

bh_20m = Body(mass=20*m_sun,
              pos=np.array([0, 4000, 0])*pc,
              radius=r_jup,
              J2=20*r_sun/r_jup*J2_jup)

pl_3j = Body(mass=3*m_jup,
             pos=np.array([0, 4000, 0])*pc,
             radius=r_jup,
             J2=3*J2_jup)

pl_7j = Body(mass=7*m_jup,
             pos=np.array([0, 4000, 0])*pc,
             radius=r_jup,
             J2=7*J2_jup)

worm_1j = Body(mass=0,
               pos=np.array([0, 3, 0])*pc,
               radius=r_jup)

exos = [bh_7m, bh_20m, pl_3j, pl_7j]

#################
#
# Algorithm
#
#################


# observer
x_obs = AU*np.array([np.cos(g), np.sin(g), 0])

# take bodies which generate grav. field
planets = []
for pl in list_p:

    if pl != 'earth':
        anom = np.pi/2
    else:
        anom = -np.pi/2

    planets.append(ss.getPlanet(pl, anom=anom))

v_null = np.array([0, 0, 0])
s_test = np.array([0, 0, 1])

# add exos
planets += exos

# targets
x_stars = [np.array([0, d, 0]) for d in dist]

# external loop on the targets
for x_r in x_stars:

    # internal loop on the masses, evaluate deflections
    imp_ang = []
    dl1 = []
    dl2 = []  # w/ null velocities
    dlq = []  # standard quadrupole
    dl_er = []  # Erez-Rosen
    dl_er_c1 = []  # Erez-Rosen monopole correction
    dlq_er = []  # Erez-Rosen quadrupole
    dlq_er_c2 = []  # Erez-Rosen quadrupole correction
    cs = []  # centroid-shift

    count = 1
    for pl in planets:

        # impact angle, use grazing condition for solar system bodies
        # and einstein ring for exo bodies.
        # comment "count += 1" to use always the grazing condition.
        if count <= len(list_p):
            chi = pl.radius / np.linalg.norm(x_obs - pl.pos)
            count += 1
        else:
            # chi = 1 * np.sqrt(4 * pl.mass * (np.linalg.norm(x) - pl.dist) / (np.linalg.norm(x) * pl.dist)) / c
            chi = einstein_ring(pl.mass, eps, pl.pos, x, x_obs)

            # evaluate t_E
            r_E = chi * np.linalg.norm(pl.pos - x_obs)
            t_E = r_E/220
            print(f't_E: {t_E} s = {t_E/3600} h = {t_E/3600/24} d\n')


        print(f'chi: {np.rad2deg(chi)*3600*1e6} muas')
        imp_ang.append(chi)

        # direction
        l0 = -np.array([np.sin(chi), np.cos(chi), 0])
        x = -np.linalg.norm(x_r - x_obs) * l0 + x_obs

        # deflection
        dls = deflection(l0, x, pl.pos, x_obs, eps, pl.mass, pl.vel)
        dl1.append(np.linalg.norm(dls))
        # deflection w/ null velocities
        dls = deflection(l0, x, pl.pos, x_obs, eps, pl.mass, v_null)
        dl2.append(np.linalg.norm(dls))
        # deflection quadrupole
        dls = deflection(l0, x, pl.pos, x_obs, eps, pl.mass, v_null, s_test, pl.J2, pl.radius)
        dlq.append(np.linalg.norm(dls))
        # deflection Erez-Rosen
        dls = er_deflection(l0, x, pl.pos, x_obs, eps, pl.mass, J2=pl.J2, R=pl.radius, c1=False, quad=False)
        dl_er.append(np.linalg.norm(dls))
        # deflection Erez-Rosen monopole correction
        dls = er_deflection(l0, x, pl.pos, x_obs, eps, pl.mass, J2=pl.J2, R=pl.radius, quad=False)
        dl_er_c1.append(np.linalg.norm(dls))
        # deflection quadrupole Erez-Rosen
        dls = er_deflection(l0, x, pl.pos, x_obs, eps, pl.mass, J2=pl.J2, R=pl.radius, c2=False)
        dlq_er.append(np.linalg.norm(dls) - dl_er_c1[-1])  # subtract the monopole contribution
        # deflection quadrupole correction Erez-Rosen
        dls = er_deflection(l0, x, pl.pos, x_obs, eps, pl.mass, J2=pl.J2, R=pl.radius)
        dlq_er_c2.append(np.linalg.norm(dls) - dl_er_c1[-1])  # subtract the monopole contribution
        # centroid shift
        delta = centroid_shift(x, pl.pos, x_obs, eps, pl.mass, pl.J2, pl.radius)
        cs.append(delta)

    # ellis wormhole deflection
    chi_ell = 1 * np.cbrt(np.pi / 4 * np.linalg.norm(x - worm_1j.pos) * worm_1j.radius ** 2
                  / (np.linalg.norm(x - x_obs) * np.linalg.norm(worm_1j.pos - x_obs) ** 2))
    l0 = -np.array([np.sin(chi_ell), np.cos(chi_ell), 0])
    x_el = worm_1j.dist * l0 + x_obs
    dls = ellis_deflection(l0, x_el, worm_1j.pos, x_obs, worm_1j.radius)
    # print(f'd: {chi*np.linalg.norm(worm_1j.pos - x_obs)} km')

    print(f'\n---------------------------------\nexoplanet at {np.linalg.norm(x)/pc} pc')
    print(f'angle errors: {np.rad2deg(dl1)*3600*1e6} muas')
    print(f'angle errors v null: {np.rad2deg(dl2) * 3600 * 1e6} muas')
    print(f'angle errors er: {np.rad2deg(dl_er) * 3600 * 1e6} muas')
    print(f'angle errors er_c1: {np.rad2deg(dl_er_c1) * 3600 * 1e6} muas')
    print(f'quadrupole: {np.rad2deg(dlq) * 3600 * 1e6} muas')
    print(f'quadrupole er: {np.rad2deg(dlq_er) * 3600 * 1e6} muas')
    print(f'quadrupole er_c2: {np.rad2deg(dlq_er_c2) * 3600 * 1e6} muas')
    print(f'centroid shift: {np.rad2deg(cs) * 3600 * 1e6} muas')

    print(f'\nellis chi: {np.rad2deg(chi) * 3600 * 1e6} muas')
    print(f'ellis wormhole: {np.rad2deg(dls)*3600*1e6} muas\n')

    # saving
    if save:
        rows = list_p + list_exo
        columns = ['dl_vn', 'dl', 'dl_er', 'dl_er_c1', 'dlq', 'dlq_er', 'dlq_er_c2', 'centroid']
        data = [np.rad2deg(dl2) * 3600 * 1e6,
                np.rad2deg(dl1) * 3600 * 1e6,
                np.rad2deg(dl_er) * 3600 * 1e6,
                np.rad2deg(dl_er_c1) * 3600 * 1e6,
                np.rad2deg(dlq) * 3600 * 1e6,
                np.rad2deg(dlq_er) * 3600 * 1e6,
                np.rad2deg(dlq_er_c2) * 3600 * 1e6,
                np.rad2deg(cs) * 3600 * 1e6]
        path = f'Data/comparison_d{np.round(np.linalg.norm(x)/pc, 4)}pc'
        save_df(data, columns, rows, path)

    imp_ang = np.array(imp_ang)
    dl_er = np.array(dl_er)
    dl_er_c1 = np.array(dl_er_c1)
    dlq_er = np.array(dlq_er)
    dlq_er_c2 = np.array(dlq_er_c2)
    # saving
    if save_latex:
        rows = ['sun', 'jupiter', 'saturn', 'uranus', 'neptune', 'bh $7m_\odot$', 'bh $20m_\odot$', 'pl $3m_J$', 'pl $7m_J$']
        columns = ['chi', 'dl_vn', 'dl', 'dl_er', 'dl_er_c1', 'dlq', 'dlq_er', 'dlq_er_c2', 'centroid']
        data = [np.round(rad2muas(imp_ang), 0),
                np.round(np.rad2deg(dl2) * 3600 * 1e6, 2),
                np.round(np.rad2deg(dl1) * 3600 * 1e6, 2),
                np.round(np.rad2deg(dl_er) * 3600 * 1e6, 2),
                np.round(np.rad2deg(dl_er_c1-dl_er) * 3600 * 1e6, 6),
                np.round(np.rad2deg(dlq) * 3600 * 1e6, 6),
                np.round(np.rad2deg(dlq_er) * 3600 * 1e6, 6),
                np.round(np.rad2deg(dlq_er_c2-dlq_er) * 3600 * 1e6, 6),
                np.round(np.rad2deg(cs) * 3600 * 1e6, 2)]
        path = f'Data/comparison_d{np.round(np.linalg.norm(x) / pc, 1)}pc_latex'
        print(path)
        save_df(data, columns, rows, path)

