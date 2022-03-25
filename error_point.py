import numpy as np
from deflection import *
from astropy import constants
from planets_2 import Body, SolarSystem

pc = constants.pc.to('km').value
AU = constants.au.to('km').value
c = constants.c.to('km/s').value
eps = 1/c

# angle of observation
g = -np.pi/2
x_obs = AU*np.array([np.cos(g), np.sin(g), 0])

list_p = ['sun', 'jupiter', 'saturn', 'uranus', 'neptune']
ss = SolarSystem()
planets = [ss.getPlanet(pl) for pl in list_p]

# planets data
x_p = [planet.pos for planet in planets]
v_p = [planet.vel for planet in planets]
m_p = [planet.mass for planet in planets]
r_p = [planet.radius for planet in planets]
chi_p = [r_p[i]/np.linalg.norm(x_obs - x_p[i]) for i in range(len(planets))]
print(f'chi: {chi_p}')

# stars
dist = np.array([1.3012, 188, 2300])*pc
x_stars = [np.array([0, d, 0]) for d in dist]
l0 = [-np.array([np.sin(chi), np.cos(chi), 0]) for chi in chi_p]

dl1 = np.zeros(len(x_p))
for x in x_stars:
    for i in range(len(x_p)):
        dls = deflection_mod(l0[i], x, x_p[i], x_obs, eps, v_p[i], m_p[i], chi_p[i])
        dl1[i] = np.linalg.norm(dls)

    dlt = np.cumsum(dl1)

    # point error
    dr = np.linalg.norm(x-x_obs)*dlt
    print(f'\n---------------------------------\nexoplanet at {np.linalg.norm(x)/pc} pc')
    print(f'pointing errors: {dr/AU} AU')
    print(f'pointing errors - sun: {(dr-dr[0])/AU} AU')

    # quadrupole jupiter-saturn
    list_q = ['jupiter', 'saturn']
    planets_q = [ss.getPlanet(pl) for pl in list_q]

    dlq = []
    for i in range(len(planets_q)):
        chi = planets_q[i].radius/np.linalg.norm(x_obs-planets_q[i].pos)
        l0q = -np.array([np.sin(chi), np.cos(chi), 0])
        dls = deflection_mod(l0q, x, planets_q[i].pos, x_obs, eps, planets_q[i].vel, planets_q[i].mass, chi,
                             planets_q[i].s, planets_q[i].J2, planets_q[i].radius)
        dlq.append(np.linalg.norm(dls))
    dlq = np.cumsum(np.array(dlq))
    drq = np.linalg.norm(x-x_obs)*dlq
    print(f'quadrupole jup-sat: {drq/AU} AU')

