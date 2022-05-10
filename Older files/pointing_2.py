from astronavigation.deflection import *
from planets import SolarSystem

# pc = constants.pc.to('km').value
#AU = constants.au.to('km').value
# c = constants.c.to('km/s').value
AU = 149597870.691  # [km]
pc = 3.0856775814672e13  # [km]
c = 299792.458  # [km/s]
eps = 1/c

# angle of observation
g = -np.pi/2 + 6.955e5/AU  # constants.R_sun.to('km').value/AU
x_obs = AU*np.array([np.cos(g), np.sin(g), 0])

ss = SolarSystem()
sun = ss.getSun()
planets = [ss.getPlanet(i) for i in range(0, 4)]

# Sun data
x_s = sun.pos
v_s = sun.vel
MS = sun.mass

# planets data
x_p = [planets[i].pos for i in range(len(planets))]
v_p = [planets[i].vel for i in range(len(planets))]
m_p = [planets[i].mass for i in range(len(planets))]

sources_M = np.insert(m_p, 0, MS, axis=0)
sources_x = np.insert(x_p, 0, x_s, axis=0)
sources_v = np.insert(v_p, 0, v_s, axis=0)
#print(sources_x)

# star
dist = 1.3012*pc
x = np.array([0, dist, 0])
l0 = -(x - x_obs) / (np.linalg.norm(x - x_obs))

dl1 = np.zeros(len(sources_x))
dpsi = np.zeros(len(sources_x))
dr = np.zeros(len(sources_x))
for i in range(len(sources_x)):
    dls = deflection(l0, x, sources_x[i], x_obs, eps, sources_v[i], sources_M[i])
    dl1[i] = np.linalg.norm(dls)
    # dln = dls - l0*np.dot(dls, l0)
    # dpsi[i] = np.rad2deg(np.linalg.norm(dln))*3600*1e6

dlt = np.cumsum(dl1)
# dpsi_tot = [sum(dpsi[0:i+1]) for i in range(0, len(dpsi))]
dll = dlt - dlt[0]*np.ones_like(dlt)
# dpsi_rel = dpsi_tot - dpsi_tot[0]
print(f'dll = {np.rad2deg(dll)*3600*1e6} muas')
# print(dpsi_rel)

# point error
dr = np.array([np.linalg.norm(x-x_obs)*dlt[i] for i in range(0, len(dlt))])
print(f'pointing errors: {dr/AU} AU')
# print(np.rad2deg(dl1)*3600*1e6)
print(dlt)

print(f'dl1: {np.rad2deg(dl1)*3600*1e6}')
print(f'dlt: {np.rad2deg(dlt)*3600*1e6}')
print(f'Neptune data : x={planets[0].pos}, v={planets[0].vel}, mass={planets[0].mass}')
print(f'l0: {l0}')
print(f'x: {x}')
print(f'x_obs: {x_obs}')

# quadrupole
# Jupiter
sJ2 = quadru(1)
# R = 71492  # [km]
R = 69911
dls = deflection(l0, x, planets[0].pos, x_obs, eps, planets[0].vel, planets[0].mass, sJ2[:3], sJ2[3], R)
dlq = np.linalg.norm(dls)
drq = np.linalg.norm(x-x_obs)*dlq
print(f'quadrupole_jup: {(drq)/AU} AU')

# Saturn
sJ2 = quadru(2)
# R = 60268  # [km]
R = 58232
dls = deflection(l0, x, planets[1].pos, x_obs, eps, planets[1].vel, planets[1].mass, sJ2[:3], sJ2[3], R)
dlq = np.linalg.norm(dls)
drq += np.linalg.norm(x-x_obs)*dlq
print(f'quadrupole_sat: {(drq)/AU} AU')
