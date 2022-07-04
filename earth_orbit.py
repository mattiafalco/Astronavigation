""" This program evaluates the deflection on a target planet due to some bodies in
the solar system and a black hole over a given period and plot the results.

Creator: Mattia Falco
Date: 06/04/2022 
"""

from astronavigation.deflection import *
from astropy import constants
from astronavigation.planets import Body, SolarSystem
import matplotlib.pyplot as plt

# Define constants
pc = constants.pc.to('km').value
AU = constants.au.to('km').value
c = constants.c.to('km/s').value
GM_sun = constants.GM_sun.to('km3/s2').value
eps = 1/c

######################################
#
# Write here the data of the problem
#
######################################

# black hole
# bh = Body(mass=7.1*GM_sun,
#           pos=np.array([-4, -1580, -45])*pc,
#           vel=np.array([0, 0, 0]))#np.array([3, 0, 40]))
bh = Body(mass=7*GM_sun,
          pos=np.array([0, 1, 0])*1000*pc)

# observer
obs = 'earth'
# masses
list_p = ['sun', 'jupiter', 'saturn']
# target
dist = 3000*pc

# Time
t_span = np.arange(0, 2*365)  # day

#################
#
# Algorithm
#
#################

# Create Solar System
ss = SolarSystem()

# target
ll = bh.pos/bh.dist
the = 10*np.sqrt(4*bh.mass*(dist - bh.dist)/(dist * bh.dist))/c
l_tar = np.array([ll[0]*np.cos(the) - ll[1]*np.sin(the),
                  ll[0]*np.sin(the) + ll[1]*np.cos(the),
                  ll[2]])
x1 = l_tar*dist
# x2 = ll*dist
print(np.rad2deg(the) * 3600 * 1e6)

x = x1

# create dictionary to save data
dl_dict = {}
dl_dict['bh'] = []
for pl in list_p:
    dl_dict[pl] = []
dl_dict['tot'] = []

for t in t_span:

    # bh.pos = bh.pos + t*24*3600*bh.vel

    # observer
    anom = ss.getPlanet(obs).speed / ss.getPlanet(obs).dist * t * 24 * 3600
    x_obs = ss.getPlanet(obs, anom=anom).pos

    # line of sight
    l0 = -(x - x_obs) / (np.linalg.norm(x - x_obs))

    dl1 = []

    # bh deflection
    dls = deflection(l0, x, bh.pos, x_obs, eps, bh.mass, bh.vel)
    dl1.append(np.linalg.norm(dls))

    dl_dict['bh'].append(np.linalg.norm(dls))

    # planets deflection
    for planet in list_p:

        if planet != 'sun':
            anom_p = ss.getPlanet(planet).speed/ss.getPlanet(planet).dist * t * 24 * 3600 + np.pi/2+np.pi/200
        else:
            anom_p = 0

        pl = ss.getPlanet(planet, anom=anom_p)
        dls = deflection(l0, x, pl.pos, x_obs, eps, pl.mass, pl.vel)
        dl1.append(np.linalg.norm(dls))
        dl_dict[planet].append(np.linalg.norm(dls))

    dlt = np.sum(dl1)

    dl_dict['tot'].append(dlt)

    #################
    #
    # Printing
    #
    #################

    #print(f'day = {t}')
    #print(f'angle errors: {np.rad2deg(dl1) * 3600 * 1e6} muas\n')

#################
#
# Plot
#
#################

# deflection plots
fig, ax = plt.subplots()
dl_muas = np.rad2deg(np.array(dl_dict['bh'])) * 3600 * 1e6
ax.plot(t_span, dl_muas)
plt.title('BH deflection')
plt.xlabel('t [day]')
plt.ylabel('dl [muas]')

for pl in list_p:
    fig, ax = plt.subplots()
    dl_muas = np.rad2deg(np.array(dl_dict[pl])) * 3600 * 1e6
    ax.plot(t_span, dl_muas)
    plt.title(f'{pl} deflection')
    plt.xlabel('t [day]')
    plt.ylabel('dl [muas]')

# # distance plots
# fig, ax = plt.subplots()
# dr = np.array(dl_dict['bh']) * dist / AU
# ax.plot(t_span, dr)
# plt.title('BH err dist')
# plt.xlabel('t [day]')
# plt.ylabel('dr [AU]')
#
# for pl in list_p:
#     fig, ax = plt.subplots()
#     dr = np.array(dl_dict[pl]) * dist / AU
#     ax.plot(t_span, dr)
#     plt.title(f'{pl} err dist')
#     plt.xlabel('t [day]')
#     plt.ylabel('dr [AU]')


fig2 = plt.figure()
ax2 = plt.axes(projection='3d')
ax2.plot(0.0, 0.0, 0.0, marker='^', color='blue', label='sun')
ax2.plot(x[0]/pc, x[1]/pc, x[2]/pc, marker='o', color='red', label='target')
ax2.plot(bh.pos[0]/pc, bh.pos[1]/pc, bh.pos[2]/pc, marker='o', color='black', label='bh')
# ax2.plot(x2[0]/pc, x2[1]/pc, x2[2]/pc, marker='o', color='green')
plt.legend()

fig4 = plt.figure()
ax4 = plt.axes(projection='3d')
ax4.plot(0.0, 0.0, 0.0, marker='^', color='blue', label='sun')
for pl in ['earth', 'jupiter', 'saturn']:
    anom = 0 if pl == 'earth' else np.pi/2+np.pi/200
    x_p = ss.getPlanet(pl, anom=anom).pos
    ax4.plot(x_p[0], x_p[1], x_p[2], label=pl, marker='o')
ax4.plot(ll[0]*15*AU, ll[1]*15*AU, ll[2]*15*AU, label='bh', marker='+')
plt.legend()
plt.xlim((-1e9, 1e9))


plt.show()




