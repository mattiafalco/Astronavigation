import numpy as np
from astropy import constants

from astropy.time import Time

#t = Time('2000-01-01T12:00:00', format='isot', scale='utc')

#print(f'time_jd: {t.jd}')
#print(f'time_mjd: {t.mjd}')

#from calcephpy import *
#peph = CalcephBin.open("inpop21a_TCB_m1000_p1000_tcg.dat")
#jd0 = 2451542
#dt = 0.0
#PV = peph.compute_unit(t.jd, dt, NaifId.MOON, NaifId.SUN,
#                       Constants.UNIT_KM+Constants.UNIT_SEC+Constants.USE_NAIFID)
#print(PV)
#peph.close()

class Body(object):

    def __init__(self, mass, pos, vel=np.array([0, 0, 0]), radius=0):
        self.mass = mass
        self.pos = pos
        self.vel = vel
        self.radius = radius


class SolarSystem(object):

    def __init__(self):
        self.p = [0, 778412010, 1426725400, 2870972200, 4498252900]
        self.v = [0, 13.0697, 9.6724, 6.8352, 5.4778]
        # self.m = [132712e6, 126.687e6, 37.931e6, 5.7940e6, 6.8351e6]
        self.m = [1.32712440017987e11, 1.26712767863e8, 3.79406260630e7, 5.79454900700e6, 6.83653406400e6]

    def getSun(self):
        x_s = np.array([0, 0, 0])
        v_s = np.array([0, 0, 0])
        r_s = constants.R_sun.to('km').value
        sun = Body(self.m[0], x_s, v_s, r_s)

        return sun

    def getPlanet(self, i, anom=np.pi/2):
        x_p = self.p[i+1]*np.array([np.cos(anom), np.sin(anom), 0])
        v_p = self.v[i + 1] * np.array([-np.sin(anom), np.cos(anom), 0])
        r_p = 0
        planet = Body(self.m[i+1], x_p, v_p, r_p)

        return planet
