import numpy as np
from astropy import constants
from astropy.time import Time
from calcephpy import *

class Body(object):

    def __init__(self, mass, pos, vel=np.array([0, 0, 0]), radius=0):
        self.mass = mass
        self.pos = pos
        self.vel = vel
        self.radius = radius

    def __str__(self):
        str = f'\nmass*G: {self.mass} km3/s2 \npos: {self.pos} km' \
              f'\nvel: {self.vel} km/s \nradius: {self.radius} km'
        return str

class SolarSystem(object):

    def __init__(self, ephemerides=False):

        self.ephemerides = ephemerides

        p = [0, 57.909e6, 108.210e6, 149.598e6, 227.956e6, 778412010, 1426725400, 2870972200, 4498252900]
        v = [0, 47.36, 35.02, 29.78, 24.07, 13.0697, 9.6724, 6.8352, 5.4778]
        # m = [132712e6, 126.687e6, 37.931e6, 5.7940e6, 6.8351e6]
        m = [1.32712440017987e11, 2.203208e4, 3.24858599e5, 3.98600433e5, 4.2828314e4,
             1.26712767863e8, 3.79406260630e7, 5.79454900700e6, 6.83653406400e6]
        r = [6.955e5, 2439.7, 6051.8, 6371.01, 3389.9, 69911, 58232, 25362, 24624]

        bodies = []
        for i in range(len(p)):
            x_b = p[i]*np.array([0, 1, 0])
            v_b = v[i] * np.array([0, 1, 0])
            bodies.append(Body(m[i], x_b, v_b, r[i]))
        self.bodies = {'sun': bodies[0],
                       'mercury': bodies[1],
                       'venus': bodies[2],
                       'earth': bodies[3],
                       'mars': bodies[4],
                       'jupiter': bodies[5],
                       'saturn': bodies[6],
                       'uranus': bodies[7],
                       'neptune': bodies[8]}

    def getSun(self, date='2000-01-01T12:00:00'):

        if not self.ephemerides:
            return self.bodies['sun']
        else:
            peph = CalcephBin.open("inpop21a_TCB_m1000_p1000_tcg.dat")
            t = Time(date, format='isot', scale='utc')
            jd0 = t.jd
            dt = 0.0
            PV = peph.compute_unit(t.jd, dt, NaifId.SUN, NaifId.SUN,
                                   Constants.UNIT_KM+Constants.UNIT_SEC+Constants.USE_NAIFID)
            # print(PV)
            pos = PV[0:3]
            vel = PV[3:6]
            peph.close()

            x_s = np.array(pos)
            v_s = np.array(vel)
            m_s = self.bodies['sun'].mass
            r_s = self.bodies['sun'].radius
            return Body(m_s, x_s, v_s, r_s)

    def getPlanet(self, pl, date='2000-01-01T12:00:00', anom=np.pi/2):

        if not self.ephemerides:
            m_p = self.bodies[pl].mass
            x_p = np.linalg.norm(self.bodies[pl].pos) * np.array([np.cos(anom), np.sin(anom), 0])
            v_p = np.linalg.norm(self.bodies[pl].vel) * np.array([-np.sin(anom), np.cos(anom), 0])
            r_p = self.bodies[pl].radius
            planet = Body(m_p, x_p, v_p, r_p)

            return planet
        else:
            dic = {'sun': NaifId.SUN,
                   'mercury': NaifId.MERCURY_BARYCENTER,
                   'venus': NaifId.VENUS_BARYCENTER,
                   'earth': NaifId.EARTH,
                   'mars': NaifId.MARS_BARYCENTER,
                   'jupiter': NaifId.JUPITER_BARYCENTER,
                   'saturn': NaifId.SATURN_BARYCENTER,
                   'uranus': NaifId.URANUS_BARYCENTER,
                   'neptune': NaifId.NEPTUNE_BARYCENTER}

            peph = CalcephBin.open("inpop21a_TCB_m1000_p1000_tcg.dat")
            t = Time(date, format='isot', scale='utc')
            jd0 = t.jd
            dt = 0.0
            PV = peph.compute_unit(jd0, dt, dic[pl], NaifId.SUN,
                                   Constants.UNIT_KM + Constants.UNIT_SEC + Constants.USE_NAIFID)
            pos = PV[0:3]
            vel = PV[3:6]
            peph.close()

            x_p = np.array(pos)
            v_p = np.array(vel)
            m_p = self.bodies[pl].mass
            r_p = self.bodies[pl].radius
            return Body(m_p, x_p, v_p, r_p)

if __name__ == '__main__':

    print('\n------------- Tests with no ephemerides ------------\n')
    ss = SolarSystem()
    print(ss.getSun())
    print(ss.getPlanet('jupiter'))

    ss = SolarSystem(ephemerides=True)

    print('\n------------- Tests with ephemerides -----------------\n')
    print(f'Sun {ss.getSun()}\n')
    print(f'sun second function {ss.getPlanet("sun")}\n')
    print(f'jupiter {ss.getPlanet("jupiter")}\n')
    print(f'earth {ss.getPlanet("earth")}\n')
    print(f'earth orbital radius: {np.linalg.norm(ss.getPlanet("earth").pos)}')

