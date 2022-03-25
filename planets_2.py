import numpy as np
from astropy import constants
from astropy.time import Time
from calcephpy import *
from deflection import quadru

class Body(object):

    def __init__(self, mass, pos, vel=np.array([0, 0, 0]), radius=0, s=np.array([0, 0, 0]), J2=0):
        self.mass = mass
        self.pos = pos
        self.vel = vel
        self.radius = radius
        self.s = s
        self.J2 = J2

    def __str__(self):
        str = f'\nmass*G: {self.mass} km3/s2 \npos: {self.pos} km' \
              f'\nvel: {self.vel} km/s \nradius: {self.radius} km' \
              f'\ns: {self.s} \nJ2: {self.J2}'

        return str

class SolarSystem(object):

    def __init__(self, ephemerides=False):

        self.ephemerides = ephemerides

        self.b_names = ['sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter',
                        'saturn', 'uranus', 'neptune']

        p = {'sun': 0, 'mercury': 57.909e6, 'venus': 108.210e6, 'earth': 149.598e6, 'mars': 227.956e6,
             'jupiter': 778412010, 'saturn': 1426725400, 'uranus': 2870972200, 'neptune': 4498252900}
        v = {'sun': 0, 'mercury': 47.36, 'venus': 35.02, 'earth': 29.78, 'mars': 24.07, 'jupiter': 13.0697,
             'saturn': 9.6724, 'uranus': 6.8352, 'neptune': 5.4778}
        # m = [132712e6, 126.687e6, 37.931e6, 5.7940e6, 6.8351e6]
        m = {'sun': 1.32712440017987e11, 'mercury': 2.203208e4, 'venus': 3.24858599e5, 'earth': 3.98600433e5,
             'mars': 4.2828314e4, 'jupiter': 1.26712767863e8, 'saturn': 3.79406260630e7, 'uranus': 5.79454900700e6,
             'neptune': 6.83653406400e6}
        r = {'sun': 6.955e5, 'mercury': 2439.7, 'venus': 6051.8, 'earth': 6371.01, 'mars': 3389.9, 'jupiter': 69911,
             'saturn': 58232, 'uranus': 25362, 'neptune': 24624}
        s = {}
        J2 = {}
        for pl in self.b_names:
            s[pl], J2[pl] = quadru(pl)

        bodies = {}
        for pl in self.b_names:
            x_b = p[pl]*np.array([0, 1, 0])
            v_b = v[pl] * np.array([0, 1, 0])
            bodies[pl] = Body(m[pl], x_b, v_b, r[pl], s[pl], J2[pl])
        self.bodies = bodies

    def getSun(self, date='2000-01-01T12:00:00'):

        if not self.ephemerides:
            return self.bodies['sun']
        else:
            peph = CalcephBin.open("inpop21a_TCB_m1000_p1000_tcg.dat")
            t = Time(date, format='isot', scale='utc')
            jd0 = t.jd
            dt = 0.0
            PV = peph.compute_unit(jd0, dt, NaifId.SUN, NaifId.SUN,
                                   Constants.UNIT_KM+Constants.UNIT_SEC+Constants.USE_NAIFID)
            # print(PV)
            pos = PV[0:3]
            vel = PV[3:6]
            peph.close()

            x_s = np.array(pos)
            v_s = np.array(vel)
            m_s = self.bodies['sun'].mass
            r_s = self.bodies['sun'].radius
            s_s = self.bodies['sun'].s
            J2_s = self.bodies['sun'].J2
            return Body(m_s, x_s, v_s, r_s, s_s, J2_s)

    def getPlanet(self, pl, date='2000-01-01T12:00:00', anom=np.pi/2):

        if not self.ephemerides:
            m_p = self.bodies[pl].mass
            x_p = np.linalg.norm(self.bodies[pl].pos) * np.array([np.cos(anom), np.sin(anom), 0])
            v_p = np.linalg.norm(self.bodies[pl].vel) * np.array([-np.sin(anom), np.cos(anom), 0])
            r_p = self.bodies[pl].radius
            s_p = self.bodies[pl].s
            J2_p = self.bodies[pl].J2
            planet = Body(m_p, x_p, v_p, r_p, s_p, J2_p)

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
            s_p = self.bodies[pl].s
            J2_p = self.bodies[pl].J2
            return Body(m_p, x_p, v_p, r_p, s_p, J2_p)

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

