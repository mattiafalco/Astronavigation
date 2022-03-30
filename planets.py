""" This file defines the classes Body and SolarSystem which organize the various information
of the parameters of the planets and the Sun. It also implements the ephemerides from
the INPOP catalogue.
Creator: Mattia Falco
Date: 23/03/2022
"""

import numpy as np
from astropy import constants
from astropy.time import Time
from calcephpy import *
from deflection import quadru


def read_from_INPOP(pl, date):
    """
    Read the information from the INPOP catalogue and returns position and velocity of the
    required body.

    Parameters
    ----------
    pl : str
        body name
    date : str
        iso date

    Returns
    -------
    pos : np.ndarray
        position
    vel : np.nd.array
        velocity
    """

    dic = {'sun': NaifId.SUN,
           'mercury': NaifId.MERCURY_BARYCENTER,
           'venus': NaifId.VENUS_BARYCENTER,
           'earth': NaifId.EARTH,
           'mars': NaifId.MARS_BARYCENTER,
           'jupiter': NaifId.JUPITER_BARYCENTER,
           'saturn': NaifId.SATURN_BARYCENTER,
           'uranus': NaifId.URANUS_BARYCENTER,
           'neptune': NaifId.NEPTUNE_BARYCENTER,
           'moon': NaifId.MOON}

    if pl not in dic:
        raise ValueError(f'{pl} is not an implemented body.')

    peph = CalcephBin.open("inpop21a_TCB_m1000_p1000_tcg.dat")
    t = Time(date, format='isot', scale='utc')
    jd0 = t.jd  # -2400000.5 - 51544.5
    dt = 0.0
    PV = peph.compute_unit(jd0, dt, dic[pl], NaifId.SUN,
                           Constants.UNIT_KM + Constants.UNIT_SEC + Constants.USE_NAIFID)
    peph.close()

    return np.array(PV[0:3]), np.array(PV[3:6])


class Body(object):
    """
    A class used to represent a celestial body.

    Attributes
    ----------
    mass : float
        body mass*G (in km3/s2)
    pos : 3-array
        body position (in km)
    vel : 3-array
        body velocity (in km/s)
    radius : float
        body radius (in km)
    s : 3-array
        rotation vector
    J2 : float
        body J2 parameter
    """

    def __init__(self, mass, pos, vel=np.array([0, 0, 0]), radius=0, s=np.array([0, 0, 0]), J2=0):
        """
        Parameters
        ----------
        mass : float
            body mass*G (in km3/s2)
        pos : np.ndarray
            body position (in km)
        vel : np.ndarray
            body velocity (in km/s)
        radius : float
            body radius (in km)
        s : np.ndarray
            rotation vector
        J2 : float
            body J2 parameter
        """
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
    """
    A class used to encode the Solar System information and to evaluate the dynamical elements of its bodies
    either in a simplified version of circular and coplanar orbits either using the ephemerides taken from
    the INPOP catalogue.

    Attributes
    ----------
    ephemerides : bool
        use or not use the ephemerides catalogue (default False)
    b_names : list
        names of all the Solar System bodies
    bodies : dict
        dictionary of all the object type Body
    """

    def __init__(self, ephemerides=False):
        """
        Create the Solar System with all its bodies.

        Parameters
        ----------
        ephemerides : bool
            use or not use the ephemerides catalogue (default False)
        """

        self.ephemerides = ephemerides

        self.b_names = ['sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter',
                        'saturn', 'uranus', 'neptune', 'moon']

        # position, velocity, mass*G, radius
        p = {'sun': 0, 'mercury': 57.909e6, 'venus': 108.210e6, 'earth': 149.598e6, 'mars': 227.956e6,
             'jupiter': 778412010, 'saturn': 1426725400, 'uranus': 2870972200, 'neptune': 4498252900,
             'moon': 149.598e6+0.3633e6}  # km
        v = {'sun': 0, 'mercury': 47.36, 'venus': 35.02, 'earth': 29.78, 'mars': 24.07, 'jupiter': 13.0697,
             'saturn': 9.6724, 'uranus': 6.8352, 'neptune': 5.4778, 'moon': 29.78}  # km/s
        m = {'sun': 1.32712440017987e11, 'mercury': 2.203208e4, 'venus': 3.24858599e5, 'earth': 3.98600433e5,
             'mars': 4.2828314e4, 'jupiter': 1.26712767863e8, 'saturn': 3.79406260630e7, 'uranus': 5.79454900700e6,
             'neptune': 6.83653406400e6, 'moon': 0.0049e6}  # km3/s2
        r = {'sun': 6.955e5, 'mercury': 2439.7, 'venus': 6051.8, 'earth': 6371.01, 'mars': 3389.9, 'jupiter': 69911,
             'saturn': 58232, 'uranus': 25362, 'neptune': 24624, 'moon': 1737}  # km

        # rotation vector, J2 parameter
        s = {}
        J2 = {}
        for pl in self.b_names:
            s[pl], J2[pl] = quadru(pl)

        # create bodies dictionary with all planets on the y-axis
        bodies = {}
        for pl in self.b_names:
            x_b = p[pl]*np.array([0, 1, 0])
            v_b = v[pl] * np.array([-1, 0, 0])
            bodies[pl] = Body(m[pl], x_b, v_b, r[pl], s[pl], J2[pl])
        self.bodies = bodies

    def getPlanet(self, pl, date='2000-01-01T12:00:00', anom=np.pi/2):
        """
        Return the required body in a specified position given by the date or the anomaly parameter

        Parameters
        ----------
        pl : str
            body name
        date : str
            iso date
        anom : float
            angle

        Returns
        -------
        planet : Body
            required body object in the given position
        """

        if not self.ephemerides:
            x_p = np.linalg.norm(self.bodies[pl].pos) * np.array([np.cos(anom), np.sin(anom), 0])
            v_p = np.linalg.norm(self.bodies[pl].vel) * np.array([-np.sin(anom), np.cos(anom), 0])
            m_p = self.bodies[pl].mass
            r_p = self.bodies[pl].radius
            s_p = self.bodies[pl].s
            J2_p = self.bodies[pl].J2

            return Body(m_p, x_p, v_p, r_p, s_p, J2_p)
        else:
            x_p, v_p = read_from_INPOP(pl, date)
            m_p = self.bodies[pl].mass
            r_p = self.bodies[pl].radius
            s_p = self.bodies[pl].s
            J2_p = self.bodies[pl].J2

            return Body(m_p, x_p, v_p, r_p, s_p, J2_p)

    def getSun(self, date='2000-01-01T12:00:00'):
        """
        Return the Sun in a specified position given by the date
        Parameters
        ----------
        date : str
            iso date

        Returns
        -------
        Sun : Body
            required body object in the given position
        """

        return self.getPlanet('sun', date=date)

if __name__ == '__main__':

    print('\n------------- Tests with no ephemerides ------------\n')
    ss = SolarSystem()
    print(f'Sun {ss.getSun()}\n')
    print(f'jupiter {ss.getPlanet("jupiter")}\n')

    ss = SolarSystem(ephemerides=True)

    print('\n------------- Tests with ephemerides -----------------\n')
    print(f'Sun {ss.getSun()}\n')
    print(f'sun second function {ss.getPlanet("sun")}\n')
    print(f'jupiter {ss.getPlanet("jupiter")}\n')
    print(f'earth {ss.getPlanet("earth")}\n')
    print(f'earth orbital radius: {np.linalg.norm(ss.getPlanet("earth").pos)}\n')
    # read_from_INPOP('PCB', '2000-01-01T12:00:00')

    day = '2050-01-01T12:00:00'
    print(f'moon, day = {day} {ss.getPlanet("moon", day)}')
