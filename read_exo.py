""" This file defines functions to read the exoplanet catalogue. These functions can be used
to get a particular exoplanet, a system or even the whole catalogue.
Creator: Mattia
Date: 31/03/22.
"""

import pandas as pd
import numpy as np
from planets import Body
from deflection import cartesian
from astropy import constants


def getExo(pl_name, df):
    """
    Returns the Body which corresponds to the required identification label. Does not
    give the values of s or J2. If the value of mass or radius is NaN then it is set to zero.

    Parameters
    ----------
    pl_name : str
        name of the planet
    df :  pd.DataFrame
        catalogue in which search for the exoplanet

    Returns
    -------
    planet : Body
        required exoplanet
    """

    df_copy = df.set_index('pl_name')

    # parsec
    pc = constants.pc.to('km').value

    # Jupiter data
    mj = 1.26712767863e8  # [km3/s2]
    rj = 69911  # [km]

    planet = df_copy.loc[pl_name]

    # mass
    m_p = planet.loc['pl_massj'] * mj
    if pd.isna(m_p):
        m_p = 0

    # radius
    r_p = planet.loc['pl_radj'] * rj
    if pd.isna(r_p):
        r_p = 0

    # position
    ra = planet.loc['ra']
    dec = planet.loc['dec']
    x_p = planet.loc['sy_dist'] * cartesian(ra, dec) * pc

    # velocity
    v_p = np.array([0, 0, 0])

    # rotation
    s_p = np.array([0, 0, 0])

    # J2
    J2_p = 0

    return Body(m_p, x_p, v_p, r_p, s_p, J2_p)


def getExoSystem(hostname, df):
    """
    Returns dictionary of bodies of a given hostname.

    Parameters
    ----------
    hostname : str
        name of the star
    df : pd.DataFrame
        catalogue in which search for the exoplanet

    Returns
    -------
    system : dict
        dictionary of bodies in the required system. The first one is the star.
    """

    # select the system
    filt = df['hostname'] == hostname
    df_system = df.loc[filt]

    # name of planets
    pl_list = df_system['pl_name']

    # create dict
    system = {}
    for pl in pl_list:
        system[pl] = getExo(pl, df_system)

    # Sun data
    m_sun = 1.32712440017987e11
    r_sun = 6.955e5

    # add the star using mean values from all the planets in the system
    m_s = df_system['st_mass'].mean() * m_sun
    r_s = df_system['st_rad'].mean() * r_sun
    x_s = sum([system[pl].pos for pl in system]) / len(system)

    system[hostname] = Body(m_s, x_s, radius=r_s)

    return system


def getExoAll(df):
    """
    Returns a dictionary with all exoplanets in the catalogue.

    Parameters
    ----------
    df : pd.DataFrame
        catalogue in which search for the exoplanet

    Returns
    -------
    cat : dict
        dictionary of bodies in the required system. The first one is the star.
    """

    # name of planets
    pl_names = df['pl_name']

    # create dict
    cat = {}
    for pl in pl_names:
        cat[pl] = getExo(pl, df)

    return cat


if __name__ == "__main__":

    path = 'exo_archive.csv'
    data = pd.read_csv(path)

    print('---------------- test1 --------------')

    print(data.columns)
    print(data)
    target = 'Kepler-80 b'
    exop = getExo(target, data)

    print(exop)
    print(exop.dist/constants.pc.to('km').value)

    print('---------------- test2 --------------')

    target = 'Kepler-80'
    ss = getExoSystem(target, data)
    for name in ss:
        print(f'{name}: {ss[name]}\n')

    print(ss[target].mass/1.32712440017987e11)
    print(ss.keys())

    print('---------------- test3 --------------')

    all_p = getExoAll(data)

    # print(len(all_p))
