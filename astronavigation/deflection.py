""" This file contains useful functions which make the code easier to read
and also the two main functions used to evaluate the deflection of light rays
in the IAU metric as well as the distances. Other types of deflection formulas are
implemented such as Erez-Rosen or Ellis wormhole.

Creator: Mattia Falco
Date: 17/03/2022
"""

import numpy as np


#########################
#
# Useful functions
#
#########################


def cartesian(ra, dec):
    """This function evaluates the cartesian coordinates from the values of right ascension and declination
    expressed in degree.

    Parameters
    ----------
    ra : float
        right ascension (in degree)
    dec : float
        declination (in degree)

    Returns
    -------
    np.ndarray
        cartesian coordinates
    """

    alpha = np.deg2rad(ra)
    delta = np.deg2rad(dec)

    return np.array([np.cos(alpha) * np.cos(delta), np.sin(alpha) * np.cos(delta), np.sin(delta)])


def quadru(pl):
    """
    Evaluate the rotation vector and the J2 parameter of a given body of the Solar System.
    Data are taken from the NASA fact sheets (https://nssdc.gsfc.nasa.gov/planetary)

    Parameters
    ----------
    pl : str
        body name

    Returns
    -------
    s : np.ndarray
        rotation vector
    J2 : float
        J2 parameter
    """

    alpha = 0
    delta = 0
    J2 = 0

    if pl == 'jupiter':
        alpha = np.deg2rad(268.057)
        delta = np.deg2rad(64.495)
        J2 = 14736e-6
    elif pl == 'saturn':
        alpha = np.deg2rad(40.589)
        delta = np.deg2rad(83.537)
        J2 = 16298e-6
    elif pl == 'earth':
        alpha = np.deg2rad(0.0)
        delta = np.deg2rad(90.0)
        J2 = 1082.63e-6
    elif pl == 'uranus':
        alpha = np.deg2rad(257.311)
        delta = np.deg2rad(-15.175)
        J2 = 3343.43e-6
    elif pl == 'neptune':
        alpha = np.deg2rad(299.36 + 0.70 * np.sin(np.deg2rad(357.85)))
        delta = np.deg2rad(43.46 - 0.51 * np.cos(np.deg2rad(357.85)))
        J2 = 3411e-6
    else:
        pass

    s = np.array([np.cos(alpha) * np.cos(delta), np.sin(alpha) * np.cos(delta), np.sin(delta)])

    return s, J2


def parallax_shift(dl, l0, r):
    """ Evaluate the parallax shift of a source due to the gravitational light deflection.

    Parameters
    ----------
    dl : float
        norm of the deflection
    l0 : np.ndarray
        unperturbed direction
    r : np.ndarray
        observer position (in km)

    Returns
    -------
    dw : float
        parallactic shift

    """

    p = np.cross(l0, np.cross(l0, r))
    dw = dl / np.linalg.norm(p)

    return dw


def einstein_ring(mg, eps, x_a, x, x_obs):
    """ Evaluate Einstein ring of a black hole.

    Parameters
    ----------
    mg : float
        mass parameter m*G (in km3/s2)
    eps : float
        1/c
    x_a : np.ndarray
        mass position (in km)
    x : np.ndarray
        source position (in km)
    x_obs : np.ndarray
        obsrver position (in km)

    Returns
    -------
    theta : float
        einsten ring (in rad)
    """

    theta = np.sqrt(4 * mg * (np.linalg.norm(x - x_a))
                    / (np.linalg.norm(x-x_obs) * np.linalg.norm(x_a-x_obs))) * eps
    return theta


def rad2muas(rad):
    """
    convert radiant to muas

    Parameters
    ----------
    rad: float
        angle in radiant

    Returns
    -------
    muas: float
        angle in muas
    """

    return np.rad2deg(rad)*3600e6


def on_triplet(l0, x_a):
    """ Evaluates three ON vectors given the direction of a source and the position of a lens.

    Parameters
    ----------
    l0 : np.ndarray
        source direction
    x_a : np.ndarray
        lens position

    Returns
    -------
    t, n, m : list
        list of np.ndarray
    """

    d = x_a - l0 * np.dot(x_a, l0)

    n = d / np.linalg.norm(d)
    t = -l0
    m = np.cross(n, t)

    return t, n, m


#########################
#
# Light deflection formulas
#
#########################

def CM_formula(l0, b, chi, M, eps, J2=0, R=0, s=np.array([0,0,1])):
    """ Crosta-Mignard formula for light deflection

    Parameters
    ----------
    l0 : np.ndarray
        unperturbed direction
    b : np.ndarray
        impact parameter (in km)
    chi : float
        impact angle
    M :  float
        mass parameter m*G (in km3/s2)
    eps : float
        small parameter, usually 1/c (in s/km)
    J2 : float
        oblateness parameter (default is 0)
    R : float
        mass radius (in km, default is 0)
    s : np.ndarray
        rotation vector (default is [0,0,1])

    Returns
    -------
    float :
        monopole deflection
    list : two contribution to quadrupole deflection
    """

    if J2 == 0 and R == 0:

        # evaluate deflection
        mono = 1 + np.cos(chi)
        dl = (2 * M * eps ** 2 / np.linalg.norm(b)) * mono

        return dl
    else:
        # define three ON vectors
        n = b / np.linalg.norm(b)
        t = -l0
        m = np.cross(n, t)

        # evaluate useful combinations
        p1 = 1 + np.cos(chi) + 0.5 * np.cos(chi) * np.sin(chi) ** 2
        p2 = -2 * (1 + np.cos(chi) + 0.5 * np.cos(chi) * np.sin(chi) ** 2
                   + 3 / 4 * np.cos(chi) * np.sin(chi) ** 4) * np.dot(n, s) ** 2
        p3 = (np.sin(chi) ** 3 - 3 * np.sin(chi) ** 5) * np.dot(n, s) * np.dot(t, s)
        p4 = -(1 + np.cos(chi) + 0.5 * np.cos(chi) * np.sin(chi) ** 2
               - 3 / 2 * np.cos(chi) * np.sin(chi) ** 4) * np.dot(t, s) ** 2
        p5 = 2 * (1 + np.cos(chi) + 0.5 * np.cos(chi) * np.sin(chi) ** 2) * np.dot(n, s) * np.dot(m, s)
        p6 = np.sin(chi) ** 3 * np.dot(m, s) * np.dot(t, s)

        # evaluate deflection
        mono = 0  # 1+np.cos(chi)
        # dl = (2 * M * eps ** 2 / np.sqrt(d2)) * (J2 * R ** 2 / d2) * ((p1 + p2 + p3 + p4) * n + (p5 + p6) * m)
        dphi_1 = (2 * M * eps ** 2 / np.linalg.norm(b)) * (mono + (J2 * R ** 2 / np.linalg.norm(b)**2) * (p1 + p2 + p3 + p4))
        dphi_2 = (2 * M * eps ** 2 / np.linalg.norm(b)) * (J2 * R ** 2 / np.linalg.norm(b)**2) * (p5 + p6)

        return dphi_1, dphi_2

def RAMOD_formula(l0, x, x_a, x_obs, eps, v, M,
               s=np.array([0, 0, 0]), J2=0, R=0):
    """RAMOD deflection.

        Parameters
        ----------
        l0 : np.ndarray
            unperturbed direction
        x : np.ndarray
            source position (in km)
        x_a : np.ndarray
            mass position (in km)
        x_obs : np.ndarray
            observer position (in km)
        eps : float
            small parameter, usually 1/c (in s/km)
        v : np.ndarray
            mass velocity (in km/s)
        M : float
            mass parameter m*G (in km3/s2)
        s : np.ndarray
            rotation vector (default is [0,0,0])
        J2 : float
            oblateness parameter (default is 0)
        R : float
            mass radius (in km, default is 0)

        Returns
        -------
        np.ndarray
            perturbation on the direction of observation
        """
    # debug parameter, if true print some information
    debug = False

    # evaluate distance mass-source
    r = x - x_a
    # evaluate distance mass-observer
    r_obs = x_obs - x_a

    # evaluate vector norm
    r_norm = np.linalg.norm(r)
    r_obs_norm = np.linalg.norm(r_obs)

    # evaluate normal vectors
    n = r / r_norm
    n_obs = r_obs / r_obs_norm

    # evaluate parameters
    sigma = np.dot(x - x_obs, l0)
    d = r_obs - l0 * np.dot(r_obs, l0)
    d2 = np.linalg.norm(d) ** 2
    dv = np.cross(l0, np.cross(v, l0))

    if J2 == 0 and R == 0:

        # evaluate useful combinations
        p1 = 1 / r_norm - 1 / r_obs_norm
        p2 = l0 - 2 * eps * (2 * dv - d * np.dot(v, r_obs) / d2)
        p3 = 2 * d * ((1 - 2 * eps * np.dot(v, l0)) * (np.dot(n, l0) - np.dot(n_obs, l0))) / d2
        p4 = r_obs_norm * (dv - 2 * d * np.dot(v, d) / d2) / (d2 * r_norm)
        p5 = r_norm - r_obs_norm - np.dot(n_obs, l0) * sigma

        if debug: print(f'p1: {p1}\np2: {p2}\np3: {p3}\np4: {p4}\np5: {p5}\np4*p5{p4 * p5}')

        # evaluate deflection
        dl = -M * eps ** 2 * (p1 * p2 + p3) + 2 * M * (eps ** 3) * p4 * p5

        if debug: print(f'dl: {dl}')

        return dl
    else:
        # define three ON vectors
        n = -d / np.linalg.norm(d)
        t = l0
        m = np.cross(t, n)

        # evaluate chi
        chi = np.arccos(np.dot(x - x_obs, x_a - x_obs) / (np.linalg.norm(x - x_obs) * np.linalg.norm(x_a - x_obs)))

        # evaluate useful combinations
        p1 = 1 + np.cos(chi) + 0.5 * np.cos(chi) * np.sin(chi) ** 2
        p2 = -2 * (1 + np.cos(chi) + 0.5 * np.cos(chi) * np.sin(chi) ** 2
                   + 3 / 4 * np.cos(chi) * np.sin(chi) ** 4) * np.dot(n, s) ** 2
        p3 = (np.sin(chi) ** 3 - 3 * np.sin(chi) ** 5) * np.dot(n, s) * np.dot(t, s)
        p4 = -(1 + np.cos(chi) + 0.5 * np.cos(chi) * np.sin(chi) ** 2
               - 3 / 2 * np.cos(chi) * np.sin(chi) ** 4) * np.dot(t, s) ** 2
        p5 = 2 * (1 + np.cos(chi) + 0.5 * np.cos(chi) * np.sin(chi) ** 2) * np.dot(n, s) * np.dot(m, s)
        p6 = np.sin(chi) ** 3 * np.dot(m, s) * np.dot(t, s)

        # evaluate deflection
        dl = (2 * M * eps ** 2 / np.sqrt(d2)) * (J2 * R ** 2 / d2) * ((p1 + p2 + p3 + p4) * n + (p5 + p6) * m)

        if debug: print(f'd = {np.sqrt(d2)}')

        return dl

def Erez_Rosen_formula(b, eps, M, J2, R, c1=True, c2=True, quad=True):
    """Erez-Rosen formula.

    Parameters
    ----------
    b : float
        impact parameter (in km)
    eps : float
        small parameter, usually 1/c (in s/km)
    M : float
        mass parameter m*G (in km3/s2)
    J2 : float
        oblateness parameter
    R : float
        mass radius (in km)
    c1 : bool
        evaluate or not the monopole correction
    c2 : bool
        evaluate or not the quadrupole correction
    quad : bool
        evaluate or not the quadrupole contribution

    Returns
    -------
    dl : np.ndarray
        perturbation angle (in rad)
    """

    # evaluate contributions
    p1 = 4 * M / b
    p2 = (15 * np.pi * M ** 2) / (4 * b**2) if c1 else 0
    if quad:
        p3 = (4 * J2 * R ** 2 * M) / (b ** 3)
        p4 = (128 * M ** 3) / (3 * b ** 3) if c2 else 0
    else:
        p3 = 0
        p4 = 0

    # evaluate deflection
    dl = eps ** 2 * p1 + eps ** 4 * p2 + eps ** 2 * p3 + eps ** 6 * p4

    return dl


def Ellis_formula(b, a):

    # evaluate contributions
    p1 = (np.pi / 4) * a ** 2 / b**2

    # evaluate deflection
    dl = p1

    return dl


#########################
#
# general functions
#
#########################

def deflection(l0, x, x_a, x_obs, eps, M, v=np.array([0,0,0]),
               s=np.array([0, 0, 1]), J2=0, R=0):
    """
    Evaluate  the difference between the unperturbed direction l0 and the perturbed direction. If one of the last three
    parameters is set to its default value it evaluates the monopole contribution whereas it evaluates only the
    quadrupole contribution.

    Parameters
    ----------
    l0 : np.ndarray
        unperturbed direction
    x : np.ndarray
        source position (in km)
    x_a : np.ndarray
        mass position (in km)
    x_obs : np.ndarray
        observer position (in km)
    eps : float
        small parameter, usually 1/c (in s/km)
    v : np.ndarray
        mass velocity (in km/s)
    M : float
        mass parameter m*G (in km3/s2)
    s : np.ndarray
        rotation vector (default is [0,0,0])
    J2 : float
        oblateness parameter (default is 0)
    R : float
        mass radius (in km, default is 0)

    Returns
    -------
    np.ndarray
        perturbation on the direction of observation
    """
    # debug parameter, if true print some information
    debug = False

    # evaluate distance mass-source
    r = x - x_a
    # evaluate distance mass-observer
    r_obs = x_obs - x_a

    # evaluate vector norm
    r_norm = np.linalg.norm(r)
    r_obs_norm = np.linalg.norm(r_obs)

    # evaluate normal vectors
    n = r / r_norm
    n_obs = r_obs / r_obs_norm

    # evaluate parameters
    sigma = np.dot(x - x_obs, l0)
    d = r_obs - l0 * np.dot(r_obs, l0)
    d2 = np.linalg.norm(d) ** 2
    dv = np.cross(l0, np.cross(v, l0))

    if debug:
        print(f'r_obs: {r_obs} km\n'
              f'r_obs_norm: {r_obs_norm} km\n'
              f'r_obs*l0: {np.dot(r_obs, l0)} \n'
              f'd: {d}')
        print(f'd = {np.sqrt(d2)} km')

    if J2 == 0 and R == 0:

        # evaluate useful combinations
        p1 = 1 / r_norm - 1 / r_obs_norm
        p2 = l0 - 2 * eps * (2 * dv - d * np.dot(v, r_obs) / d2)
        p3 = 2 * d * ((1 - 2 * eps * np.dot(v, l0)) * (np.dot(n, l0) - np.dot(n_obs, l0))) / d2
        p4 = 2 * eps * r_obs_norm * (dv - 2 * d * np.dot(v, d) / d2) / (d2 * r_norm)
        p5 = r_norm - r_obs_norm - np.dot(n_obs, l0) * sigma

        if debug: print(f'p1: {p1}\np2: {p2}\np3: {p3}\np4: {p4}\np5: {p5}\n'
                        f'p1*p2+p3: {p1*p2+p3}\n'
                        f'p4*p5{p4 * p5}')

        # evaluate deflection
        dl = -M * eps ** 2 * (p1 * p2 + p3) + M * (eps ** 2) * p4 * p5

        if debug: print(f'dl: {dl}\n')

        return dl
    else:
        # define three ON vectors
        n = -d / np.linalg.norm(d)
        t = l0
        m = np.cross(t, n)

        # # evaluate useful combinations
        # p2 = (((J2*R**2)/d2) * (1 - np.dot(s, t)**2 - 2*np.dot(s, n)**2))*n
        # p3 = (((J2*R**2)/d2)*np.dot(s, m)*np.dot(s, n))*m
        #
        # # evaluate deflection
        # dl = ((4*M*eps**2)/np.sqrt(d2))*(p2 + p3)

        # evaluate chi
        chi = np.arccos(np.dot(x - x_obs, x_a - x_obs) / (np.linalg.norm(x - x_obs) * np.linalg.norm(x_a - x_obs)))

        # evaluate useful combinations
        p1 = 1 + np.cos(chi) + 0.5 * np.cos(chi) * np.sin(chi) ** 2
        p2 = -2 * (1 + np.cos(chi) + 0.5 * np.cos(chi) * np.sin(chi) ** 2
                   + 3 / 4 * np.cos(chi) * np.sin(chi) ** 4) * np.dot(n, s) ** 2
        p3 = (np.sin(chi) ** 3 - 3 * np.sin(chi) ** 5) * np.dot(n, s) * np.dot(t, s)
        p4 = -(1 + np.cos(chi) + 0.5 * np.cos(chi) * np.sin(chi) ** 2
               - 3 / 2 * np.cos(chi) * np.sin(chi) ** 4) * np.dot(t, s) ** 2
        p5 = 2 * (1 + np.cos(chi) + 0.5 * np.cos(chi) * np.sin(chi) ** 2) * np.dot(n, s) * np.dot(m, s)
        p6 = np.sin(chi) ** 3 * np.dot(m, s) * np.dot(t, s)

        # evaluate deflection
        dl = (2 * M * eps ** 2 / np.sqrt(d2)) * (J2 * R ** 2 / d2) * ((p1 + p2 + p3 + p4) * n + (p5 + p6) * m)

        if debug: print(f'd = {np.sqrt(d2)}')

        return dl


def dx(l_obs, l0, x, x_a, x_obs, eps, v, M):
    """
    Evaluate  the difference between the unperturbed position of a source and the perturbed position.

    Parameters
    ----------
    l_obs : np.ndarray
        observed direction
    l0 : np.ndarray
        unperturbed direction
    x : np.ndarray
        source position (in km)
    x_a : np.ndarray
        mass position (in km)
    x_obs : np.ndarray
        observer position (in km)
    eps : float
        small parameter, usually 1/c (in s/km)
    v : np.ndarray
        mass velocity (in km/s)
    M : float
        mass parameter m*G (in km3/s2)

    Returns
    -------
    np.ndarray
        perturbation on the direction of observation
    """

    # debug parameter, if true print some information
    debug = False

    # evaluate distance mass-source
    r = x - x_a
    # evaluate distance mass-observer
    r_obs = x_obs - x_a

    # evaluate vector norm
    r_norm = np.linalg.norm(r)
    r_obs_norm = np.linalg.norm(r_obs)

    # evaluate normal vectors
    n = r / r_norm
    n_obs = r_obs / r_obs_norm

    # evaluate parameters
    sigma = np.dot(x - x_obs, l0)
    d = r_obs - l0 * np.dot(r_obs, l0)
    d2 = r_obs_norm ** 2 - (np.dot(r_obs, l0)) ** 2
    dv = np.cross(l0, np.cross(v, l0))

    if debug: print(f'd = {np.sqrt(d2)} km')

    # evaluate useful combinations
    p1 = l0 * ((1 + eps * np.dot(v, l0)) * np.log((r_norm + np.dot(r, l0)) / (r_obs_norm + np.dot(r_obs, l0)))
               - sigma / r_obs_norm)
    p2 = 2 * (d / d2) * (r_norm - r_obs_norm - np.dot(n_obs, l0) * sigma)
    p3 = (l_obs / (r_norm + np.dot(r, l0))) * (
                (r_norm - r_obs_norm + sigma) * np.dot(v, d) / (r_obs_norm + np.dot(r_obs, l0))
                - np.dot(v, l0) * sigma)
    p4 = -2 * dv * (np.log((r_norm + np.dot(r, l0)) / (r_obs_norm + np.dot(r_obs, l0))) - 2 * sigma / r_obs_norm)
    p5 = -r_norm * r_obs_norm / d2 * (dv - 2 * d / d2 * np.dot(v, d)) * (np.dot(n, l0) - np.dot(n_obs, l0))
    p6 = 2 * d / (d2 * r_obs_norm) * (np.dot(v, l0) * np.dot(r_obs, l0) - np.dot(v, d)) * sigma

    # evaluate deflection
    deltax = l_obs * sigma - M * eps ** 2 * (p1 + p2) - M * eps ** 3 * (p3 + p4 + p5 + p6)

    if debug: print(f'dl: {deltax}')

    return deltax


def er_deflection(l0, x, x_a, x_obs, eps, M, v=[0,0,0], J2=0, R=0, s=[0,0,0], c1=True, c2=True, quad=True):
    """ Evaluate the light deflection of a Erez-Rosen blak hole. This function is valid only in grazing condition.

    Parameters
    ----------
    l0 : np.ndarray
        unperturbed direction
    x : np.ndarray
        source position (in km)
    x_a : np.ndarray
        mass position (in km)
    x_obs : np.ndarray
        observer position (in km)
    eps : float
        small parameter, usually 1/c (in s/km)
    M : float
        mass parameter m*G (in km3/s2)
    J2 : float
        oblateness parameter
    R : float
        mass radius (in km)
    c1 : bool
        evaluate or not the monopole correction
    c2 : bool
        evaluate or not the quadrupole correction
    quad : bool
        evaluate or not the quadrupole contribution

    Returns
    -------
    dl : np.ndarray
        perturbation angle (in rad)
    """
    # debug parameter, if true print some information
    debug = False

    # evaluate distance mass-source
    r = x - x_a
    # evaluate distance mass-observer
    r_obs = x_obs - x_a

    # evaluate vector norm
    r_norm = np.linalg.norm(r)
    r_obs_norm = np.linalg.norm(r_obs)

    # evaluate parameters
    d = r_obs - l0 * np.dot(r_obs, l0)
    d2 = np.linalg.norm(d) ** 2

    return Erez_Rosen_formula(np.linalg.norm(d), eps, M, J2, R, c1, c2, quad)


def ellis_deflection(l0, x, x_a, x_obs, a):
    """ Evaluate the light deflection of a Ellis wormhole.

    Parameters
    ----------
    l0 : np.ndarray
        unperturbed direction
    x : np.ndarray
        source position (in km)
    x_a : np.ndarray
        mass position (in km)
    x_obs : np.ndarray
        observer position (in km)
    a : float
        wormhole radius (in km)

    Returns
    -------
    dl : np.ndarray
        perturbation on the direction of observation
    """
    # debug parameter, if true print some information
    debug = False

    # evaluate distance mass-source
    r = x - x_a
    # evaluate distance mass-observer
    r_obs = x_obs - x_a

    # evaluate vector norm
    r_norm = np.linalg.norm(r)
    r_obs_norm = np.linalg.norm(r_obs)

    # evaluate parameters
    d = r_obs - l0 * np.dot(r_obs, l0)
    d_norm = np.linalg.norm(d)

    return Ellis_formula(d_norm, a)


def cs_beta(beta, ds, dls, theta, q=0):
    """ This function evaluates the centroid shift of a source with a given value
    of the impact angle beta expressed in units of Einstein ring. It is valid only in grazing conditions.

    Parameters
    ----------
    beta : float
        impact angle in units of einstein ring
    ds : float
        observer - source distance
    dls : float
        lens - source distance
    theta : float
        einstein ring
    q : float
        Erez-Rosen quadrupole term
    Returns
    -------
    dtheta : float
        centroid shift
    """

    # expansion parameter
    xi = theta * ds / (4 * dls)

    p1 = beta / (beta ** 2 + 2)
    p2 = -(15 * np.pi * (beta ** 2 + 1)) / (8 * (beta ** 2 + 2) ** 2)
    p3 = 8 / 3 * dls ** 2 / ds ** 2 * (beta ** 4 + 9 * beta ** 2 - 2)
    p4 = -16 * (dls / ds * beta ** 2 - 2)
    p5 = - (225 * np.pi ** 2) / (128 * (beta ** 2 + 2))
    p6 = -4 / 15 * q

    dtheta = p1 + p2 * xi + (beta / (beta ** 2 + 2) ** 2) * (p3 + p4 + p5 + p6) * xi ** 2

    return dtheta * theta


def centroid_shift(x, x_a, x_obs, eps, M, J2, R):
    """ This function evaluates the centroid shift of a source. It is valid only in grazing conditions, i.e.
    x, x_a, x_obs have to be quasi-alligned.

    Parameters
    ----------
    x : np.ndarray
        source position (in km)
    x_a : np.ndarray
        mass position (in km)
    x_obs : np.ndarray
        observer position (in km)
    eps : float
        small parameter, usually 1/c (in s/km)
    M : float
        mass parameter m*G (in km3/s2)
    J2 : float
        oblateness parameter
    R : float
        mass radius (in km)

    Returns
    -------
    dtheta : float
        centroid shift

    """
    # debug parameter
    debug = False

    # distances
    dl = np.linalg.norm(x_a - x_obs)
    ds = np.linalg.norm(x - x_obs)
    dls = np.linalg.norm(x - x_a)

    # einstein ring
    theta_e = np.sqrt((4 * M * dls) / (dl * ds) * eps ** 2)

    # expansion parameter
    xi = theta_e * ds / (4 * dls)

    # beta tilde
    beta = np.arccos(np.dot(x_a - x_obs, x - x_obs)
                     / (np.linalg.norm(x_a - x_obs) * np.linalg.norm(x - x_obs))) / theta_e

    if debug: print(f'beta = {beta * theta_e}')

    # q
    q = - J2 * R ** 2 / (M ** 2 * eps ** 4) * (15 / 2)

    # evaluate useful combinations
    # p1 = beta * (beta ** 2 + 3) / (beta ** 2 + 2)
    p1 = beta / (beta ** 2 + 2)
    p2 = -(15 * np.pi * (beta ** 2 + 1)) / (8 * (beta ** 2 + 2) ** 2)
    p3 = 8 / 3 * dls ** 2 / ds ** 2 * (beta ** 4 + 9 * beta ** 2 - 2)
    p4 = -16 * (dls / ds * beta ** 2 - 2)
    p5 = - (225 * np.pi ** 2) / (128 * (beta ** 2 + 2))
    p6 = -4 / 15 * q

    if debug: print(f'beta: {beta},   theta: {theta_e}')
    if debug: print(f'p1: {p1}, p2: {p2}, p3: {p3}, p4:{p4}, p5: {p5}, p6:{p6}')

    # evaluate dtheta
    dtheta = p1 + p2 * xi + (beta / (beta ** 2 + 2) ** 2) * (p3 + p4 + p5 + p6) * xi ** 2

    return dtheta * theta_e


def deflection_mod2(l0, x, x_a, x_obs, eps, M,
                J2=0, R=0, s=np.array([0, 0, 0]), v=[0,0,0]):
    """ Evaluate the light deflection angle using Crosta-Mignard formula. If one of the last three
        parameters is set to its default value it evaluates the monopole contribution whereas it evaluates only the
        quadrupole contribution.

        Parameters
        ----------
        l0 : np.ndarray
            unperturbed direction
        x : np.ndarray
            source position (in km)
        x_a : np.ndarray
            mass position (in km)
        x_obs : np.ndarray
            observer position (in km)
        eps : float
            small parameter, usually 1/c (in s/km)
        M : float
            mass parameter m*G (in km3/s2)
        s : np.ndarray
            rotation vector (default is [0,0,0])
        J2 : float
            oblateness parameter (default is 0)
        R : float
            mass radius (in km, default is 0)

        Returns
        -------
        float
            perturbation on the direction of observation
        """
    # evaluate distance mass-source
    r = x - x_a
    # evaluate distance mass-observer
    r_obs = x_obs - x_a

    # evaluate parameters
    d = r_obs - l0 * np.dot(r_obs, l0)
    chi = np.arccos(np.dot(x - x_obs, x_a - x_obs) / (np.linalg.norm(x - x_obs) * np.linalg.norm(x_a - x_obs)))

    return CM_formula(l0, d, chi, M, eps, J2, R, s)


def deflection_mod3(l0, x_a, eps, M,
               s=np.array([0, 0, 1]), J2=0, R=0):
    """ Evaluate the light deflection angle using Crosta-Mignard formula. If one of the last three
    parameters is set to its default value it evaluates the monopole contribution whereas it evaluates only the
    quadrupole contribution. The quantities are centered on the observer.


    Parameters
    ----------
    l0 : np.ndarray
        unperturbed direction
    x_a : np.ndarray
        mass position (in km)
    eps : float
        small parameter, usually 1/c (in s/km)
    M : float
        mass parameter m*G (in km3/s2)
    s : np.ndarray
        rotation vector (default is [0,0,0])
    J2 : float
        oblateness parameter (default is 0)
    R : float
        mass radius (in km, default is 0)

    Returns
    -------
    float
        perturbation on the direction of observation
    """
    # debug parameter, if true print some information
    debug = False

    # evaluate parameters
    d = (x_a - l0 * np.dot(x_a, l0))
    d2 = np.linalg.norm(d) ** 2
    cos_chi = np.dot(x_a, l0)/np.linalg.norm(x_a)
    sin_chi = np.linalg.norm(np.cross(x_a, l0))/np.linalg.norm(x_a)
    chi = np.arccos(np.dot(x_a, l0)/np.linalg.norm(x_a))

    if debug: print(f'chi: {np.rad2deg(np.arccos(cos_chi))*3600/20}')

    return CM_formula(l0, d, chi, M, eps, J2, R, s)

#########################
#
# Main function
#
#########################


METHODS = {'CM': deflection_mod2,
           'RAMOD': deflection,
           'ER': er_deflection}


def light_defl(l0, x, x_a, x_obs, eps, M, v=np.array([0, 0, 0]),
               s=np.array([0, 0, 1]), J2=0, R=0, method='CM', *args):
    """
       Evaluate  the difference between the unperturbed direction l0 and the perturbed direction. If J2 and R are
       set to its default value it evaluates the monopole contribution whereas it evaluates only the
       quadrupole contribution.

       Parameters
       ----------
       l0 : np.ndarray
           unperturbed direction
       x : np.ndarray
           source position (in km)
       x_a : np.ndarray
           mass position (in km)
       x_obs : np.ndarray
           observer position (in km)
       eps : float
           small parameter, usually 1/c (in s/km)
       v : np.ndarray
           mass velocity (in km/s), this parameter gives contribtion only with 'RAMOD'
       M : float
           mass parameter m*G (in km3/s2)
       s : np.ndarray
           rotation vector (default is [0,0,1])
       J2 : float
           oblateness parameter (default is 0)
       R : float
           mass radius (in km, default is 0)
       method : str
           method for evaluation of light deflection, default is Crosta-Mignard

       Returns
       -------
       any
           perturbation on the direction of observation
       """

    method = METHODS[method]

    return method(l0, x, x_a, x_obs, eps, M, v=v, s=s, J2=J2, R=R, *args)


if __name__ == "__main__":

    from planets import *
    from astropy import constants

    # Define constants
    pc = constants.pc.to('km').value
    AU = constants.au.to('km').value
    c = constants.c.to('km/s').value
    eps = 1 / c

    ss = SolarSystem()
    jupiter = ss.getPlanet('jupiter')

    x = np.array([0, 1, 0])*pc

    # observer
    x_obs = AU * np.array([0, -1, 0])

    # impact angle
    chi = jupiter.radius / np.linalg.norm(x_obs - jupiter.pos)
    # direction
    l0 = -np.array([np.sin(chi), np.cos(chi), 0])
    x = -np.linalg.norm(x - x_obs) * l0 + x_obs

    for method in METHODS:
        print(rad2muas(np.linalg.norm(light_defl(l0, x, jupiter.pos, x_obs, eps, jupiter.mass, method=method))))

