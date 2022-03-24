import numpy as np

def deflection(l0, x, x_a, x_obs, eps, v, M, s=0, J2=0, R=0):
    """
    :param l0: unpertubed direction, 3-array
    :param x: source position [km], 3-array
    :param x_a: mass position [km], 3-array
    :param x_obs: observer position [km], 3-array
    :param eps: 1/c
    :param v: mass velocity, 3-array
    :param M: gravitational parameter, mass*G
    :param s: spin vector, 3-array
    :param J2: planet oblateness
    :param R: planet radius [km]
    :return: dl, perturbed direction
    """

    # evaluate distance mass-source
    r = x - x_a
    # evaluate distance mass-observer
    r_obs = x_obs - x_a

    # evaluate vector norm
    r_norm = np.linalg.norm(r)
    r_obs_norm = np.linalg.norm(r_obs)

    # evaluate normal vectors
    n = r/r_norm
    n_obs = r_obs/r_obs_norm

    # evaluate parameters
    sigma = np.dot(x-x_obs, l0)
    d = r_obs - l0*np.dot(r_obs, l0)
    d2 = r_obs_norm**2 - (np.dot(r_obs, l0))**2
    #d = r_obs - l0*r_obs_norm*np.cos(chi)
    #d2 = r_obs_norm**2*np.sin(chi)**2
    dv = np.cross(l0, np.cross(v, l0))

    if np.all(s==0) and J2==0 and R==0:

        # evaluate useful combinations
        p1 = 1/r_norm - 1/r_obs_norm
        p2 = l0 - 2*eps*(2*dv - d*np.dot(v, r_obs)/d2)
        p3 = 2*d*((1-2*eps*np.dot(v, l0))*(np.dot(n, l0) - np.dot(n_obs, l0)))/d2
        p4 = r_obs_norm*(dv - 2*d*np.dot(v, d)/d2)/(d2*r_norm)
        p5 = r_norm - r_obs_norm - np.dot(n_obs, l0)*sigma
        # print(f'p1: {p1}\np2: {p2}\np3: {p3}\np4: {p4}\np5: {p5}\np4*p5{p4*p5}')

        # evaluate deflection
        dl = -M*eps**2*(p1*p2 + p3) + 2*M*(eps**3)*p4*p5

        return dl
    else:
        n = -d/np.linalg.norm(d)
        t = l0
        m = np.cross(t, n)
        p2 = (((J2*R**2)/d2) * (1 - np.dot(s, t)**2 - 2*np.dot(s, n)**2))*n
        p3 = (((J2*R**2)/d2)*np.dot(s, m)*np.dot(s, n))*m
        dl = ((4*M*eps**2)/np.sqrt(d2))*(p2 + p3)
        # print(f'd = {np.sqrt(d2)}')

        return dl


def quadru(i):

    sJ2 = np.zeros(4)

    if i == 1:
        alpha = np.deg2rad(268.057)
        delta = np.deg2rad(64.495)
        sJ2[3] = 14736e-6
    elif i == 2:
        alpha = np.deg2rad(40.589)
        delta = np.deg2rad(83.537)
        sJ2[3] = 16298e-6

    s = [np.cos(alpha)*np.cos(delta), np.sin(alpha)*np.cos(delta), np.sin(delta)]

    for j in range(len(s)):
        sJ2[j] = s[j]

    return sJ2


if __name__ == "__main__":
    AU = 149597870.691  # [km]
    pc = 3.0856775814672e13  # [km]
    c = 299792.458  # [km/s]

    R = 71492
    ang = 90
    chi = np.deg2rad(ang)
    star = 10*pc

    x_a = np.array([0, 5.2*AU, 0])  # Jupiter
    x_obs = np.array([0, AU + 1.5e6, 0])  # L2
    x = np.array([star*np.sin(chi), AU + 1.5e6 + star*np.cos(chi), 0])
    eps = 1/c
    v = np.array([0, 0, 0])
    MG = 1.26686534e8
    l0 = -(x - x_obs)/(np.linalg.norm(x-x_obs))

    print('\n--------------- test -------------\nmonopole deflection at chi = 90 deg')
    dls = deflection(l0, x, x_a, x_obs, eps, v, MG)
    # print(dls)
    print(f'dl_n mono = {np.rad2deg(((np.linalg.norm(dls)))) * 3600 * 1e6} muas')

    print('\n--------------- test -------------\nquadrupole deflection at chi = 22 as')

    ang = 22/3600
    chi = np.deg2rad(ang)
    star = 1 * pc

    x_a = np.array([0, 5.2 * AU, 0])  # Jupiter
    x_obs = np.array([0, AU + 1.5e6, 0])  # L2
    x = np.array([star * np.sin(chi), AU + 1.5e6 + star * np.cos(chi), 0])
    eps = 1 / c
    v = np.array([0, 0, 0])
    MG = 1.26686534e8
    l0 = -(x - x_obs) / (np.linalg.norm(x - x_obs))

    #print(f'exp_chi = {np.rad2deg(np.arcsin(R/(np.linalg.norm(x_obs-x_a))))*3600}')
    print(f'exp_chi = {np.rad2deg(np.arctan(R/(4.2*AU-1.5e6)))*3600} as')

    sJ2 = quadru(1)

    s = sJ2[:3]
    J2 = sJ2[3]

    dls = deflection(l0, x, x_a, x_obs, eps, v, MG, s, J2, R)
    dln = dls - l0*np.dot(dls, l0)
    print(f'dl_n quadru = {np.rad2deg(np.linalg.norm(dln))*3600*1e6} muas')

    print('\n--------------- test -------------\nmonopole deflection Sun + Jupiter')

    star = 1 * pc

    x_s = np.array([0, 0, 0])  # Sun
    x_j = np.array([0, 5.2*AU, 0])  # Jupiter
    x_obs = np.array([AU, 0, 0])  # Earth
    x = np.array([0, star, 0])
    eps = 1/c
    v_s = np.array([0, 0, 0])
    v_j = np.array([0, 0, 0])
    MG_s = 1.32712440017987e11  # [km^2/s^2]
    MG_j = 1.26686534e8  # [km^2/s^2]
    l0 = -(x - x_obs) / (np.linalg.norm(x - x_obs))

    print(f'l0 = {l0}, l0_norm = {np.linalg.norm(l0)}')

    dls = deflection(l0, x, x_s, x_obs, eps, v_s, MG_s)
    dlj = deflection(l0, x, x_j, x_obs, eps, v_j, MG_j)
    dls_norm = np.linalg.norm(dls)
    dlj_norm = np.linalg.norm(dlj)
    dln_j = dlj - l0*np.dot(dlj, l0)
    print(f'dpsi = {np.rad2deg(dlj_norm)*3600*1e6} muas')
