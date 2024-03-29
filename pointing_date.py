from astronavigation.deflection import *
from astropy import constants
from astronavigation.planets import SolarSystem
import pandas as pd
from astronavigation.read_exo import getExo
from astronavigation.save_df import save_df
import matplotlib.pyplot as plt


# Define constants
pc = constants.pc.to('km').value
AU = constants.au.to('km').value
c = constants.c.to('km/s').value
eps = 1/c

# read exo catalogue
path = 'exo_archive.csv'
catalogue = pd.read_csv(path)

# save parameter
save = False
save_latex = True

######################################
#
# Write here the data of the problem
#
######################################

# observer
obs = 'earth'
# masses
list_p = ['sun', 'jupiter', 'saturn', 'uranus', 'neptune']
# targets
targets = ['Proxima Cen b', 'Kepler-220 b', 'Kepler-847 b', 'Kepler-288 b', 'OGLE-2015-BLG-0966L b',
           'OGLE-2014-BLG-0124L b', 'WASP-84 b'] #'KMT-2016-BLG-2605L b', 'OGLE-2008-BLG-092L b',
           # 'GJ 1252 b', 'HR 858 c', 'K2-80 b', 'HAT-P-46 b']
# date
date_ref = np.datetime64('2122-01-01T12:00:00')
month = 0  # set 0 to have only one date

#################
#
# Algorithm
#
#################

# create dates
day = np.timedelta64(30, 'D')
dates = [date_ref]
for i in range(month):
    dates.append(dates[-1]+day)
dates = np.datetime_as_string(dates)

# no dot save multiple days
if len(dates) > 1:
    save = False

# list in which save data
dr_date = []
dr_nosun_date = []

# loop over days
for date in dates:

    print(f'\n--------------------------\n'
          f'evaluating date: {date}')

    # Create Solar System with ephemerides
    ss = SolarSystem(ephemerides=True)

    # observer
    x_obs = ss.getPlanet(obs, date=date).pos

    # take bodies which generate grav. field
    planets = [ss.getPlanet(pl, date=date) for pl in list_p]
    v_null = np.array([0, 0, 0])

    # temporary lists
    dr_temp = []
    dr_nosun_temp = []

    # loop over targets
    for target in targets:

        # target
        x = getExo(target, catalogue).pos
        l0 = -(x - x_obs) / (np.linalg.norm(x - x_obs))

        # Evaluate deflection for each body
        dl1 = []
        dl2 = []  # w/ null velocities
        dpsi = []
        dl1_q = []
        dpsi_q = []

        for pl in planets:
            # monopole deflection
            dls = deflection(l0, x, pl.pos, x_obs, eps, pl.mass, pl.vel)
            # monopole deflection w/ null vel
            dls_vn = deflection(l0, x, pl.pos, x_obs, eps, pl.mass, v_null)
            # quadrupole deflection
            dlq = deflection(l0, x, pl.pos, x_obs, eps, pl.mass, pl.vel, pl.s, pl.J2, pl.radius)

            # save norm values in lists
            dl1.append(np.linalg.norm(dls))
            dl2.append(np.linalg.norm(dls_vn))
            dl1_q.append(np.linalg.norm(dlq))

            # projections orthogonal to l0
            dln = dls - l0*np.dot(dls, l0)
            dln_vn = dls_vn - l0*np.dot(dls_vn, l0)
            dln_q = dlq - l0 * np.dot(dlq, l0)

            # save norm values
            dpsi.append(np.linalg.norm(dln))
            dpsi_q.append(np.linalg.norm(dln_q))

        # transform lists into array
        dl1 = np.array(dl1)
        dl2 = np.array(dl2)
        dpsi = np.array(dpsi)
        dl1_q = np.array(dl1_q)
        dpsi_q = np.array(dpsi_q)

        # Take the cumulative sum
        dlt = np.cumsum(dl1)
        dlt_vn = np.cumsum(dl2)
        dpsi_tot = np.cumsum(dpsi)

        # evaluate dr
        dr = np.linalg.norm(x - x_obs) * dl1

        #################
        #
        # Printing
        #
        #################
        if len(dates) == 1:
            print(f'dl: {np.rad2deg(dlt)*3600*1e6} muas')
            print(f'dl - sun: {np.rad2deg(dlt-dlt[0])*3600*1e6} muas\n')

            # point error
            print(f'pointing errors: {dr/AU} AU')
            print(f'pointing errors - sun: {(dr-dr[0])/AU} AU\n')

            # quadrupole
            print(f'dl quadrupole: {np.rad2deg(dl1_q)*3600*1e6} muas')
            print(f'dr quadrupole: {np.linalg.norm(x-x_obs)*dl1_q/AU} AU')

        #################
        #
        # Saving
        #
        #################

        if save:
            columns = ['dl_vn', 'dl', 'dlt', 'dlt - sun', 'dr', 'dr - sun']
            index = list_p
            path = f'Data/pointing_errors_{target}_{date}'
            save_df([np.rad2deg(dl2)*3600*1e6,
                     np.rad2deg(dl1) * 3600 * 1e6,
                     np.rad2deg(dlt)*3600*1e6,
                     np.rad2deg(dlt-dlt[0])*3600*1e6,
                     dr/AU,
                     (dr-dr[0])/AU],
                    columns=columns,
                    index=index,
                    path=path)

        if save_latex:
            columns = ['dl_vn', 'dl', 'dlt', 'dlt - sun', 'dr', 'dr - sun']
            index = list_p
            path = f'Data/pointing_date_{target}_latex'
            save_df([np.round(np.rad2deg(dl2)*3600*1e6, 2),
                     np.round(np.rad2deg(dl1) * 3600 * 1e6, 2),
                     np.round(np.rad2deg(dlt)*3600*1e6, 2),
                     np.round(np.rad2deg(dlt-dlt[0])*3600*1e6, 2),
                     np.round(dr/AU, 9),
                     np.round((dr-dr[0])/AU, 9)],
                    columns=columns,
                    index=index,
                    path=path)

        # save dr
        dr_temp.append(dr[-1])
        dr_nosun_temp.append(dr[-1]-dr[0])

    # save dr
    dr_date.append(dr_temp)
    dr_nosun_date.append(dr_nosun_temp)

    #################
    #
    # Plotting
    #
    #################

    if len(dates) == 1:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot(x_obs[0] / AU, x_obs[1] / AU, x_obs[2] / AU, label=obs, marker='x')
        for i, pl in enumerate(planets):
            x_p = pl.pos
            ax.plot(x_p[0] / AU, x_p[1] / AU, x_p[2] / AU, label=list_p[i], marker='.')
        for target in targets:
            x = getExo(target, catalogue).pos
            l0 = -(x - x_obs) / (np.linalg.norm(x - x_obs))
            ax.plot(l0[0] * 50, l0[1] * 50, l0[2] * 50, label=target, marker='*')
        plt.legend(loc=(1.05, 0.3), fontsize='xx-small')
        plt.title('Solar System, 01/01/2122')
        ax.set_xlabel('x [AU]')
        ax.set_ylabel('y [AU]')
        ax.set_zlabel('z [AU]')
        plt.show()

dr_date = np.array(dr_date).T
dr_nosun_date = np.array(dr_nosun_date).T
#################
#
# Plotting
#
#################
if len(dates) > 1:
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(dr_date[6]/AU)

    ax.set_xlabel('t [month]')
    ax.set_ylabel('dr [AU]')
    plt.title('Sun pointing error')

    fig2 = plt.figure()
    ax2 = plt.axes()
    ax2.plot(dr_nosun_date[6]/AU)

    ax2.set_xlabel('t [month]')
    ax2.set_ylabel('dr [AU]')
    plt.title('Planets pointing error')

    plt.show()

