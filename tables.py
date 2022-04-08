""" Description

Creator: mattiafalco
Date: 08/04/2022 
"""

import pandas as pd
from save_df import *
from read_exo import getExo
from astropy import constants

# Define constants
pc = constants.pc.to('km').value
AU = constants.au.to('km').value
c = constants.c.to('km/s').value
eps = 1/c

# read exo catalogue
path = 'exo_archive.csv'
catalogue = pd.read_csv(path)

targets = ['Proxima Cen b', 'Kepler-220 b', 'Kepler-847 b', 'Kepler-288 b', 'OGLE-2015-BLG-0966L b',
           'OGLE-2014-BLG-0124L b', 'KMT-2016-BLG-2605L b', 'OGLE-2008-BLG-092L b',
           'GJ 1252 b', 'HR 858 c', 'WASP-84 b', 'K2-80 b', 'HAT-P-46 b']

dist = []
dr = []
dr_no_sun = []

for target in targets:

    path = f'Data/pointing_errors_{target}_2122-01-01T12:00:00.csv'

    df = read_df(path)
    errors = df['dr']
    errors_no_sun = df['dr - sun']
    dr.append(errors[-1])
    dr_no_sun.append(errors_no_sun[-1])

    pl = getExo(target, catalogue)
    dist.append(pl.dist/pc)

rows = targets
columns = ['dist', 'dr', 'dr - sun']
data = [dist,
        dr,
        dr_no_sun]
path = 'test_tab'
save_df(data, columns, rows, path)

