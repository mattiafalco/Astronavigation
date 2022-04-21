""" Description

Creator: Mattia Falco
Date: 21/04/2022
"""

import numpy as np
from deflection import *
from astropy import constants
from planets import Body, SolarSystem
import pandas as pd
from save_df import save_df

# Define constants
pc = constants.pc.to('km').value
AU = constants.au.to('km').value
c = constants.c.to('km/s').value
eps = 1/c

path = 'comp_obs_70.csv'

df_targets = pd.read_csv(path)
print(df_targets.columns)

x_star = []
for i in df_targets.index:

    star = df_targets.iloc[i]

    ra = star['ra_s_new']
    dec = star['dec_s_new']
    dist = 10000 * pc

    x_star.append(dist * cartesian(ra, dec))

print([np.linalg.norm(s)/pc for s in x_star])
