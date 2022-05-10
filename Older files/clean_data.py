""" This script generates a clean dataset from the NASA exoplanet archive
that can be found at http://exoplanetarchive.ipac.caltech.edu
This file can also be used as a template to write a code for data reading.
Creator: Mattia Falco
Date: 27/03/2022
"""
import pandas as pd

archive = 'PS_2022.03.28_05.32.23.csv'

data = pd.read_csv(archive, comment='#')
print(data.columns)

# take data with no Nan distance
data_clean = data.loc[pd.notna(data['sy_dist'])]

# data_clean = data_clean.set_index('pl_name')
data_clean = data_clean.drop_duplicates(subset=['pl_name'])

data_clean = data_clean.set_index('pl_name')

# filt = data_clean['hostname']=='Proxima Cen'
# print(data_clean.loc[filt])

# nump = num_data.to_numpy()
# print(num_data.head())
#
# print(nump)

if __name__ == "__main__":
    data_clean.to_csv('exo_archive.csv', sep=',', index=True)
