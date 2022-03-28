import numpy as np
import pandas as pd

data = pd.read_csv('PS_2022.03.28_05.32.23.csv', comment='#')
print(data.columns)

# take data with no Nan distance
data_clean = data.loc[pd.notna(data['sy_dist'])]

# take numerical data
# num_data = data_clean[['pl_masse', 'sy_dist']]

# data_clean = data_clean.set_index('pl_name')
data_clean = data_clean.drop_duplicates(subset=['pl_name'])

filt = data_clean['hostname']=='Proxima Cen'
print(data_clean.loc[filt])



# nump = num_data.to_numpy()
# print(num_data.head())
#
# print(nump)