import numpy as np
import pandas as pd

data = pd.read_csv('PS_2022.03.28_01.02.29.csv', comment='#')
print(data.columns)

# take data with no Nan distance
data_clean = data.loc[pd.notna(data['sy_dist'])]

# take numerical data
num_data = data_clean[['pl_masse', 'sy_dist']]
nump = num_data.to_numpy()
print(num_data.head())

print(nump)