""" Description

Creator: Mattia Falco
Date: 01/04/2022 
"""

import pandas as pd
import numpy as np


def save_df(data, columns, index, path):
    """ Save the data in a csv file with the passed information.

    Parameters
    ----------
    data : list
        numerical data to be saved
    columns : list
        name of columns
    index : list
        name of rows
    path : str
        file path
    """

    data_saved = pd.DataFrame(np.array(data).T,
                              columns=columns,
                              index=index)
    data_saved.to_csv(path, sep=',')


def read_df(path):
    """ Give data in form of pd.DataFrame.

    Parameters
    ----------
    path : str
        path of data

    Returns
    -------
    pd.DataFrame
        required data
    """

    return pd.read_csv(path, index_col=0)


