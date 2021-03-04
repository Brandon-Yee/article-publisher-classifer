# -*- coding: utf-8 -*-
"""
EE 511

Final Project
"""

import pandas as pd
import numpy as np
from numpy.random import default_rng

"""
# 
# https://towardsdatascience.com/3-simple-ways-to-handle-large-data-with-pandas-d9164a3c02c1
columns = ['Unnamed: 0', 'Unnamed: 0.1', 'date', 'year', 'month', 'day', 'author', 'title', 'article', 'url', 'section', 'publication']
use_cols = ['year', 'month', 'day', 'title', 'article', 'section', 'publication']
ignore_cols = ['Unnamed: 0', 'Unnamed: 0.1', 'date', 'author', 'title', 'article', 'section', 'publication']


chunksize = 1000

data_iter = pd.read_csv('./article_data.csv', chunksize=chunksize)

for data in data_iter:
    print(data)
    break
"""

def generate_data_sets(path):
    """
    Generates training, validation and test dataframes with the data in random
    ordering.
    Note: the returned dataframes have an 'orig_index' column (separate from
          the actual index of the dataframe) that indicates the original row
          within the original raw data that it corresponds to.
    PARAMETERS:
    path - string: path to original data csv
    RETURNS:
    train - pandas dataframe: df with 4 columns for training
    val - pandas dataframe: df with 4 columns for validation
    test - pandas dataframe: df w/ 4 cols for test
    """

    rng = default_rng(seed=6)
    data = pd.read_csv(path, usecols=['publication', 'title', 'article'])

    # determine all indexes that are not nan
    any_nan = np.logical_or(data['publication'].isna().values,
                            data['article'].isna().values)
    any_nan = np.logical_or(any_nan, data['title'].isna().values)
    nan_idx = np.nonzero(any_nan)[0]
    remain_idx = np.setdiff1d(np.arange(len(data)), nan_idx, assume_unique=True)

    # named target classes (does not include 'Other')
    tgt_classes = [
        'Reuters',
        'TechCrunch',
        'Economist',
        'CNN',
        'CNBC',
        'Fox News',
        'Politico',
        'The New York Times',
        'Washington Post',
        'Business Insider'
        ]

    class_idxs = {}
    train_idxs = {}
    val_idxs = {}
    test_idxs = {}

    # determine indexes for each named class
    for each in tgt_classes:
        # for each class, select 16K for training, 2K for val, and 2K for test

        # all indexes for the given class
        class_idxs[each] = np.nonzero((data['publication'] == each).values)[0]
        # randomly shuffle the indexes
        rng.shuffle(class_idxs[each])
        # select the needed amount
        train_idxs[each] = class_idxs[each][:16000]
        val_idxs[each] = class_idxs[each][16000:18000]
        test_idxs[each] = class_idxs[each][18000:20000]

    # concatenate all of the named classes of each category
    # to aggregate the data set
    class_train_idx = np.concatenate([arr for arr in train_idxs.values()])
    class_val_idx = np.concatenate([arr for arr in val_idxs.values()])
    class_test_idx = np.concatenate([arr for arr in test_idxs.values()])

    # determine the non-named class indexes and label them 'Other
    all_class_idx = np.concatenate([arr for arr in class_idxs.values()])
    extra_idx = np.setdiff1d(remain_idx, all_class_idx, assume_unique=True)
    data['publication'].loc[extra_idx] = 'Other'

    # shuffle them and select 20K
    rng.shuffle(extra_idx)
    other_train_idx = extra_idx[:16000]
    other_val_idx = extra_idx[16000:18000]
    other_test_idx = extra_idx[18000:20000]

    # add in the 'Other' rows
    all_train_idx = np.concatenate((class_train_idx, other_train_idx))
    all_val_idx = np.concatenate((class_val_idx, other_val_idx))
    all_test_idx = np.concatenate((class_test_idx, other_test_idx))

    # shuffle the indexes so that the classes are not grouped together
    rng.shuffle(all_train_idx)
    rng.shuffle(all_val_idx)
    rng.shuffle(all_test_idx)

    train = data.loc[all_train_idx]
    val = data.loc[all_val_idx]
    test = data.loc[all_test_idx]

    # reset the indexes and adjust the name of the new original index col
    train.reset_index(inplace=True)
    val.reset_index(inplace=True)
    test.reset_index(inplace=True)
    train.rename(columns={'index': 'orig_index'}, inplace=True)
    val.rename(columns={'index': 'orig_index'}, inplace=True)
    test.rename(columns={'index': 'orig_index'}, inplace=True)

    return train, val, test
