# -*- coding: utf-8 -*-
"""
EE 511

Final Project
"""

import pandas as pd
import numpy as np
from numpy.random import default_rng


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


def generate_data_sets(path):
    rng = default_rng(seed=6)
    data = pd.read_csv(path, usecols=['publication', 'title', 'article'])

    # drop rows where publisher is na
    data.drop(np.nonzero((data['publication'].isna()).values)[0], inplace=True)
    data = data.reindex(np.arange(len(data)))

    remain_idx = np.arange(len(data))

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

    for each in tgt_classes:
        # for each class, select 20K
        class_idxs[each] = np.nonzero((data['publication'] == each).values)[0]
        rng.shuffle(class_idxs[each])
        train_idxs[each] = class_idxs[each][:16000]
        val_idxs[each] = class_idxs[each][16000:18000]
        test_idxs[each] = class_idxs[each][18000:20000]

    class_train_idx = np.concatenate([arr for arr in train_idxs.values()])
    class_val_idx = np.concatenate([arr for arr in val_idxs.values()])
    class_test_idx = np.concatenate([arr for arr in test_idxs.values()])

    all_class_idx = np.concatenate([arr for arr in class_idxs.values()])

    extra_idx = np.setdiff1d(remain_idx, all_class_idx, assume_unique=True)
    data['publication'].loc[extra_idx] = 'Other'
    rng.shuffle(extra_idx)
    other_train_idx = extra_idx[:16000]
    other_val_idx = extra_idx[16000:18000]
    other_test_idx = extra_idx[18000:20000]

    all_train_idx = np.concatenate((class_train_idx, other_train_idx))
    all_val_idx = np.concatenate((class_val_idx, other_val_idx))
    all_test_idx = np.concatenate((class_test_idx, other_test_idx))

    train = data.loc[all_train_idx].copy()
    val = data.loc[all_val_idx].copy()
    test = data.loc[all_test_idx].copy()

    return train, val, test
