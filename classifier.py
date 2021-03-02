# -*- coding: utf-8 -*-
"""
EE 511

Final Project
"""

import pandas as pd


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
