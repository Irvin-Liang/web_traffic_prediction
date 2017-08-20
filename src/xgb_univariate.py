#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 15:03:44 2017

@author: qianlinliang
"""

import pandas as pd
import numpy as np

import xgboost as xgb

from tqdm import tqdm
from settings import PREDICT_DATE, DATA_FILENAME

#================== FUNCTION DEFILE FIELD ===============

def split_train_valid(df, pred_window, nb_day, nb_valid):
    
    columns = np.append('Page', df.columns.values[-nb_day:])
    data = df.loc[:, columns]
    
    train_rows = []
    
    for i, row in tqdm(data.iterrows()):
        for j in range(1, nb_day-nb_valid-pred_window):
            train_rows.append(row[j:j+pred_window+1].values)
            
    return pd.DataFrame(data = train_rows)
            

    


#================== LOAD DATA ===========================
df = pd.read_csv(DATA_FILENAME)

train = split_train_valid(df, 50, 180, 60)


d_train = xgb.DMatrix(train.iloc[:, :-1], train.iloc[:, -1])

watchlist = [(d_train, 'train')]

params = dict(objective='reg:linear',
              eval_metric='rmse',
              eta=0.02,
              max_depth=7,
              subsample=.6)

bst = xgb.train(params,
                d_train,
                2000,
                watchlist,
                early_stopping_rounds=50,
                verbose_eval=1)