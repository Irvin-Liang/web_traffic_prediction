#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 01:05:12 2017

@author: qianlinliang
"""

import pandas as pd
import numpy as np

from settings import *
from helper import to_submission


# Function define

def naive_mean(row):
    
    ts = row[START_INDEX:]
    
    mean = np.round(np.mean(ts), 4)
    
    return pd.Series(data=len(PREDICT_DATE)*[mean])


def naive_median(row):    
#    ts = row[START_INDEX:]
    ts = row[-50:]
    
    median = np.round(np.median(ts), 4)
    
    return pd.Series(data=len(PREDICT_DATE)*[median])


def naive_last(row):
    
    last_value = row[-1]
    
    return pd.Series(data=len(PREDICT_DATE)*[last_value])
    
    
    
    
    
# LOAD DATA

ts_data = pd.read_csv('../data/train_1_parsed.csv')

#print("Use mean to make prediction...")
#ts_data[PREDICT_DATE] = ts_data.apply(naive_mean, axis=1)

print("Use median to make prediction...")
ts_data[PREDICT_DATE] = ts_data.apply(naive_last, axis=1)


# LOAD key

key = pd.read_csv('../data/key_1.csv')
key = key.set_index('Page')

result = to_submission(ts_data, key)

result.to_csv('../last_value_result.csv', index=False)