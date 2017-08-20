#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 21:28:57 2017

@author: qianlinliang
"""

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt


ts_data = pd.read_csv('data/train_1_parsed.csv')


def fill_missed_data(row):
    
    for i in range(8, 558):
        
        
        if np.isnan(row[i]):
            # if it is the first element
            if i == 8:
                # fill with next element
                row[i] = row[i+1]
            else:
                # fill with previous element
                row[i] = row[i-1]
                
            
            if np.isnan(row[i]):
                row[i] = 0
                   
    return row



ts_data = ts_data.apply(fill_missed_data, axis=1)
