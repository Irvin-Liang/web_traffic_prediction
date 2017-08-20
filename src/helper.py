#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 01:23:28 2017

@author: qianlinliang
"""

import pandas as pd

from tqdm import tqdm
from settings import *

# convert data frame to submission file
def to_submission(df, key):
    
    
    rows = []

    for i, row in tqdm(df.iterrows()):
        for date in PREDICT_DATE:
            page = row.Page + '_' + date
            
            rows.append([key.loc[page, 'Id'],
                         row[date]])
            
            
            
    return pd.DataFrame(data=rows, columns=['Id', 'Visits'])
