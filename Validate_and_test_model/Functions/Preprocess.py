# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 12:55:35 2025

@author: Fabienne Doll
"""
import numpy as np
import random
import numpy.random
import tensorflow as tf
import pandas as pd
import glob



np.random.seed(42)
tf.random.set_seed(42) 

tf.config.experimental.enable_op_determinism()

#%%

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_weekly_Data(i):
    path = "../Data"
    
    pathconnect = "/"

    Data_list = glob.glob(path+pathconnect+'*.csv');

    
    Well_ID = Data_list[i]
    Well_ID = Well_ID.replace(path+'\\', '')
    Well_ID = Well_ID.replace('.csv', '') 
    
    #Load time series
    Data = pd.read_csv(path+pathconnect+Well_ID+'.csv', 
                          parse_dates=['Date'],index_col=0, 
                          decimal = '.', sep=',')
  
    return Data, Well_ID

def make_sequence(data, SETTINGS):
    X, Y = list(), list()
    # step over the entire history one time step at a time
    for i in range(len(data)):
        # find the end of this pattern
        end_idx = i + SETTINGS['seq_length']
        # check if we are beyond the dataset
        if end_idx >= len(data):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = data[i:end_idx, 2:], data[end_idx, :2] #2 weil spalte 2 (index1) = key
        X.append(seq_x)
        Y.append(seq_y)
    return np.array(X), np.array(Y)

