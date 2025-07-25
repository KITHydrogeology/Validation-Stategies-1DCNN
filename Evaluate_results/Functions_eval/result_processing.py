# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 10:59:40 2025

@author: Fabienne Doll
"""

import glob
import numpy as np
import pandas as pd
from scipy import stats
import os


def load_ID(i, exp):
    pathid = os.path.join(exp, "Results")

    pathconnect = "/"

    GWData_list = glob.glob(pathid+pathconnect+ "*");
    
    Well_ID = GWData_list[i]
    Well_ID = Well_ID.replace(pathid+'\\', '')
        
    return  Well_ID

#%%
def load_epochs_cv(ID, exp):    
    # Load epochs set for the current fold
    epochs = pd.read_csv(exp / f'Results/{ID}/Train_Test_Stats/{ID}_cv_epochs_all_folds.csv', decimal=',', sep=';', header=0, index_col=0)
    
    cv_epochs_mean = int(epochs['epochs'].mean(axis=0))
    
    return cv_epochs_mean

def load_epochs_oos(ID, exp):   
    # Load epochs set 
    epochs = pd.read_csv(exp / f'Results/{ID}/Train_Test_Stats/{ID}_cv_epochs_all_folds.csv', decimal=',', sep=';', header=0, index_col=0)

    oos_epochs_mean = int(epochs.iloc[4,0])
    
    return oos_epochs_mean

def load_epochs_repOOS(ID, exp):   
    #Load epochs set for the current repetition
    epochs = pd.read_csv(exp / f'Results/{ID}/Train_Test_Stats/{ID}_repHO_epochs_all_folds.csv', decimal=',', sep=';', header=0, index_col=0)

    repOOS_epochs_mean = int(epochs['epochs'].mean(axis=0))
        
    return repOOS_epochs_mean

#%%

def cv_fold_obs_sim(ID, fold, exp):
    # Load validation set for the current fold
    cv_val_set = pd.read_csv(exp / f'Results/{ID}/Sim_Obs/{ID}_val_sim_obs_{fold}.csv', decimal=',', sep=';', header=0, parse_dates=['Date'],index_col=0)
    cv_val_set.columns = cv_val_set.columns.str.replace(f'_Fold{fold}', '')    #Clean up column names
    cv_val_set['fold']=fold
    
    #Compile training data from the other folds
    train_folds = []
    for i in range(1, 6):
        if i != fold:  # Do not use the current fold as training data
            df = pd.read_csv(exp / f'Results/{ID}/Sim_Obs/{ID}_val_sim_obs_{i}.csv', decimal=',', sep=';', header=0, parse_dates=['Date'], index_col=0)
            df.columns = df.columns.str.replace(f'_Fold{i}', '')  #Clean up column names
            df['fold']= i
            train_folds.append(df)
    
    cv_train_set = pd.concat(train_folds, axis=0)  #concat all training data

    
    return cv_val_set, cv_train_set


#%%
def load_repOOS(ID, ts, rep, exp):
    #Load validation set for the current repatition
    repOOS_val_set = pd.read_csv(exp / f'Results/{ID}/Sim_Obs/{ID}_val_sim_obs_repHO{rep}.csv', decimal=',', sep=';', header=0, parse_dates=['Date'],index_col=0)
    repOOS_val_set.columns = repOOS_val_set.columns.str.replace(f'_Fold{rep}', '')  #Clean up column names
    val_start = repOOS_val_set.index[0] #start validation data
        
    train_sub= ts[(ts.index < val_start)] #end of train set
    repOOS_train_set = train_sub.iloc[-751:] #complete train set
    
    return repOOS_val_set, repOOS_train_set 

#%%
def load_test_cv(ID, exp):
    #test set for bl-cv loading 
    test_cv = pd.read_csv(exp / f'Results/{ID}/Sim_Obs/{ID}_cv_test_sim_obs.csv', decimal=',', sep=';', header=0, parse_dates=['Date'],index_col=0)
    
    return test_cv

def load_test_oos(ID, exp):   
    #test set for oos loading 
    test_oos = pd.read_csv(exp / f'Results/{ID}/Sim_Obs/{ID}_oos_test_sim_obs.csv', decimal=',', sep=';', header=0, parse_dates=['Date'],index_col=0)
    
    return test_oos

def load_test_rep(ID, exp):
    #test set for repOOS loading 
    test_repOOS = pd.read_csv(exp / f'Results/{ID}/Sim_Obs/{ID}_repHO_test_sim_obs.csv', decimal=',', sep=';', header=0, parse_dates=['Date'],index_col=0)
    
    return test_repOOS  


#%%

#calculate scores for validation data 
def get_scores_val(val, train, ID):
    sim_val = val.iloc[:,1:11]
    obs_val = val.iloc[:,0]
    
    obs_train = train.iloc[:,0]   

       
    sim_array = np.array(sim_val.median(axis=1)) #Median of ensemble members
    obs_array = np.array(obs_val)
    
    err  = sim_array - obs_array
    rmse = np.sqrt(np.mean(err ** 2))
    err_nash = obs_array - np.mean(obs_array)
    nse = 1 - ((np.sum(err ** 2)) / (np.sum((err_nash) ** 2)))

    
    result = pd.DataFrame({
        'RMSE': [rmse],
        'NSE':[nse]
    }, index=[0])
      
    return sim_val, obs_val, obs_train, result

#calculate scores for test data 
def get_scores_test(test, ID):
    sim_test = test.iloc[:,1:11]
    obs_test = test.iloc[:,0]
    

       
    sim_array = np.array(sim_test.median(axis=1)) #Median of ensemble members
    obs_array = np.array(obs_test)
    
    err  = sim_array - obs_array
    rmse = np.sqrt(np.mean(err ** 2))
    err_nash = obs_array - np.mean(obs_array)
    nse = 1 - ((np.sum(err ** 2)) / (np.sum((err_nash) ** 2)))

    
    result = pd.DataFrame({
        'RMSE': [rmse],
        'NSE':[nse]
    }, index=[0])
      
    return sim_test, obs_test, result