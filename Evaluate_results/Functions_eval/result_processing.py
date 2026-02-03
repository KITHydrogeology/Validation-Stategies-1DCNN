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


#%%load cv and oos predictions

def cv_fold_obs_sim(ID, fold, exp):
    # Load validation set for the current fold
    cv_val_set = pd.read_csv(exp / f'Results/{ID}/Sim_Obs/{ID}_val_sim_obs_{fold}.csv', decimal=',', sep=';', header=0, parse_dates=True,index_col=0)
    cv_val_set.columns = cv_val_set.columns.str.replace(f'_Fold{fold}', '')    #Clean up column names
    cv_val_set['fold']=fold
       
    return cv_val_set


#%%load repHO predictions

def load_repHO(ID,rep, exp):
    #Load validation set for the current repatition
    repOOS_val_set = pd.read_csv(exp / f'Results/{ID}/Sim_Obs/{ID}_val_sim_obs_repHO{rep}.csv', decimal=',', sep=';', header=0, parse_dates=True,index_col=0)
    repOOS_val_set.columns = repOOS_val_set.columns.str.replace(f'_Fold{rep}', '')  #Clean up column names
    
    return repOOS_val_set


#%%load test predictions

def load_test_cv(ID, exp):
    #test set for bl-cv loading 
    test_cv = pd.read_csv(exp / f'Results/{ID}/Sim_Obs/{ID}_cv_test_sim_obs.csv', decimal=',', sep=';', header=0, parse_dates=True,index_col=0)
    
    return test_cv

def load_test_oos(ID, exp):   
    #test set for oos loading 
    test_oos = pd.read_csv(exp / f'Results/{ID}/Sim_Obs/{ID}_oos_test_sim_obs.csv', decimal=',', sep=';', header=0, parse_dates=True,index_col=0)
    
    return test_oos

def load_test_rep(ID, exp):
    #test set for repOOS loading 
    test_repOOS = pd.read_csv(exp / f'Results/{ID}/Sim_Obs/{ID}_repHO_test_sim_obs.csv', decimal=',', sep=';', header=0, parse_dates=True,index_col=0)
    
    return test_repOOS  


#%%
#calculate scores 
def get_scores(df, ID):
    sim = df.iloc[:,1:11]#predictions col 1-10
    obs = df.iloc[:,0]#first column = observation
         
    sim_array = np.array(sim.median(axis=1)) #final fold simulation = Median of ensemble members
    obs_array = np.array(obs)
    
    err  = sim_array - obs_array
    rmse = np.sqrt(np.mean(err ** 2))
    
    result = pd.DataFrame({
        'RMSE': [rmse]
    }, index=[0])
      
    return sim, obs, result
