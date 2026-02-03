# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:24:13 2025

@author: Fabienne Doll
"""


import glob
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from pathlib import Path
import importlib
import sys
import os

sys.path.append("./Functions_eval")
import result_processing as res_proces

#%%
experiment = 'CNN'

exp = Path(f'../Results_1DCNN_LSTM/{experiment}')

sid = pd.read_csv('../ids_wells.csv', index_col=0, sep=';')

#%%

all_scores =[]    

for i in range(100):
    ID = res_proces.load_ID(i, exp)
       
    ############# Scores CV and OOS    
    sim_cv=[]
    results_folds =[]
    
    #Folds 1-5
    for fold in range(1,6):
        
        cv_val_set =  res_proces.cv_fold_obs_sim(ID, fold, exp)
        sim_val_cv, obs_val_cv, result_cv = res_proces.get_scores(cv_val_set, ID)              
        results_folds.append(result_cv)
        
        sim_val_cv['fold']=fold
        sim_cv.append(sim_val_cv)
    
    sim_blCV = pd.concat(sim_cv, axis=0)
    sim_OOS = sim_cv[4]
    
    scores_cv = pd.concat(results_folds, axis=0, ignore_index=True)
    scores_cv.to_csv(exp/ f'Results/{ID}/cv_fold_scores.csv', sep=';', decimal=',')    
    
    scores_cv_mean = pd.DataFrame(scores_cv.mean(axis=0)).T #Mean score (RMSE) of all folds
    scores_oos = scores_cv[4:].reset_index(drop=True) #RMSE OOS (Fold 5)
    
    scores_cv_mean = scores_cv_mean.add_suffix("_cv")#add cv as suffix to score
    scores_oos = scores_oos.add_suffix("_oos")# add oos as suffix to score
    
    
    
    ############# Scores repHO
    sim_rep=[]
    results_rep = []
    
    #repetition 1-5
    for rep in range (1,6):
        
        repOOS_val_set = res_proces.load_repHO(ID, rep, exp)
        sim_val_repOOS, obs_val_repOOS, result_repOOS = res_proces.get_scores(repOOS_val_set, ID)  
        results_rep.append(result_repOOS)
        
        sim_val_repOOS['rep']=rep
        sim_rep.append(sim_val_repOOS)
    
    sim_repOOS = pd.concat(sim_rep, axis=0)
    
    scores_repOOS = pd.concat(results_rep, axis=0, ignore_index=True)
    scores_repOOS.to_csv(exp/ f'Results/{ID}/repOOS_scores.csv', sep=';', decimal=',')  
    
    scores_repOOS_mean = pd.DataFrame(scores_repOOS.mean(axis=0)).T #Mean score (RMSE) of all repetitions 
    scores_repOOS_mean = scores_repOOS_mean.add_suffix("_repOOS")# ad repOO as suffix
    
    
    
    ###########Scores Test
           
    test_cv = res_proces.load_test_cv(ID, exp) 
    sim_test_cv, obs_test_cv, scores_test_cv = res_proces.get_scores(test_cv, ID)
    
    test_oos = res_proces.load_test_oos(ID, exp) 
    sim_test_oos, obs_test_oos, scores_test_oos = res_proces.get_scores(test_oos, ID)
    
    test_repOOS = res_proces.load_test_rep(ID, exp) 
    sim_test_repOOS, obs_test_repOOS, scores_test_repOOS = res_proces.get_scores(test_repOOS, ID)
    
    #add suffix
    scores_test_cv = scores_test_cv.add_suffix("_test_cv")
    scores_test_oos = scores_test_oos.add_suffix("_test_oos")
    scores_test_repOOS = scores_test_repOOS.add_suffix("_test_repOOS")

    
    scores_test_cv.to_csv( exp/ f'Results/{ID}/test_cv_scores.csv', sep=';', decimal=',')       
    scores_test_oos.to_csv(exp/ f'Results/{ID}/test_oos_scores.csv', sep=';', decimal=',')
    scores_test_repOOS.to_csv( exp/ f'Results/{ID}/test_repOOS_scores.csv', sep=';', decimal=',')
            
    all_err = pd.concat([scores_cv_mean, scores_repOOS_mean, scores_oos, scores_test_cv, scores_test_repOOS, scores_test_oos], axis=1)
    all_err.index = [ID] 
    
    all_scores.append(all_err)


#bring all scores together    
df = pd.concat(all_scores, axis=0)
df.index = pd.to_numeric(df.index)
df = df.round(2)

#calculation of APAE and PAE
df['APAE_cv'] = abs(df['RMSE_cv']-df['RMSE_test_cv'])
df['APAE_repOOS'] = abs(df['RMSE_repOOS']-df['RMSE_test_repOOS']) 
df['APAE_oos'] = abs(df['RMSE_oos']-df['RMSE_test_oos']) 

df['PAE_cv'] = df['RMSE_cv']-df['RMSE_test_cv']
df['PAE_repOOS'] = df['RMSE_repOOS']-df['RMSE_test_repOOS']  
df['PAE_oos'] = df['RMSE_oos']-df['RMSE_test_oos'] 

#save all scores
df.to_csv(exp / f'all_scores_{experiment}.csv', sep=';', decimal=',')   

#scores for stationary timeseries  
stat = df[(df.index >= 1) & (df.index <= 50)] 
stat.to_csv(exp/f'stat_scores_{experiment}.csv', sep=';', decimal=',')

#scores for non-stationary timeseries 
nonstat = df[(df.index > 50) & (df.index <= 100)]
nonstat.to_csv(exp/f'nonstat_scores_{experiment}.csv', sep=';', decimal=',')





