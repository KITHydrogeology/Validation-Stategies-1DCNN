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

exp = Path('../CNN_Compare_Val_2025-07-25_11-19-19')

sid = pd.read_csv('../ids_wells.csv', index_col=0, sep=';')


# Path for the “TS_Plots” subfolder
path = os.path.join(exp, "TS_Plots")

# Create “TS_Plots” subfolder if it does not exist
if not os.path.exists(path):
    os.makedirs(path)
    print(f"Subfolder created: {path}")
else:
    print(f"Subfolder already exists: {path}")


#%%


all_scores =[]    

for i in range(4):
    ID = res_proces.load_ID(i, exp)
    
    ts = pd.read_csv(f'../Data/{ID}.csv', decimal='.', sep=',', header=0, parse_dates=['Date'], index_col=0)
    in_set = ts[ts.index<pd.to_datetime('29122014', format='%d%m%Y')] # get in-set
    out_set = ts[ts.index>pd.to_datetime('29122014', format='%d%m%Y')] # get in-set
    ########################################################## Validation
    
    ############# CV and OOS
    
    sim_cv=[]
    results_folds =[]
    
    #Folds 1-5
    for fold in range(1,6):
        
        cv_val_set, cv_train_set =  res_proces.cv_fold_obs_sim(ID, fold, exp)
        sim_val_cv, obs_val_cv, obs_train_cv, result_cv = res_proces.get_scores_val(cv_val_set, cv_train_set, ID)              
        results_folds.append(result_cv)
        
        sim_val_cv['fold']=fold
        sim_cv.append(sim_val_cv)
    
    sim_blCV = pd.concat(sim_cv, axis=0)
    sim_OOS = sim_cv[4]
    
    scores_cv = pd.concat(results_folds, axis=0, ignore_index=True)
    scores_cv.to_csv(exp/ f'Results/{ID}/cv_fold_scores.csv', sep=';', decimal=',')    
    
    scores_cv_mean = pd.DataFrame(scores_cv.mean(axis=0)).T #Mean
    scores_oos = scores_cv[4:].reset_index(drop=True) #OOS (Fold 5)
    
    scores_cv_mean = scores_cv_mean.add_suffix("_cv")
    scores_oos = scores_oos.add_suffix("_oos")
    
    ############# repOOS
    sim_rep=[]
    results_rep = []
    #repetition 1-5
    for rep in range (1,6):
        
        repOOS_val_set, repOOS_train_set = res_proces.load_repOOS(ID, ts, rep, exp)
        sim_val_repOOS, obs_val_repOOS, obs_train_repOOS, result_repOOS = res_proces.get_scores_val(repOOS_val_set, repOOS_train_set, ID)  
        results_rep.append(result_repOOS)
        
        sim_val_repOOS['rep']=rep
        sim_rep.append(sim_val_repOOS)
    
    sim_repOOS = pd.concat(sim_rep, axis=0)
    
    scores_repOOS = pd.concat(results_rep, axis=0, ignore_index=True)
    scores_repOOS.to_csv(exp/ f'Results/{ID}/repOOS_scores.csv', sep=';', decimal=',')  
    
    scores_repOOS_mean = pd.DataFrame(scores_repOOS.mean(axis=0)).T #Mean
    scores_repOOS_mean = scores_repOOS_mean.add_suffix("_repOOS")
#     ########################################################## Test

           
    test_cv = res_proces.load_test_cv(ID, exp) 
    sim_test_cv, obs_test_cv, scores_test_cv = res_proces.get_scores_test(test_cv, ID)
    
    test_oos = res_proces.load_test_oos(ID, exp) 
    sim_test_oos, obs_test_oos, scores_test_oos = res_proces.get_scores_test(test_oos, ID)
    
    test_repOOS = res_proces.load_test_rep(ID, exp) 
    sim_test_repOOS, obs_test_repOOS, scores_test_repOOS = res_proces.get_scores_test(test_repOOS, ID)
    
    

    scores_test_cv = scores_test_cv.add_suffix("_test_cv")
    scores_test_oos = scores_test_oos.add_suffix("_test_oos")
    scores_test_repOOS = scores_test_repOOS.add_suffix("_test_repOOS")

    
    scores_test_cv.to_csv( exp/ f'Results/{ID}/test_cv_scores.csv', sep=';', decimal=',')       
    scores_test_oos.to_csv(exp/ f'Results/{ID}/test_oos_scores.csv', sep=';', decimal=',')
    scores_test_repOOS.to_csv( exp/ f'Results/{ID}/test_repOOS_scores.csv', sep=';', decimal=',')
            
    all_err = pd.concat([scores_cv_mean, scores_repOOS_mean, scores_oos, scores_test_cv, scores_test_repOOS, scores_test_oos], axis=1)
    all_err.index = [ID] 
    
    all_scores.append(all_err)


    #%%
    # Farben für die Test-Daten          
    median_cv_color = "#034e7b" 
    median_oos_color = "#005a32"
    median_rep_color = "#4a1486"

    
    fold_colors = {
        1: '#7fcdbb',  
        2: '#a6bddb',  
        3: '#74a9cf',  
        4: '#3690c0',
        5: '#0570b0'   
    }
    
    rep_colors = {
        1: '#9ebcda',  
        2: '#8c96c6',  
        3: '#8c6bb1',  
        4: '#88419d',
        5: '#810f7c'   
    }
    

    fig, axes = plt.subplots(3, 2, figsize=(13, 8), sharex=False, sharey=True, gridspec_kw={'width_ratios': [0.8, 0.2]})
    
    fig.suptitle(f"Validation and Test Results {ID}", fontsize=16, fontweight='bold')
    
    # Plot 1: Validation & Test (bl-CV)
    ax1, ax2 = axes[0]
    ax1.plot(in_set.index, in_set.iloc[:,0], color='k', label='Observed', alpha=1, linewidth=1)
    for fold, group in sim_blCV.groupby('fold'):
        median_values = group.iloc[:, 1:11].median(axis=1)        
        ax1.plot(group.index, median_values, color=fold_colors[fold], linestyle='--', linewidth=3)
    ax1.set_title(f"In-set (bl-CV): RMSE = {all_err['RMSE_cv'].iloc[0]:.2f}")
    ax1.set_xlabel("Year")
    ax1.set_xticks(in_set.index)
    
    ax2.plot(out_set.index, out_set.iloc[:,0], color='black', linestyle='-', linewidth=1)
    ax2.plot(sim_test_cv.index, sim_test_cv.iloc[:, 1:11].median(axis=1), color=median_cv_color, linestyle='--', linewidth=3)
    ax2.set_title(f"Out-set (bl-CV): RMSE = {all_err['RMSE_test_cv'].iloc[0]:.2f}")
    ax2.set_xlabel("Year")
    ax2.set_xticks(out_set.index)
    
    # Plot 2: Validation & Test (OOS)
    ax3, ax4 = axes[1]
    ax3.plot(in_set.index, in_set.iloc[:,0], color='k', linewidth=1)
    median_values = sim_OOS.iloc[:,1:11].median(axis=1)
    ax3.plot(sim_OOS.index, median_values, color='#78c679', linestyle='--', linewidth=3)
    ax3.set_title(f"In-set (OOS): RMSE = {all_err['RMSE_oos'].iloc[0]:.2f}")
    ax3.set_xlabel("Year")
    ax3.set_xticks(in_set.index)
    
    ax4.plot(out_set.index, out_set.iloc[:,0], color='black', linestyle='-', linewidth=1)
    ax4.plot(sim_test_oos.index, sim_test_oos.iloc[:, 1:11].median(axis=1), color=median_oos_color, linestyle='--', linewidth=3)
    ax4.set_title(f"Out-set (OOS): RMSE = {all_err['RMSE_test_oos'].iloc[0]:.2f}")
    ax4.set_xlabel("Year")
    ax4.set_xticks(out_set.index)
    
    # Plot 3: Validation & Test (repOOS)
    ax5, ax6 = axes[2]
    ax5.plot(in_set.index, in_set.iloc[:,0], color='k', linewidth=1)
    for rep, group in sim_repOOS.groupby('rep'):
        median_values = group.iloc[:, 1:11].median(axis=1)        
        ax5.plot(group.index, median_values, color=rep_colors[rep], linestyle='--', linewidth=3)
    ax5.set_title(f"In-set (repOOS): RMSE = {all_err['RMSE_repOOS'].iloc[0]:.2f}")
    ax5.set_xlabel("Year")
    ax5.set_xticks(in_set.index)
    
    ax6.plot(out_set.index, out_set.iloc[:,0], color='black', linestyle='-', linewidth=1)
    ax6.plot(sim_test_repOOS.index, sim_test_repOOS.iloc[:, 1:11].median(axis=1), color=median_rep_color, linestyle='--', linewidth=3)
    ax6.set_title(f"Out-set (repOOS): RMSE = {all_err['RMSE_test_repOOS'].iloc[0]:.2f}")
    ax6.set_xlabel("Year")
    ax6.set_xticks(out_set.index)
    
    # axis formating 
    for ax in axes.flat:
        ax.grid(axis='y', linestyle='--', linewidth=0.5, color='grey')
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    axes[1, 0].set_ylabel("GWL [m a.s.l]")
    
    plt.tight_layout()
    plt.savefig(exp/f'TS_Plots/{ID}_result_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
df = pd.concat(all_scores, axis=0)
df = df.round(2)

df['APAE_cv'] = abs(df['RMSE_cv']-df['RMSE_test_cv'])
df['APAE_repOOS'] = abs(df['RMSE_repOOS']-df['RMSE_test_repOOS']) 
df['APAE_oos'] = abs(df['RMSE_oos']-df['RMSE_test_oos']) 

df['PAE_cv'] = df['RMSE_cv']-df['RMSE_test_cv']
df['PAE_repOOS'] = df['RMSE_repOOS']-df['RMSE_test_repOOS']  
df['PAE_oos'] = df['RMSE_oos']-df['RMSE_test_oos'] 

df.to_csv(exp / 'all_scores.csv', sep=';', decimal=',')           
        

