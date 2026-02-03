# -*- coding: utf-8 -*-


import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))


import numpy as np
import numpy.random 
import pandas as pd
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold


import sys
sys.path.append("./Functions")
import Preprocess as prepro
import Val_Test_functions as val_test


np.random.seed(42)
tf.random.set_seed(42)
tf.config.experimental.enable_op_determinism()

#%%
###############################################################################
# Experiment Settings and Result Folder 
###############################################################################
seq_length = 52

setup = { 
    'Experiment': 'Compare_Val',
    'Model': 'CNN',
    'test_size':0.2,
    'shuffle_TS': False,
    
    'CV': 'kFold',
    'n_splits': 5,
    'shuffle_CV': False,
    
    'n_rep':5,
    
    ## Global Model setup
    'batch_size': 16,            
    'max_epochs': 300, 
    'learning_rate': 1e-4,        
    'clip_norm': True,
    'clip_value': 1,             
    'seq_length': seq_length,               
    'start':pd.to_datetime('01011990', format='%d%m%Y'),
    'end': pd.to_datetime('31122020', format='%d%m%Y'), 
  
    ## CNN
    'filters': 128, 
    'kernel_size': 3,           
    'dense_size': 16,
    
    #LSTM
    'lstm_units': 128
    }
        
time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

Exp = str(setup['Model'])+'_'+str(setup['Experiment'])+f'_{time}'

############################# Folder ##########################################
# Base path for all experiments
base_path = "../"
# Path for this specific experiment
experiment_path = os.path.join(base_path, Exp)

# Create folder if it does not exist
if not os.path.exists(experiment_path):
    os.makedirs(experiment_path)
    print(f"Subfolder created: {experiment_path}")
else:
    print(f"Subfolder already exists: {experiment_path}")
    
# Path for the “Results” subfolder
results_path = os.path.join(experiment_path, "Results")

# Create “Results” subfolder if it does not exist
if not os.path.exists(results_path):
    os.makedirs(results_path)
    print(f"Subfolder created: {results_path}")
else:
    print(f"Subfolder already exists: {results_path}")
    
###############################################################################
# Start Experiment  with GPU
###############################################################################
with tf.device("/gpu:0"): 

   
    for well in range(100):
            
        # loading data
        Data, Well_ID = prepro.load_weekly_Data(well)
        
        #Select the data period
        data = Data[(Data.index >= setup['start']) & (Data.index < setup['end'])]
        # set index key
        data.insert(1,'key', range(len(data)))
             
        #make sequences
        seq_features, seq_targets = prepro.make_sequence(np.asarray(data), setup)
        
        #######################################################################
        #Subfolder for this Well ID
        #######################################################################
        
        # Subfolder name
        subfolders = [f"{Well_ID}"]
        # create subfolder 
        for folder in subfolders:
            folder_path = os.path.join(results_path, folder)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"Subfolder created: {folder_path}")
            else:
                print(f"Subfolder already exists: {folder_path}")
                        
        # Path for the “Results” subfolder
        id_path = os.path.join(results_path, f"{Well_ID}")
                                        
        #Names of subfolders
        subfolders = ["Train_Test_Stats", "Sim_Obs"]
        
        # Creating subfolders in “Results”
        for folder in subfolders:
            folder_path = os.path.join(id_path, folder)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"Subfolder created: {folder_path}")
            else:
                print(f"Subfolder already exists: {folder_path}")
        
        ####################################################################### 
        # Train/Test Split 
        #######################################################################
                
        X_train, X_test, Y_train, Y_test = train_test_split(seq_features, seq_targets, test_size=setup['test_size'], random_state=2, shuffle=setup['shuffle_TS'])
        
        
        ################################ Scale Main Train/Test Split #################################
       
        # Define the scalers for X and Y sequences
        scaler_X = MinMaxScaler()
        scaler_Y = MinMaxScaler()

        # Scaling the training data
        X_train_shape = X_train.shape
        X_train_n = scaler_X.fit_transform(X_train.reshape(-1, X_train_shape[2])).reshape(X_train_shape)
        
        # Scale the first column of Y_train and combine the unchanged second column (key)
        Y_train_n_scaled = scaler_Y.fit_transform(Y_train[:, 0].reshape(-1, 1))
        Y_train_n = np.hstack((Y_train_n_scaled, Y_train[:, 1].reshape(-1, 1))) 
        
        # Scaling test data
        X_test_shape = X_test.shape
        X_test_n = scaler_X.transform(X_test.reshape(-1, X_test_shape[2])).reshape(X_test_shape)
        
        # Scale the first column of Y_test and combine the unchanged second column (key)
        Y_test_n_scaled = scaler_Y.transform(Y_test[:, 0].reshape(-1, 1))
        Y_test_n = np.hstack((Y_test_n_scaled, Y_test[:, 1].reshape(-1, 1)))  
        
               
        #######################################################################
        #Define CV
        #######################################################################
                
        kf = KFold(setup['n_splits'], shuffle=setup['shuffle_CV'])
                      
        #######################################################################
        
        #Define seeds for ensemble predictions 
        seeds = [1,52,123,572,1254,2457,5321,10479,16284,20932]#
        
        
        #%% Validation 
        
        ###CV and HO Validation        
        results_cv = val_test.validate_model(Exp, Well_ID, kf, seeds, data, X_train, Y_train, setup)        
        train_epochs_cv = int(results_cv.mean(axis=0))#mean train epochs over all folds
        train_epochs_oos = int(results_cv.iloc[4])# train epochs of fold5 (oos fold)        
        results_cv.name = Well_ID  
        
        
        ###RepHO
        results_repHO= val_test.rep_HO_validation(Exp, Well_ID, seeds, data, X_train, Y_train, setup) 
        train_epochs_repHO = int(results_repHO.mean(axis=0))#mean train epochs over all reps
        results_repHO.name = Well_ID  
        
        
        #%% Testing using the determined validation epochs
        
        ####### CV Test    
        cv_test_obs_sim = val_test.test(Exp, Well_ID, train_epochs_cv, setup, seeds, data, X_train_n, Y_train_n, X_test_n, Y_test, scaler_Y)
        cv_test_obs_sim.to_csv(f'../{Exp}/Results/{Well_ID}/Sim_Obs/{Well_ID}_cv_test_sim_obs.csv', sep =';' , decimal =',',index=True)    
    
        
        ####### OOS Test    
        oos_test_obs_sim = val_test.test(Exp, Well_ID, train_epochs_oos, setup, seeds, data, X_train_n, Y_train_n, X_test_n, Y_test, scaler_Y)
        oos_test_obs_sim.to_csv(f'../{Exp}/Results/{Well_ID}/Sim_Obs/{Well_ID}_oos_test_sim_obs.csv',sep =';' , decimal =',', index=True)    
              

        ####### rep_HO Test    
        repHO_test_obs_sim = val_test.test(Exp, Well_ID, train_epochs_repHO, setup, seeds, data, X_train_n, Y_train_n, X_test_n, Y_test, scaler_Y)
        repHO_test_obs_sim.to_csv(f'../{Exp}/Results/{Well_ID}/Sim_Obs/{Well_ID}_repHO_test_sim_obs.csv',sep =';' , decimal =',', index=True)  

        
        
        