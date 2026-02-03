# -*- coding: utf-8 -*-

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

import numpy as np
import numpy.random 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import sys
sys.path.append("./Functions")
import Model as mod
import Preprocess as prepro

np.random.seed(42)
tf.random.set_seed(42)
tf.config.experimental.enable_op_determinism()


#%%
###############################################################################
# CV and OOS 
###############################################################################

def validate_model(Exp, Well_ID, kf, seeds, data, X_train, Y_train, setup):
    
    epochs = pd.DataFrame()
    
    val_sim_gwl = [pd.DataFrame() for _ in range(setup['n_splits'])]
    val_obs_gwl = [pd.DataFrame() for _ in range(setup['n_splits'])]

    for i, (train_ix, val_ix) in enumerate(kf.split(X_train, Y_train)): 
        
        x_train, x_val = X_train[train_ix], X_train[val_ix]           
        y_train, y_val = Y_train[train_ix], Y_train[val_ix]
        
        val_obs_gwl[i]['val_obs_gwl'] = data['GWL'][data['key'].isin(y_val[:, 1])]
        
        # Scale Data
        scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
        
        x_train_n = scaler_x.fit_transform(x_train.reshape(-1, x_train.shape[2])).reshape(x_train.shape)       
        y_train_n = np.hstack((scaler_y.fit_transform(y_train[:, 0].reshape(-1, 1)), y_train[:, 1].reshape(-1, 1)))
          
        x_val_n = scaler_x.transform(x_val.reshape(-1, x_val.shape[2])).reshape(x_val.shape)
        y_val_n = np.hstack((scaler_y.transform(y_val[:, 0].reshape(-1, 1)), y_val[:, 1].reshape(-1, 1)))

        print(f'Fold {i+1} - Train Shape: {x_train_n.shape}, Validation Shape: {x_val_n.shape}')        
                
        #seed loop
        for _, ini in enumerate(seeds):  
            
            prepro.set_global_seed(42 + ini)
            
            if setup['Model']== 'CNN':
                # Load model
                model_val = mod.CNN(ini, setup, x_train_n)
            elif setup['Model']== 'lstm':
                # Load model
                model_val = mod.lstm(ini, setup, x_train_n)
            
            #Early stopping callback
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
            
            #train model
            history_val = model_val.fit(x_train_n, y_train_n[:, 0],
                validation_data=(x_val_n, y_val_n[:, 0]),
                epochs=setup["max_epochs"], batch_size=setup["batch_size"], 
                verbose=2, callbacks=[es])
            
            ###################################################################
            # Visualization of the loss curve
            plt.figure(figsize=(10, 4))
            plt.plot(history_val.history['loss'], label='Train Loss')
            plt.plot(history_val.history['val_loss'], label='Validation Loss', linestyle='dashed')
            plt.title(f'Loss Curve - Fold {i+1}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss [MSE]')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.show()
            ###################################################################
            
            #read epochs 
            epochs.at[i, f'ini_{ini}'] = np.argmin(history_val.history['val_loss']) + 1
            print(f"Best Model saved at epoch {np.argmin(history_val.history['val_loss']) + 1}")
            
            #Predict validation data and rescale to original scale
            val_sim_n = model_val.predict(x_val_n)
            val_sim = scaler_y.inverse_transform(val_sim_n).flatten()

            # save results
            val_sim_gwl[i][f'ini_{ini}_Fold{i+1}'] = val_sim


    # save epochs   
    epochs['epochs'] = epochs.median(axis=1) #median epochs over all seeds  
    epochs.to_csv(f'../{Exp}/Results/{Well_ID}/Train_Test_Stats/{Well_ID}_cv_epochs_all_folds.csv',sep =';' , decimal =',', index=True)

    
    for fold, (v_obs, v_sim) in enumerate(zip(val_obs_gwl, val_sim_gwl)):
        v_sim.reset_index(drop=True, inplace=True)
        df = pd.concat([v_obs.reset_index(drop=False), v_sim], axis=1)
        df.set_index('Date', inplace=True)
        df.to_csv(f'../{Exp}/Results/{Well_ID}/Sim_Obs/{Well_ID}_val_sim_obs_{fold+1}.csv',sep =';' , decimal =',', index=True)  
        
    return epochs['epochs']



#%%
###############################################################################
#Repeated holdout validation using the repHO time periods from the paper run
###############################################################################

def rep_holdout_by_dates(X, Y, data, S_ID,n_reps=5, rep_names=None,start_dates_path="start_dates.csv",end_dates_path="end_dates.csv",train_ratio=0.6):
    # read start and end dates
    start_dates = pd.read_csv(start_dates_path, index_col=0)
    end_dates = pd.read_csv(end_dates_path, index_col=0)
    for col in start_dates.columns:
        start_dates[col] = pd.to_datetime(start_dates[col], errors="coerce")
    for col in end_dates.columns:
        end_dates[col] = pd.to_datetime(end_dates[col], errors="coerce")

    if rep_names is None:
        rep_names = [f'repHO{i}' for i in range(1, n_reps+1)]

    splits = []
    for rep in rep_names:
        start_date = start_dates.loc[int(S_ID), rep]
        end_date = end_dates.loc[int(S_ID), rep]
        if pd.isnull(start_date) or pd.isnull(end_date):
            splits.append(None)
            continue

        # Find Test-Keys in test period [start_date, end_date]
        mask_test = (data.index >= start_date) & (data.index <= end_date)
        keys_test = data.loc[mask_test, 'key'].values

        # Find the keys für the test data in Y-Array 
        keys_Y = Y[:, 1]
        test_indices = np.where(np.isin(keys_Y, keys_test))[0]

        if len(test_indices) == 0:
            splits.append(None)
            continue

        # Training data is the train_ratio (here 60% of the training data) immediately before the test data.
        first_test_idx = test_indices[0]
        train_size = int(train_ratio * len(X))
        train_indices = list(range(first_test_idx - train_size, first_test_idx)) if train_size > 0 else []

        # build ubsets
        X_train_subset = X[train_indices] if train_indices else np.array([])
        Y_train_subset = Y[train_indices] if train_indices else np.array([])
        X_test_subset = X[test_indices]
        Y_test_subset = Y[test_indices]

        splits.append((X_train_subset, Y_train_subset, X_test_subset, Y_test_subset))

    return splits



def rep_HO_validation(Exp, Well_ID, seeds, data, X_train, Y_train, setup):
    
    # reproduce Rep-Holdout of the paper run
    splits = rep_holdout_by_dates(X_train, Y_train, data, Well_ID, n_reps=setup['n_rep'],rep_names=None,
                              start_dates_path="../repHO_dates/start_dates.csv", end_dates_path="../repHO_dates/end_dates.csv", train_ratio=0.6)
    
    epochs = pd.DataFrame()
    
    val_sim_gwl = [pd.DataFrame() for _ in range(setup['n_rep'])]
    val_obs_gwl = [pd.DataFrame() for _ in range(setup['n_rep'])]
    
    #sample split
    for i, (x_train, y_train, x_val, y_val) in enumerate(splits):
        print(f"Iteration {i+1}:")
        print(f"Train shape: {x_train.shape}, Test shape: {x_val.shape}")
        print()
        
        val_obs_gwl[i]['val_obs_gwl'] = data['GWL'][data['key'].isin(y_val[:, 1])]
        
        # Scale Data
        scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
        
        x_train_n = scaler_x.fit_transform(x_train.reshape(-1, x_train.shape[2])).reshape(x_train.shape)
        y_train_n = np.hstack((scaler_y.fit_transform(y_train[:, 0].reshape(-1, 1)), y_train[:, 1].reshape(-1, 1)))
        
        x_val_n = scaler_x.transform(x_val.reshape(-1, x_val.shape[2])).reshape(x_val.shape)
        y_val_n = np.hstack((scaler_y.transform(y_val[:, 0].reshape(-1, 1)), y_val[:, 1].reshape(-1, 1)))
        
        #seed loop
        for _, ini in enumerate(seeds):  
            
            prepro.set_global_seed(42 + ini)

            if setup['Model']== 'CNN':
                # Load model
                model_val = mod.CNN(ini, setup, x_train_n)
            elif setup['Model']== 'lstm':
                # Load model
                model_val = mod.lstm(ini, setup, x_train_n)
            
            #Early stopping callback
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
            
            #train model
            history_val = model_val.fit(x_train_n, y_train_n[:, 0],
                validation_data=(x_val_n, y_val_n[:, 0]),
                epochs=setup["max_epochs"], batch_size=setup["batch_size"], 
                verbose=2, callbacks=[es])
            
            ###################################################################
            # Visualization of the loss curve
            plt.figure(figsize=(10, 4))
            plt.plot(history_val.history['loss'], label='Train Loss')
            plt.plot(history_val.history['val_loss'], label='Validation Loss', linestyle='dashed')
            plt.title(f'Loss Curve - Fold {i+1}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss [MSE]')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.show()
            ###################################################################
            
            #save epochs
            epochs.at[i, f'ini_{ini}'] = np.argmin(history_val.history['val_loss']) + 1
            print(f"Best Model saved at epoch {np.argmin(history_val.history['val_loss']) + 1}")
            
            #Predict validation data and rescale to original scale 
            val_sim_n = model_val.predict(x_val_n)
            val_sim = scaler_y.inverse_transform(val_sim_n).flatten()

            # save results
            val_sim_gwl[i][f'ini_{ini}_Fold{i+1}'] = val_sim


    # save results  
    epochs['epochs'] = epochs.median(axis=1)  #median epochs over all seeds 
    epochs.to_csv(f'../{Exp}/Results/{Well_ID}/Train_Test_Stats/{Well_ID}_repHO_epochs_all_folds.csv',sep =';' , decimal =',', index=True)

    
    for it, (v_obs, v_sim) in enumerate(zip(val_obs_gwl, val_sim_gwl)):
        v_sim.reset_index(drop=True, inplace=True)
        df = pd.concat([v_obs.reset_index(drop=False), v_sim], axis=1)
        df.set_index('Date', inplace=True)
        df.to_csv(f'../{Exp}/Results/{Well_ID}/Sim_Obs/{Well_ID}_val_sim_obs_repHO{it+1}.csv',sep =';' , decimal =',', index=True)  
            
    return epochs['epochs']

#%%
###############################################################################
#Test run
###############################################################################

def test(Exp, Well_ID, train_epochs, setup, seeds, data, X_train_n, Y_train_n, X_test_n, Y_test, scaler_Y):
        
    test_sim =  pd.DataFrame() 
    
    #seed loop       
    for _, ini in enumerate(seeds):  
            
        prepro.set_global_seed(42 + ini)  
                         
        if setup['Model']== 'CNN':
            # Load model
            model_test = mod.CNN(ini, setup, X_train_n)
        elif setup['Model']== 'lstm':
            # Load model
            model_test = mod.lstm(ini, setup, X_train_n)
          
        # Train model
        model_test.fit(X_train_n, Y_train_n[:, 0],
                       epochs=train_epochs,
                       verbose=2,
                       batch_size= setup["batch_size"]
                       )
        
        #predict test data and rescale to original scale                    
        sim_n = model_test.predict(X_test_n)
        sim = scaler_Y.inverse_transform(sim_n).flatten()
                
        # Store test sims 
        test_sim[f'ini_{ini}'] = sim.flatten()
                               
        
    test_obs  = data['GWL'][data['key'].isin(Y_test[:, 1])]#first column = observations of the testdata
  
    test_sim.reset_index(drop=True, inplace=True)
    obs_sim = pd.concat([test_obs.reset_index(drop=False), test_sim], axis=1)
    obs_sim.set_index('Date', inplace=True)                                      

    return obs_sim




#%%
###############################################################################
#Repeated holdout validation using renewed random periods (results may differ from the paper)
###############################################################################

# def rep_holdout(X_train, Y_train, n_reps=5, train_ratio=0.6, test_ratio=0.1):
#     t = len(X_train)
#     train_size = int(train_ratio * t)
#     test_size = int(test_ratio * t)
#     splits = []
    
#     for _ in range(n_reps):
#         # Pick a random point 'a' ensuring space for training and testing
#         a = np.random.randint(train_size, t - test_size)
        
#         train_indices = list(range(a - train_size, a))
#         test_indices = list(range(a, a + test_size))
        
#         X_train_subset = X_train[train_indices]
#         Y_train_subset = Y_train[train_indices]
#         X_test_subset = X_train[test_indices]
#         Y_test_subset = Y_train[test_indices]
        
#         splits.append((X_train_subset, Y_train_subset, X_test_subset, Y_test_subset))
    
#     return splits

# def rep_HO_validation(Exp, Well_ID, seeds, data, X_train, Y_train, setup):
    
#     # perform Rep-Holdout 
#     splits = rep_holdout(X_train, Y_train, n_reps=setup['n_rep'])
    
#     epochs = pd.DataFrame()
    
#     val_sim_gwl = [pd.DataFrame() for _ in range(setup['n_rep'])]
#     val_obs_gwl = [pd.DataFrame() for _ in range(setup['n_rep'])]
    
#     # Displaying a sample split
#     for i, (x_train, y_train, x_val, y_val) in enumerate(splits):
#         print(f"Iteration {i+1}:")
#         print(f"Train shape: {x_train.shape}, Test shape: {x_val.shape}")
#         print()

#         val_obs_gwl[i]['val_obs_gwl'] = data['GWL'][data['key'].isin(y_val[:, 1])]
        
#         # Scale Data
#         scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
#         x_train_n = scaler_x.fit_transform(x_train.reshape(-1, x_train.shape[2])).reshape(x_train.shape)
#         y_train_n = np.hstack((scaler_y.fit_transform(y_train[:, 0].reshape(-1, 1)), y_train[:, 1].reshape(-1, 1)))
#         x_val_n = scaler_x.transform(x_val.reshape(-1, x_val.shape[2])).reshape(x_val.shape)
#         y_val_n = np.hstack((scaler_y.transform(y_val[:, 0].reshape(-1, 1)), y_val[:, 1].reshape(-1, 1)))
        
#         for _, ini in enumerate(seeds):  
            
#             prepro.set_global_seed(42 + ini)

#             if setup['Model']== 'CNN':
#                 # Load model
#                 model_val = mod.CNN(ini, setup, x_train_n)
#             elif setup['Model']== 'lstm':
#                 # Load model
#                 model_val = mod.lstm(ini, setup, x_train_n)
            
#             #Early stopping callback
#             es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
            
#             #train model
#             history_val = model_val.fit(x_train_n, y_train_n[:, 0],
#                 validation_data=(x_val_n, y_val_n[:, 0]),
#                 epochs=setup["max_epochs"], batch_size=setup["batch_size"], 
#                 verbose=2, callbacks=[es])
            
#             ###################################################################
#             # Visualization of the loss curve
#             plt.figure(figsize=(10, 4))
#             plt.plot(history_val.history['loss'], label='Train Loss')
#             plt.plot(history_val.history['val_loss'], label='Validation Loss', linestyle='dashed')
#             plt.title(f'Loss Curve - Fold {i+1}')
#             plt.xlabel('Epochs')
#             plt.ylabel('Loss [MSE]')
#             plt.legend()
#             plt.grid(True, linestyle='--', alpha=0.5)
#             plt.show()
#             ###################################################################
            
#             epochs.at[i, f'ini_{ini}'] = np.argmin(history_val.history['val_loss']) + 1
#             print(f"Best Model saved at epoch {np.argmin(history_val.history['val_loss']) + 1}")
            
#             #Predict validation data 
#             val_sim_n = model_val.predict(x_val_n)
#             val_sim = scaler_y.inverse_transform(val_sim_n).flatten()

#             # save results
#             val_sim_gwl[i][f'ini_{ini}_Fold{i+1}'] = val_sim

#     # save results  
#     epochs['epochs'] = epochs.median(axis=1)   
#     epochs.to_csv(f'../{Exp}/Results/{Well_ID}/Train_Test_Stats/{Well_ID}_repHO_epochs_all_folds.csv',sep =';' , decimal =',', index=True)

    
#     for it, (v_obs, v_sim) in enumerate(zip(val_obs_gwl, val_sim_gwl)):
#         v_sim.reset_index(drop=True, inplace=True)
#         df = pd.concat([v_obs.reset_index(drop=False), v_sim], axis=1)
#         df.set_index('Date', inplace=True)
#         df.to_csv(f'../{Exp}/Results/{Well_ID}/Sim_Obs/{Well_ID}_val_sim_obs_repHO{it+1}.csv',sep =';' , decimal =',', index=True)  
            
#     return epochs['epochs']
