# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 12:54:18 2025

@author: Fabienne Doll
"""
import numpy as np
import numpy.random 
import tensorflow as tf

import keras as ks
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, GRU
from keras.models import Model

np.random.seed(42)
tf.random.set_seed(42)

def CNN(ini, setup, inp):   
    #seed
    np.random.seed(42+ini)
    tf.random.set_seed(42+ini)
    
    #Input Layer
    inp = Input(shape=(setup['seq_length'], inp.shape[2]), name='input')
                
    #Convolutional Layer 
    CNN = Conv1D(filters=setup["filters"], kernel_size=setup["kernel_size"],
                 padding='same', activation='relu', name='CNN_1')(inp)
    
    #Pooling und Flatten
    Pool = MaxPooling1D(pool_size=2, padding='same', name='max_pool')(CNN)
    Pool = Dropout(0.2, name='dropout_1')(Pool) 
    Flat = Flatten(name='flatten')(Pool)


    #Dense Layer
    dense = Dense(setup["dense_size_cnn"], activation='relu', name='dense_gwl')(Flat)
    dense = Dropout(0.2, name='dropout_2')(dense)

    #Output Layer
    out = Dense(1, activation='linear', name='out_gwl')(dense)
    
#################### Define Model 

    model = Model(inputs=inp, outputs=out)

#################### Optimizers  
     
    optimizer = ks.optimizers.Adam(learning_rate=setup["learning_rate"], epsilon=10E-3, clipnorm=setup["clip_norm"], clipvalue=setup["clip_value"])    
    
#################### Compile    
    model.compile(loss='mse', optimizer=optimizer, metrics=[ks.metrics.MeanAbsoluteError()])  
            
    return model
