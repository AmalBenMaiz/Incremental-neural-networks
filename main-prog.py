# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:52:02 2022

@author: benma
"""
import numpy as np
import time
from I_ELM import *
from make_data import *
from EM_ELM import *
from TLFN import *

 
if __name__ == '__main__':
    
    flag_mo = {'I-ELM' : False, 'EM-ELM' : True, 'TLFN': False}
    
    flag_d = {'zoo': False, 'Iris': False, 'wine':True , 'glass' : False,
               'fetal_health_NSP' : False, 'Hand_gesture_emg': False}
    # ===============================
    # Read data
    # ===============================
    if flag_d['Iris']:
        data_name = 'iris'
        X, Y_True, classes = make_iris()
    if flag_d['Hand_gesture_emg']:
        data_name = 'hand'
        X, Y_True, classes = make_Hand_gesture_emg()
    if flag_d['fetal_health_NSP']:
        data_name = 'fetal_health_NSP'
        X, Y_True, classes = make_fetal_health_NSP()
    if flag_d['glass']:
        data_name = 'glass'
        X, Y_True, classes = make_glass()
    if flag_d['wine']:
        data_name = 'wine'
        X, Y_True, classes = make_wine()
    if flag_d['zoo']:
        data_name = 'zoo'
        X, Y_True, classes = make_zoo()
       
    # ===============================
    # Split data to 80% training and 20% testing
    # ===============================    
    X_train, X_test, Y_train, Y_test= split_data(X, Y_True, 0.2)
    
    # ===============================
    # Define Constants
    # ===============================
    no_classes = len(classes)
    Lmax =6000 # maximum node number 
    L=1 #number of subregions
    quantizer_factor= 25
    scale_factor= 2
    error = 0.01
    loss_function = "mean_squared_error"  #It can be mean_absolute_error also
    activation_function = "sigmoid"
    l_eval = []
    l_test_eval = []
    l_eval_test_out=[]
    
    for sim in range(1):
        if flag_mo['I-ELM']:
            file_name = 'I-ELM'
            model = I_ELM(
                no_input_nodes= X.shape[1],
                max_no_hidden_nodes= Lmax,
                no_output_nodes= 1 ,
                loss_function=loss_function,
                activation_function=activation_function)
            
        if flag_mo['EM-ELM']:
            file_name = 'EM-ELM'
            model = EM_ELM(
                no_input_nodes= X.shape[1],
                max_no_hidden_nodes= Lmax,
                no_output_nodes= 1,
                loss_function=loss_function,
                activation_function=activation_function)
            
        if flag_mo['TLFN']:
            file_name = 'TLFN'
            model = TLFN(
                no_subregions= L, 
                quantizer_T= quantizer_factor, 
                quantizer_U= quantizer_factor,
                c = scale_factor,
                no_input_nodes= X.shape[1],
                max_no_hidden_nodes= Lmax,
                no_output_nodes= 1,
                loss_function=loss_function,
                activation_function=activation_function)
        
        start= time.time()
        eval_df, eval_test_df = model.fit(X_train, Y_train, X_test, Y_test, Lmax,error)
        final = time.time()
        training_time= final-start
        eval_df['Training Time'] = training_time
        eval_df['sim'] = sim
        eval_test_df['sim'] = sim
        l_eval.append(eval_df[['sim','nodes','loss','accuracy', 'Training Time']])
        l_test_eval.append(eval_test_df[['sim','loss','accuracy']])
       
    
    #model.saveMPR()
    eval_df = pd.concat(l_eval)
    eval_test_df = pd.concat(l_test_eval)
    path_eval = f'results/{file_name}/train_{data_name}_{L}subregions_{quantizer_factor}qt.csv' if file_name=='TLFN' else f'results/{file_name}/train_{data_name}.csv'
    eval_df.to_csv(path_eval, index=None)
    path_test = f'results/{file_name}/test_{data_name}_{L}subregions_{quantizer_factor}qt.csv' if file_name=='TLFN' else f'results/{file_name}/test_{data_name}.csv'
    eval_test_df.to_csv(path_test, index=None)
    
    
    
