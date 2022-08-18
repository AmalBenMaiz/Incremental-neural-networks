# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:59:06 2022

@author: benma
"""

import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

def encode_data(v):
    # encode class values as integers between 0 and n_class-1
    encoder = LabelEncoder()
    encoder.fit(v)
    classes= list(encoder.classes_)
    encoded_y = encoder.transform(v) #convert string to integers variables
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_y)
    return classes,encoded_y, dummy_y

def split_data(x, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return X_train, X_test, y_train, y_test

def make_iris():
    # ===============================
    # Load dataset 
    # ===============================
    dataset= pd.read_csv("data/iris.csv", sep=';')
    
    # ===============================
    #separate data from labels 
    # ===============================   
    X = np.asarray(dataset.iloc[:,1:].values) #input values(features)
    y = np.asarray(dataset.iloc[:,0].values) #target values (labels)
    
    # ===============================
    #encode data
    # =============================== 
    res = encode_data(y)
    classes =res[0]  #liste des classes
    Y_True= res[2]
    return X, Y_True,classes 

def make_Hand_gesture_emg():
    # ===============================
    # Load dataset 
    # ===============================
    dataset= pd.read_csv("data/Hand_gesture_emg.csv", sep=';')
    
    # ===============================
    #separate data from labels 
    # ===============================   
    X = np.asarray(dataset.iloc[:,1:].values) #input values(features)
    y = np.asarray(dataset.iloc[:,0].values) #target values (labels)
    
    # ===============================
    #encode data
    # =============================== 
    res = encode_data(y)
    classes =res[0]  #liste des classes
    Y_True= res[2]
    return X, Y_True,classes 

def make_fetal_health_NSP():
    # ===============================
    # Load dataset 
    # ===============================
    dataset= pd.read_csv("data/fetal_health_NSP.csv", sep=';')
    
    # ===============================
    #separate data from labels 
    # ===============================   
    X = np.asarray(dataset.iloc[:,1:].values) #input values(features)
    y = np.asarray(dataset.iloc[:,0].values) #target values (labels)
    
    # ===============================
    #encode data
    # =============================== 
    res = encode_data(y)
    classes =res[0]  #liste des classes
    Y_True= res[2]
    return X, Y_True,classes 
def make_glass():
    # ===============================
    # Load dataset 
    # ===============================
    dataset= pd.read_csv("data/glass.csv", sep=';')
    
    # ===============================
    #separate data from labels 
    # ===============================   
    X = np.asarray(dataset.iloc[:,1:].values) #input values(features)
    y = np.asarray(dataset.iloc[:,0].values) #target values (labels)
    
    # ===============================
    #encode data
    # =============================== 
    res = encode_data(y)
    classes =res[0]  #liste des classes
    Y_True= res[2]
    return X, Y_True,classes 
def make_wine():
    # ===============================
    # Load dataset 
    # ===============================
    dataset= pd.read_csv("data/Wine.csv", sep=';')
    
    # ===============================
    #separate data from labels 
    # ===============================   
    X = np.asarray(dataset.iloc[:,1:].values) #input values(features)
    y = np.asarray(dataset.iloc[:,0].values) #target values (labels)
    
    # ===============================
    #encode data
    # =============================== 
    res = encode_data(y)
    classes =res[0]  #liste des classes
    Y_True= res[2]
    return X, Y_True,classes 

def make_zoo():
    # ===============================
    # Load dataset 
    # ===============================
    dataset= pd.read_csv("data/zoo.csv", sep=';')
    
    # ===============================
    #separate data from labels 
    # ===============================   
    X = np.asarray(dataset.iloc[:,2:].values) #input values(features)
    y = np.asarray(dataset.iloc[:,0].values) #target values (labels)
    
    # ===============================
    #encode data
    # =============================== 
    res = encode_data(y)
    classes =res[0]  #liste des classes
    Y_True= res[2]
    return X, Y_True,classes 
