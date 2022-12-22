# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 12:23:36 2022

@author: benma
"""

import sqlite3
import numpy as np
import pandas as pd

class infmodel():
    def __init__(self):
        #con = sqlite3.connect("I_ELM_HAND.db") #connection with file type sqlite
        con = sqlite3.connect("EM_ELM_HAND.db") #connection with file type sqlite
        c = con.cursor()
        dalpha=c.execute("select val from alpha").fetchall()
        dbeta=c.execute("select val from beta").fetchall()
        dbias=c.execute("select val from bias").fetchall()
        self.alpha=np.zeros((22,len(dbias)),dtype=float)
        self.beta = np.zeros((len(dbias),10), dtype=float)
        self.bias = np.zeros((len(dbias)), dtype=float)
        print("dim alpha :", self.alpha.shape, "dim beta :", self.beta.shape, "dim bias :",self.bias.shape)
        k=0
        for i in range (self.alpha.shape[0]):
            for j in range(self.alpha.shape[1]):
                self.alpha[i,j]=float(dalpha[k][0])
                k=k+1
        k=0
        for i in range(self.beta.shape[0]):
            for j in range(self.beta.shape[1]):
                self.beta[i,j] = float(dbeta[k][0])
                k=k+1
        k=0
        for i in range(self.bias.shape[0]):
            self.bias[i] = float(dbias[k][0])
            k=k+1
            
    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))
    
    def predict(self, X):
        return list(self(X))
    
    def __call__(self, X):
        h = self.sigmoid(X.dot(self.alpha) + self.bias)
        return h.dot(self.beta)
    

Mi=infmodel() #construct model
#read input x
dataset= pd.read_csv("data/Hand_gesture_emg.csv", sep=';')
X = np.asarray(dataset.iloc[993,1:].values)
out= Mi.predict(X)

