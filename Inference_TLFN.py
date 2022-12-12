# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:33:14 2022

@author: benma
"""

import sqlite3
import numpy as np
import pandas as pd

def show_class(y):
    tfy=[]
    Classes=[0,1,2,3,4,5,6,7,8,9]
    for e in y:
        if (e >= 0.75):
            e=1
        else:
            e=0
        tfy.append(e)
    print(tfy)
    for elt in tfy:
        if elt==1:
            i=tfy.index(elt)
            res= Classes[i]
    return res

class infmodel():
    def __init__(self):
        #scale factor
        self.c= 2
        con = sqlite3.connect("TLFN_HAND.db") #connection with file type sqlite
        c = con.cursor()
        dalpha=c.execute("select val from ALPHA").fetchall()
        dbeta=c.execute("select val from BETA").fetchall()
        dbias=c.execute("select val from BIAS").fetchall()
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
        h1 = self.sigmoid(X.dot(self.alpha) + self.bias)
        out1= h1.dot(self.beta)
        
        bias_out= -0.5*self.c 
        h2= self.sigmoid(out1 + bias_out)
        y_out= h2.dot(self.c)
        return y_out


Mi=infmodel() #construct model
#read input x
dataset= pd.read_csv("data/Hand_gesture_emg.csv", sep=';')
X = np.asarray(dataset.iloc[91,1:].values)
out= Mi.predict(X)
print("prediction: Y=", out)
classe = show_class(out)
print("This input  corresponds to the class ", classe)
    
 