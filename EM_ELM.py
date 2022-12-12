# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 15:43:16 2022

@author: benma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randint
import sqlite3

    
class EM_ELM():
    """ Constructor to initialize node"""
    def __init__(self, no_input_nodes, max_no_hidden_nodes, no_output_nodes,no_nodes_rand_min=1, 
                 no_nodes_rand_max=15, activation_function='sigmoid', loss_function='mean_squared_error'):

        #self.name = name
        self.no_input_nodes = no_input_nodes
        self.max_no_hidden_nodes= max_no_hidden_nodes
        self.no_output_nodes = no_output_nodes
        self.no_nodes_rand_min= no_nodes_rand_min
        self.no_nodes_rand_max= no_nodes_rand_max
      
        
        #initialize hidden nodes randomly
        self.no_hidden_nodes = randint(no_nodes_rand_min, no_nodes_rand_max) 

        # initialize weights between hidden layer and Output Layer
        self.beta = np.random.normal(size=(self.no_hidden_nodes, self.no_output_nodes))
        # initialize weights between Input Layer and hidden layer
        self.alpha = np.random.normal(size=(self.no_input_nodes, self.no_hidden_nodes))
        # Initialize Biases
        self.bias = np.random.normal(size=self.no_hidden_nodes)
        #self.bias = np.random.uniform(-1.,1.,size= self.no_hidden_nodes)
        # set an activation function
        self.activation_function = activation_function
        # set a loss function
        self.loss_function = loss_function
       
    def mean_squared_error(self,Y_True, Y_Pred):
        return 0.5 * np.mean((Y_True - Y_Pred)**2)
    
    def mean_absolute_error(self, Y_True, Y_Pred):
        return np.mean(np.abs(Y_True - Y_Pred))
    
    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))
    
    def predict(self, X):
        return list(self(X))
    
    def __call__(self, X):
        h = self.sigmoid(X.dot(self.alpha) + self.bias)
        return h.dot(self.beta)
    
    def evaluate(self, X, Y_true, metrics=['loss', 'accuracy']):
        Y_pred = np.asarray(self.predict(X)) 
        Y_true = Y_true
        Y_pred_argmax = np.argmax(Y_pred, axis=-1)
        Y_true_argmax = np.argmax(Y_true, axis=-1)
        ret = []
        for m in metrics:
            if m == 'loss':
                loss = self.mean_squared_error(Y_true_argmax,  Y_pred_argmax)
                ret.append(loss)
            elif m == 'accuracy':
                acc = np.sum(Y_pred_argmax == Y_true_argmax) / len(Y_true)
                ret.append(acc)
            else:
                raise ValueError('an unknown evaluation indicator \'%s\'.' % m)
        return ret
    
    def fit (self, X, Y_true, X_test, Y_test, Nmax, error):
        
        I = np.identity(X.shape[0])
        # hidden layer output matrix
        H = self.sigmoid(X.dot(self.alpha)+ self.bias)
        # compute a pseudoinverse of H
        H_pinv = np.linalg.pinv(H)
    
        #output  error
        E = self.evaluate(X, Y_true)[0]
        
           
        while self.no_hidden_nodes < Nmax and E > error :
     
            
            # randomly increase by delta_N number of hidden nodes
            detla_N = randint(self.no_nodes_rand_min, self.no_nodes_rand_max)
            
            self.no_hidden_nodes= self.no_hidden_nodes + detla_N
            
            H_past = H
            inv_H_past= H_pinv
            
            # assign random input weigt biais
            self.delta_bias = np.random.normal(size= detla_N)
            self.delta_alpha = np.random.normal(size=(self.no_input_nodes, detla_N))
            self.alpha=np.hstack([self.alpha,self.delta_alpha])
            self.bias=np.hstack([self.bias,self.delta_bias])
            
            delta_H = self.sigmoid(X.dot(self.delta_alpha)+self.delta_bias)   
            #update output matrix
            H = np.hstack([H,delta_H])
            # compute a pseudoinverse of H
            H_pinv = np.linalg.pinv(H)
            
            D = np.linalg.pinv((I-(H_past.dot(inv_H_past))).dot(delta_H))
            U = inv_H_past.dot((I-delta_H.dot(D)))
            
            # update beta
            self.beta = np.vstack([U, D]).dot(Y_true)
            
            #update residual error
            E = self.evaluate(X, Y_true)[0]
            
        
        eval_df = pd.DataFrame({'nodes' : [ self.no_hidden_nodes],
                                'loss' : [self.evaluate(X, Y_true)[0]],
                                'accuracy' : [self.evaluate(X, Y_true)[1]]})
     
        
        eval_test_df = pd.DataFrame({'loss' : [self.evaluate(X_test, Y_test)[0]],
                                     'accuracy' : [self.evaluate(X_test, Y_test)[1]]})
        return eval_df, eval_test_df
    #save model's parameters (alpha, beta, biais)
    def saveMPR(self):
        print("dim alpha :", self.alpha.shape, "dim beta :", self.beta.shape, "dim bias :",self.bias.shape)
        con = sqlite3.connect("EM_ELM_HAND.db") #connection with file type sqlite
        c = con.cursor() # create cursor
        #create tables
        c.execute('''CREATE TABLE IF NOT EXISTS alpha ([id] INTEGER PRIMARY KEY autoincrement, [val] Text )''')
        c.execute('''CREATE TABLE IF NOT EXISTS beta ([id] INTEGER PRIMARY KEY autoincrement, [val] Text )''')
        c.execute('''CREATE TABLE IF NOT EXISTS bias ([id] INTEGER PRIMARY KEY autoincrement, [val] Text)''')
        #filling in the tables
        for i in range(self.alpha.shape[0]):
            for j in range(self.alpha.shape[1]):
                c.execute("insert into alpha (val) values ("+str(self.alpha[i][j])+")")
        for i in range (self.beta.shape[0]):
            for j in range (self.beta.shape[1]):
                c.execute("insert into beta (val) values ("+str(self.beta[i][j])+")")
        for bi in self.bias:
              c.execute("insert into bias (val) values ("+str(bi)+")")
        con.commit()
        con.close()
