# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:57:15 2022

@author: benma
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class I_ELM():
    """ Constructor to initialize node"""
    def __init__(self, no_input_nodes, max_no_hidden_nodes, no_output_nodes,
        activation_function='sigmoid', loss_function='mean_squared_error'):
        
        #self.name = name
        self.no_input_nodes = no_input_nodes
        self.no_hidden_nodes = 1
        self.no_output_nodes = no_output_nodes

        # initialize weights between hidden layer and Output Layer
        self.beta = np.random.normal(size=(self.no_hidden_nodes, self.no_output_nodes))
        # initialize weights between Input Layer and hidden layer
        self.alpha = np.random.normal(size=(self.no_input_nodes, self.no_hidden_nodes))
        # Initialize Biases
        self.bias = np.zeros(self.no_hidden_nodes)
        # set an activation function
        self.activation_function = activation_function
        # set a loss function
        self.loss_function = loss_function
     
       
    def mean_squared_error(self,Y_True, Y_Pred):
        return 0.5 * np.mean((Y_True - Y_Pred)**2)
    
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
    
    
    def fit(self, X, Y_true, X_test, Y_test, Lmax, error):
        
        H = self.sigmoid(X.dot(self.alpha)+self.bias)
        # compute a pseudoinverse of H
        H_pinv = np.linalg.pinv(H)
        # update beta
        self.beta = H_pinv.dot(Y_true)
        
        #residual error
        E = self.evaluate(X, Y_true)[0]
           
      
        while self.no_hidden_nodes < Lmax and E > error :
            # increase by one number of hidden nodes
            self.no_hidden_nodes= self.no_hidden_nodes + 1 
           
            # assign random input weigt biais
            self.bias = np.random.normal(size=self.no_hidden_nodes)
            self.alpha = np.random.normal(size=(self.no_input_nodes,self.no_hidden_nodes))
            
            #output weight for new hidden node
            H = self.sigmoid(X.dot(self.alpha)+self.bias)
            # compute a pseudoinverse of H
            H_pinv = np.linalg.pinv(H)
            # update beta
            self.beta = H_pinv.dot(Y_true)
            
            #residual error
            E = self.evaluate(X, Y_true)[0]
            
           
        eval_df = pd.DataFrame({'nodes' : [self.no_hidden_nodes],
                                'loss' : [self.evaluate(X, Y_true)[0]],
                                'accuracy' : [self.evaluate(X, Y_true)[1]]})
       
    
        eval_test_df = pd.DataFrame({'loss' : [self.evaluate(X_test, Y_test)[0]],
                                      'accuracy' : [self.evaluate(X_test, Y_test)[1]]})
        return eval_df, eval_test_df
    
    
            
        

       
       
           
        
           
           
           
       
       
    
           
        
           