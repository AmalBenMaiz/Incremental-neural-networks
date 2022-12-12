# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 08:13:18 2022

@author: benma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

class TLFN():
    def __init__(self, no_input_nodes, no_output_nodes, max_no_hidden_nodes, 
                 no_subregions, quantizer_T, quantizer_U, c,
                 activation_function='sigmoid', loss_function='mean_squared_error'):

        #self.name = name
        self.no_input_nodes = no_input_nodes
        self.max_no_hidden_nodes= max_no_hidden_nodes
        self.no_output_nodes = no_output_nodes
        self.no_subregions = no_subregions
        self.quantizer_T = quantizer_T
        self.quantizer_U = quantizer_U
        self.c= c
        
        # initialize weights of first hidden layer
        self.w_quant = []
        # y_predicted of first hidden layer
        self.y_pred_g = {}
        self.y_pred_g_test = {}
        
        # set an activation function
        self.activation_function = activation_function
        # set a loss function
        self.loss_function = loss_function
       
        
    def dividing_into_subregions(self, X, Y_True):
        
        #initialise random vector
        a = np.random.normal(size=X.shape[1])
        #reindex training saples
        r = a.dot(X.transpose())
        r_index_sort = np.argsort(r)
        r_sort = np.sort(r)
        b0,bL = min(r),max(r)
        L = self.no_subregions
        x_dict = {}
        y_dict={}
        b_list = []
        for p in range(L):
            bp = b0+ p*(bL-b0)/L
            bp1 = b0+ (p+1)*(bL-b0)/L
            if p!=(L-1):
                rp = r_sort[(r_sort >= bp)&(r_sort < bp1)]
                b_list.append(bp)
            else:
                rp = r_sort[r_sort >= bp]
                b_list.append(bp)
                b_list.append(bp1)
            r_index_p = r_index_sort[:len(rp)]
            r_index_sort = r_index_sort[len(rp):]
            x_dict[p+1] = X[r_index_p]
            y_dict[p+1] = Y_True[r_index_p] 
         
        return x_dict, b_list, y_dict, a
        
    def initialisation(self, X, Y_True):
        res_div = self.dividing_into_subregions(X, Y_True)
        b_list = res_div[1]
        #a= np.random.normal(size= self.no_subregions)
        a= res_div[3]
        alpha = np.array([self.quantizer_T*a, -self.quantizer_T*a])
        #activation function of AB quantizer nodes:
        Ha={}
        Hb={}
        for p in range(1, self.no_subregions+1):
            # Initialize Biases
            bias = np.array([-b_list[p], b_list[p-1]])
            Xp = res_div[0][p]
            Ha[p] = self.sigmoid(Xp.dot(alpha[0]) + bias[0])
            Hb[p] = self.sigmoid(Xp.dot(alpha[1]) + bias[1])
            
            self.w_quant.append((np.array([alpha[0],alpha[1]]).transpose(),
                      np.array([bias[0],bias[1]])))
        return res_div, Ha, Hb
    
    def mean_squared_error(self,Y_True, Y_Pred):
        return 0.5 * np.mean((Y_True - Y_Pred)**2)
    
    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))
    
    def sum_col (self, m, vect_add):
        line= m.shape[0]
        col=m.shape[1]
        for i in range(line):
            for j in range (col):
                m[i][j]= m[i][j] + vect_add[i]
        return m
        
    def predict(self, X, alpha, bias, beta):
        h = self.sigmoid(X.dot(alpha) + bias)
        return h.dot(beta)


    def evaluate(self, Y_true, X=None, alpha=None, bias=None, beta=None, metrics=['loss','accuracy'], 
                 final=False, Y_pred_new=None):
        if final:
            Y_pred = Y_pred_new
        else:
           Y_pred = np.asarray(self.predict(X, alpha, bias, beta))
        Y_true_argmax = np.argmax(Y_true, axis=-1)
        Y_pred_argmax = np.argmax(Y_pred, axis=-1)
        ret = []
        for m in metrics:
            if m == 'loss':
                loss = self.mean_squared_error(Y_true, Y_pred)
                ret.append(loss)
            elif m == 'accuracy':
                acc = np.sum(Y_pred_argmax == Y_true_argmax) / len(Y_true)
                ret.append(acc)
            else:
                raise ValueError('an unknown evaluation indicator \'%s\'.' % m)
        return ret
   
    def fit(self, X, Y_true, X_test, Y_test, Lmax, error):
        
        ini_mdl = self.initialisation(X, Y_true)
        x_sub, y_sub = ini_mdl[0][0], ini_mdl[0][2]
        Ha, Hb = ini_mdl[1], ini_mdl[2]
        ini_test = self.initialisation(X_test, Y_test)
        x_sub_test, y_sub_test = ini_test[0][0], ini_test[0][2]
        Ha_test, Hb_test = ini_test[1], ini_test[2]
        #initialisation Np hidden nodes 
        self.np_hidden_nodes = 0
        
        for p in range(1, self.no_subregions+1):
            
            X = x_sub[p]
            Y_true= y_sub[p]
            X_test = x_sub_test[p]
            Y_test = y_sub_test[p]
            #residual error
            E = 1#self.evaluate(X, Y_true , metrics=['loss', 'accuracy'])[0]
        
            while self.np_hidden_nodes < Lmax and E > error :
        
                # increase by one number of hidden nodes
                self.np_hidden_nodes= self.np_hidden_nodes + 1 
                
                if self.np_hidden_nodes <= len(self.w_quant):
                    # assign input weigts and biais of new hidden nodes
                    alpha = self.w_quant[p][0]
                    bias = self.w_quant[p][1]
                else:
                    #assign random input weights and biais of new hidden nodes
                    alpha = np.random.normal(size=(self.no_input_nodes,self.np_hidden_nodes))
                    bias = np.random.normal(size=self.np_hidden_nodes)
                    self.w_quant.append((alpha,bias))
                
                #output weight for new hidden node
                H = self.sigmoid(X.dot(alpha)+bias)
                # compute a pseudoinverse of H
                H_pinv = np.linalg.pinv(H)
                # update beta
                beta = H_pinv.dot(Y_true)
                
                #residual error
                eval_mdl = self.evaluate(Y_true=Y_true, X=X, alpha=alpha, bias=bias, beta=beta)
                E = eval_mdl[0]
            self.y_pred_g[p] = self.sum_col(self.predict(X, alpha, bias, beta), -self.quantizer_U*(Ha[p] + Hb[p]))
            self.y_pred_g_test[p] = self.sum_col(self.predict(X_test, alpha, bias, beta), -self.quantizer_U*(Ha_test[p] + Hb_test[p]))
        
        #save model's parameters (alpha, beta, biais)
        print("dim alpha :", alpha.shape, "dim beta :", beta.shape, "dim bias :", bias.shape)
        con = sqlite3.connect("TLFN_HAND.db") #connection with file type sqlite
        c = con.cursor() # create cursor
        #create tables
        c.execute('''CREATE TABLE IF NOT EXISTS ALPHA ([id] INTEGER PRIMARY KEY autoincrement, [val] Text )''')
        c.execute('''CREATE TABLE IF NOT EXISTS BETA([id] INTEGER PRIMARY KEY autoincrement, [val] Text )''')
        c.execute('''CREATE TABLE IF NOT EXISTS BIAS ([id] INTEGER PRIMARY KEY autoincrement, [val] Text)''')
        #filling in the tables
        for i in range(alpha.shape[0]):
            for j in range(alpha.shape[1]):
                c.execute("insert into ALPHA (val) values ("+str(alpha[i][j])+")")
        for i in range (beta.shape[0]):
            for j in range (beta.shape[1]):
                c.execute("insert into BETA (val) values ("+str(beta[i][j])+")")
        for bi in bias:
              c.execute("insert into BIAS (val) values ("+str(bi)+")")
        con.commit()
        con.close()
        
        bias_out= -0.5*self.c
        l_y_out= []
        l_y_true = []
        l_y_out_test= []
        l_y_true_test = []
        for p in range(1, self.no_subregions+1):
            l_y_out.append(self.sigmoid(self.y_pred_g[p]+ bias_out).dot(self.c))
            l_y_true.append(y_sub[p])
            l_y_out_test.append(self.sigmoid(self.y_pred_g_test[p]+ bias_out).dot(self.c))
            l_y_true_test.append(y_sub_test[p])
        Y_out = np.vstack(l_y_out)
        Y_true_new = np.vstack(l_y_true)
        Y_out_test = np.vstack(l_y_out_test)
        Y_true_new_test = np.vstack(l_y_true_test)
        
        eval_mdl = self.evaluate(Y_true=Y_true_new, final=True, Y_pred_new=Y_out)
        eval_df = pd.DataFrame({'nodes' : [self.np_hidden_nodes],
                                'loss' : [eval_mdl[0]],
                                'accuracy' : [eval_mdl[1]]})
        test_eval = self.evaluate(Y_true=Y_true_new_test, final=True, Y_pred_new=Y_out_test)
        eval_test_df = pd.DataFrame({'loss' : [test_eval[0]],
                                     'accuracy' : [test_eval[1]]})
        return eval_df, eval_test_df
            
            
        
        
        
        
           
      
        
    


        
        
