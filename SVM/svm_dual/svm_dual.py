# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 23:54:41 2019

@author: Larry Cheung
"""
# ============================= SVM in dual form ==============================
import math
import statistics
import numpy as np
import matplotlib.pyplot as plot
from scipy.optimize import minimize
#========================= train data process =================================
with open('train.csv',mode='r') as f:
    myList_train=[];
    for line in f:
        terms=line.strip().split(',') # 7*N matrix
        myList_train.append(terms)
with open('test.csv',mode='r') as f:
    myList_test = [];
    for line in f:
        terms=line.strip().split(',') # 7*N matrix
        myList_test.append(terms)
        
def str_2_flo(data):
    for row in data:
        for j in range(len(data[0])):
            row[j] = float(row[j])
    return data

def add_cons_feature(data):   # add constant feature 1 to as the last feature before label
    label = [row[-1] for row in data]
    temp = data;
    for i in range(len(data)):
        temp[i][-1] = 1.0
    for i in range(len(data)):
        temp[i].append(label[i])
    return temp

# convert label {0,1} to {-1,1}
# data -- float
def polar_label(data):
    temp = data;
    for i in range(len(data)):
        temp[i][-1] = 2*data[i][-1]-1;
    return temp
        
mylist_train = str_2_flo(myList_train)  #convert to float  types data 
mylist_test = str_2_flo(myList_test) 

train_data = add_cons_feature(polar_label(mylist_train))
test_data = add_cons_feature(polar_label(mylist_test))

train_len = len(train_data)   # NO. of samples
test_len = len(test_data)
dim_s = len(train_data[0]) -1 # sample dim
#===========================================================================
def sign_func(x):
    y = 0
    if x> 0:
        y = 1
    else:
        y=-1
    return y
def error_compute(xx,yy):
    cnt = 0
    length =len(xx)
    for i in range(length):
        if xx[i]!= yy[i]:
            cnt = cnt + 1
    return cnt/length
def predict(wt, data):
    pred_seq =[];
    for i in range(len(data)):
        pred_seq.append(sign_func(np.inner(data[i][0:len(data[0])-1], wt)))
    label = [row[-1] for row in data]
    return error_compute(pred_seq, label)

# Gaussian ker with para. gamma
def g_ker(s_1, s_2, gamma):
    dim = len(s_1) - 1   # feature dimension
    s_11 = s_1[0:dim]
    s_22 = s_2[0:dim]
    diff = [s_11[i]-s_22[i] for i in range(dim)]
    ker = math.e**(-np.linalg.norm(diff)**2/gamma)
    return ker

# compute the K_hat matrix
def mat_comp():
    K_hat_t = np.ndarray([train_len, train_len])
    for i in range(train_len):
        for j in range(train_len):
            K_hat_t[i,j] = (train_data[i][-1])*(train_data[j][-1])*np.inner(train_data[i][0:dim_s],train_data[j][0:dim_s]) 
    return K_hat_t

# x -alpha here, should be numpy array type
def svm_obj(x):
    tp1 = x.dot(K_hat_) 
    tp2 = tp1.dot(x)
    tp3 = -1*sum(x)
    return 0.5*tp2 + tp3

def constraint(x):
    return np.inner(x, np.asarray(label_))


# returns the optimal dual vectors
def svm_dual(C):
    bd =(0,C)
    bds = tuple([bd for i in range(train_len)])
    x0 = np.zeros(train_len)
    cons ={'type':'eq', 'fun': constraint}
    sol = minimize(svm_obj, x0, method = 'SLSQP', bounds = bds, constraints = cons)
    return [sol.fun, sol.x]
# recover weight consiting of support vectors
def wt_recover(dual_x):
    lenn = len(dual_x)
    ll = []
    for i in range(lenn):
        ll.append(dual_x[i] * train_data[i][-1] * np.asarray(train_data[i][0: dim_s]))
    return sum(ll)
        
def svm_main(C):
    [sol_f, sol_x] = svm_dual(C)
    wt = wt_recover(sol_x)
    err_1 = predict(wt, train_data)
    err_2 = predict(wt, test_data)
    print('weight=', wt)
    print('train err=', err_1)
    print('test err=', err_2)
    
#------------------------- main function ------------------------
# underline at variable name end means global variable 
K_hat_ = mat_comp()
label_ = [row[-1] for row in train_data]     
CC = [100/873, 500/873, 700/873] 
for C_ in CC:
    svm_main(C_)
    





