# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 22:57:16 2019

@author: Larry Cheung
"""
#============= Kernel perceptron algorithm =================
# keep track of the mistake counters instead of updating weight vecotr
# indeed these two are equivalent
# using Gaussian kernel
import math
import statistics
import numpy as np
#import matplotlib.pyplot as plot
from numpy.linalg import inv

#========================= data pre-processing =================================
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
# append the last column as error count
# intitalize with all zero counts
def err_cnt_append(data):
    for row in data:
        row.append(0)
    return data
                   
mylist_train = str_2_flo(myList_train)  #convert to float  types data 
mylist_test = str_2_flo(myList_test) 

train_data_ = add_cons_feature(polar_label(mylist_train))
train_data = err_cnt_append(train_data_)
test_data = add_cons_feature(polar_label(mylist_test))

train_len = len(train_data)   # NO. of samples
test_len = len(test_data)
dim_s = len(train_data[0]) -2   # sample dim
#================================end data processing ==========================

# Gaussian kernel para. gamma
# s_1, s_2 two samples excluding labels
def g_ker(s_1, s_2, gamma):
    s_1_ = np.asarray(s_1)
    s_2_ = np.asarray(s_2)
    return math.e**(-np.linalg.norm(s_1_ - s_2_)**2/gamma)   

def sign_func(x):
    y = 0
    if x> 0:
        y = 1
    else:
        y=-1
    return y

def predict_sin(traindata, sample, gamma):
    temp = sum([row[-1]*row[-2]*g_ker(row[0:dim_s], sample, gamma) for row in traindata])
    return sign_func(temp) 

def error_compute(xx,yy):
    cnt = 0
    length =len(xx)
    for i in range(length):
        if xx[i]!= yy[i]:
            cnt = cnt + 1
    return cnt/length

def predict(data, traindata, gamma):
    pred_seq =[]
    for row in data:
        pred_seq.append(predict_sin(traindata, row[0:dim_s], gamma))
    return pred_seq
        
# update error counter 
# train_data: appended with error counter
def ker_perceptron(traindata, gamma):
    for row in traindata:
        if row[-2] != predict_sin(traindata, row[0:dim_s], gamma):
            row[-1] = row[-1] + 1
    return traindata
# onyly T =1 epoch is implemented here
#permu = range(train_len)
Gamma = [0.01, 0.1, 0.5,1,2,5,10,100]
def main_ker_percp(Gamma):
    for gamma  in Gamma:
        print('gamma=',gamma)
        train_up = ker_perceptron(train_data, gamma)
        pred_seq_train = predict(train_data, train_up, gamma)
        pred_seq_test = predict(test_data, train_up, gamma)
        train_label =[row[-2] for row in train_data]
        test_label = [row[-1] for row in test_data]
        err_train = error_compute(pred_seq_train, train_label)
        err_test = error_compute(pred_seq_test, test_label)
        print('train err =', err_train)
        print('test err =', err_test)

#===================================== main ===================================
# run the file
main_ker_percp(Gamma)




        
        