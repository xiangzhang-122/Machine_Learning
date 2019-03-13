# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 19:13:12 2019

@author: Larry Cheung
"""
# perceptron algorithm
import math
import statistics
import numpy as np
import matplotlib.pyplot as plot
from numpy.linalg import inv
# constants

with open('train.csv',mode='r') as f:
    myList_train=[];
    for line in f:
        terms=line.strip().split(',') # 7*N matrix
        myList_train.append(terms)
with open('test.csv',mode='r') as f:
    myList_test=[];
    for line in f:
        terms=line.strip().split(',') # 7*N matrix
        myList_test.append(terms)
      
def str_2_flo(mylist):   # convert string to float type
    temp_list = mylist
    for k in range(len(temp_list)):
        for i in range(len(temp_list[0])):
            temp_list[k][i] = float(mylist[k][i])
    return temp_list  

def polar_label(mylist): # convert 0-1 label in +1,-1
    temp_list = mylist
    for i in range(len(mylist)):
        temp_list[i][-1] = 2*mylist[i][-1]-1
    return temp_list      
####### CONSTANTS ##########     
Num_attr = len(myList_train[0])-1; 
mylist_train = str_2_flo(myList_train)
mylist_test = str_2_flo(myList_test)      
train_polar = polar_label(mylist_train)
test_polar = polar_label(mylist_test)    

#print(myList_train)
# b: bias, w- initial weight
# pi is a random permutation
def perc_single(mylist,pi, w, b, rate):  
    for i in range(len(mylist)):
        if (mylist[pi[i]][-1])*(np.inner(w, mylist[pi[i]][0:Num_attr])+b) <=0:
            w = w + [rate*(mylist[pi[i]][-1])*x for x in mylist[pi[i]][0:Num_attr] ]    # arrary summation
            b = b + rate*mylist[pi[i]][-1]*1
    return [w,b]


def _pred(test_data, w,b):   # returns the predictiong error in test_data
    num_test =len(test_data)
    count = 0
    for i in range(num_test):
        if (test_data[i][-1])*(np.inner(w, test_data[i][0:Num_attr])+b) <=0:
            count +=1;
    return count/num_test
        
       
def epoch_perc(train_data, w, b,rate,T):
    for t in range(T):
        pi = np.random.permutation(len(train_data))
        [w, b] = perc_single(train_data, pi, w, b, rate )
    return [w,b]

rate = 1     
T =10
###################### standard perceptron  #########################
def std_perc(train_p, test_p, w , b, rate, T):
    [ww,bb] = epoch_perc(train_p, w,b, rate, T)
    err = _pred(test_polar, ww, bb)
    return [ww, bb, err]  
print(std_perc(train_polar, test_polar, np.zeros(Num_attr), 0 ,rate,T))
    

    
