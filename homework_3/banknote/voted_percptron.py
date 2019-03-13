# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:17:35 2019

@author: Larry Cheung
"""

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



def null_remove(ll): # remove the zeros at the end of ll
    temp =[];
    for i in range(len(ll)):
        if ll[i]!= 0:
            temp.append(ll[i])
    return temp
              
# oOUTPUT: CC--(w,b) matrix, C_last_col--vote/survival times   
def voted_perc_train(train_data, w, b, rate, T):
    data_len = len(train_data)
    permu = [];   # length = T*len(train_data))
    for i in range(T):
        pi= np.random.permutation(data_len).tolist()
        permu= permu + pi
#    print(len(permu))
    C_last_col =np.zeros(T*data_len).tolist()
    CC= []
    m =0
    for i in range(T*data_len):
        if (train_data[permu[i]][-1])*( np.inner(w, train_data[permu[i]][0:Num_attr])+b ) <= 0:
            w = w + [rate *(train_data[permu[i]][-1])*x for x in train_data[permu[i]][0:Num_attr]] 
            # sumamtion of a list and an array is array type
            b= b + rate*train_data[permu[i]][-1]   # scalar
            m +=1
            row = np.append(w,b)
            CC.append(row)
            C_last_col[m]=1                
        if (train_data[permu[i]][-1])*( np.inner(w, train_data[permu[i]][0:Num_attr])+b ) > 0:
            C_last_col[m] += 1
    return [CC, null_remove(C_last_col)]

#[a,b] = voted_perc_train(train_polar, np.zeros(Num_attr), 0, 1,1)
#print(len(a))
#print(len(b))
#print(len(train_polar))
def sign_func(x):
    a=0
    if x >0:
        a=1
    else:
        a=-1
    return a
# CC--array type
def _merge_(CC,c):
    temp =[]
    for i in range(len(CC)):
        tt = CC[i].tolist() + [c[i]]
        temp.append(tt)
    return temp
#INPUT: c --vote list
#OUTPUT: prediction error
def voted_perc_pred(test_data, CC,c):
    CC_c =_merge_(CC,c)
    pred_seq =[]
    for i in range(len(test_data)):
       pred_seq.append(sign_func(sum( [(row[-1])*sign_func(np.inner(test_data[i][0:Num_attr], row[0:Num_attr] )+row[Num_attr]) for row in CC_c])))
    count =0
    for i in range(len(test_data)):
        if pred_seq[i] != test_data[i][-1]:
            count+=1
    return count/len(test_data)

# warning: initial w shuld be array type 
def voted_perceptron(train_data, test_data, w, b, rate, T):
    [CC,c] = voted_perc_train(train_data, w, b, rate, T)
    print(c)
    print(_merge_(CC,c))
#    print(sum(c))
    err = voted_perc_pred(test_data, CC,c)
    print(err)


rate = 1    
T =10
w= np.zeros(Num_attr)
b=0
voted_perceptron(train_polar, test_polar, w, b, rate, T)      
        
    
                
            
        
        
