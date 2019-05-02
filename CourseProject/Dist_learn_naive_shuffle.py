# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:34:08 2019

@author: Larry Cheung
-------------------------------
This file studies the effect of data shuffling (exchange among 
different distributed workers, each is responsible for locally 
updating its model) for federated learning. 
--------------------------------
Exact problem setup:
- golobal aggregation (GA) is performed every tao local updates
- each local update uses full gradient evaluation on D_i
- Local datasets are shuffled at the beginning of each GA
"""
# SVM implement-- primal domain 
# Apr. 2019
# The bank-note data set was preciously used for the Perceptron algorithm 
import math
import numpy as np
import matplotlib.pyplot as plt
import random
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

train_data = add_cons_feature(polar_label(mylist_train))[0:100]
test_data = add_cons_feature(polar_label(mylist_test))

train_len = len(train_data)   # NO. of samples
test_len = len(test_data)
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
        
def loss_func(wt, C, data):
    temp=[];
    data_len = len(data)
    for i in range(data_len):
        temp.append(max(0, 1- data[i][-1]*np.inner(wt, data[i][0:len(data[0])-1])))
    val = 0.5*np.linalg.norm(wt)**2 + C*sum(temp)
    return val

#print(train_data)
# rate scheduling    
def rate_func_0(gamma):
    return gamma
 
def rate_func_1(x, gamma_0, d):
    return gamma_0/(1 + gamma_0*x/d)

def rate_func_2(x, gamma_0):
    return gamma_0/(1 + x)

# curr_wt: the current weight vector dim: n+1
# sample: augmentted sample including label
#iter_cnt: number of overall iterations
# sch_flag ==1 rate_1, ==2, rate_2
# data -- the data that sgd is performed upon
# return: the updated weight
def sub_grad(curr_wt, sample, data, iter_cnt, sch_flag,C, gamma_0,d):
    data_len = len(data)
    next_wt = list(np.zeros(len(sample)-1))
    w_0 = list(curr_wt[0:len(curr_wt)-1]);
    w_0.append(0)
    w_00 = w_0
    # constant learning rate, vconstant stepsize = gamma_0
    if sch_flag == 0:    
        temp_1 = 1- rate_func_0(gamma_0)
        temp_2 = rate_func_0(gamma_0)
        temp_3 = temp_2*C*data_len*sample[-1]
        if sample[-1]*np.inner(sample[0:len(sample)-1], curr_wt) <= 1: 
            next_wt_1 = [x*temp_1 for x in w_00] 
            next_wt_2 = [x*temp_3 for x in sample[0:len(sample)-1]]
            next_wt = [next_wt_1[i] + next_wt_2[i] for i in range(len(next_wt_1))]
        else:
            next_wt = [x*temp_1 for x in w_00]
    # rate schedue 1, variable stepsize
    if sch_flag == 1:
        temp_1 = 1- rate_func_1(iter_cnt, gamma_0, d)
        temp_2 = rate_func_1(iter_cnt, gamma_0, d)
        temp_3 = temp_2*C*data_len*sample[-1]
        if sample[-1]*np.inner(sample[0:len(sample)-1], curr_wt) <= 1: 
            next_wt_1 = [x*temp_1 for x in w_00] 
            next_wt_2 = [x*temp_3 for x in sample[0:len(sample)-1]]
            next_wt = [next_wt_1[i] + next_wt_2[i] for i in range(len(next_wt_1))]
        else:
            next_wt = [x*temp_1 for x in w_00]
    # rate schedue 2, variable stepsize
    if sch_flag == 2:
        temp_1 = 1- rate_func_2(iter_cnt, gamma_0)
        temp_2 = rate_func_2(iter_cnt, gamma_0)
        temp_3 = temp_2*C*data_len*sample[-1]
        if sample[-1]*np.inner(sample[0:len(sample)-1], curr_wt) <= 1: 
            next_wt_1 = [x*temp_1 for x in w_00] 
            next_wt_2 = [x*temp_3 for x in sample[0:len(sample)-1]]
            next_wt = [next_wt_1[i] + next_wt_2[i] for i in range(len(next_wt_1))]
        else:
            next_wt = [x*temp_1 for x in w_00]
    return next_wt
# single epoch SVM
# wt: initial weight 
# permu: a random permutation of train_len
# iter_cnt: the initial overall iteration 
def svm_single(wt, iter_cnt, permu, data,C, sch_flag, gamma_0, d):
    loss_ = [];
    data_len = len(data)
    for i in range(data_len):
        wt = sub_grad(wt, data[permu[i]], data, iter_cnt, sch_flag,C,gamma_0,d)
        loss_.append(loss_func(wt, C, data))
        iter_cnt = iter_cnt + 1
    return [wt, iter_cnt, loss_]

def svm_epoch(wt, T, train_data,C, sch_flag, gamma_0, d):
    iter_cnt = 1
    loss =[]
    for i in range(T):
        permu = np.random.permutation(train_len)
        [wt, iter_cnt, loss_] = svm_single(wt, iter_cnt, permu, train_data, C, sch_flag, gamma_0,d)
        loss.extend(loss_)
    return [wt, loss] 
# when sch_flag = 0, gamma_0 is the constant stepsize
def local_update(wt, tao, data, sch_flag, C, gamma_0, d):
    tao = int(tao)
    iter_cnt = 1
    data_len = len(data)
    loss = []
    for i in range(tao):
        permu = np.random.permutation(data_len)
        [wt, iter_cnt, loss_] = svm_single(wt, iter_cnt, permu, data,C, sch_flag, gamma_0, d)
        loss.extend(loss_)
    return [wt, loss]
# wt_list is a list   
# output: averaged weight    
def global_agg(wt_list):
    temp = [np.array(x) for x in wt_list]
    return list(sum(temp)/2)
# partition the dataset into disjoint subssets(blocks) of 'subset_len' samples
# num_block the number of blocks, shoud be even for two-worker setting
# throw out the outlier data
def partition(dataset, num_block):
    num_block = int(num_block)
    ll =[]
    dt_len =len(dataset)
    sub_len = math.floor(dt_len/num_block)
#    r = dt_len % sub_len
    for i in range(num_block):
        ll.append(dataset[i*sub_len :(i+1)*sub_len ])
    data_1 = ll[0:int(num_block/2)][0]
    data_2 = ll[int(num_block/2):num_block][0]
    return [ll, sub_len, data_1, data_2]

def partition_1(dataset, num):
    data_len = len(dataset)
    perm = list(np.random.permutation(data_len))
    perm_1 = perm[0:int(data_len/2)]
    perm_2 = perm[int(data_len/2): data_len]
    return perm_1, perm_2

def exchange(data, frac, perm_1, perm_2):
    data_len = len(perm_1)
    s_1 = set(list(perm_1))
    s_2 = set(list(perm_2))
    n = int(np.floor(data_len*frac))
    ex_1 = set(random.sample(s_1, n))
    ex_2 = set(random.sample(s_2, n))
    s_1 -= ex_1
    s_2 -= ex_2
    s_1.update(ex_2)
    s_2.update(ex_1)
    index_1 = list(s_1)
    index_2 = list(s_2)
    print(index_1, index_2)
    data_1 =[]
    data_2 =[]
    for i in range(data_len):
        data_1.append(data[index_1[i]])
        data_2.append(data[index_2[i]])
    return data_1, data_2


        
# ======================= set parameters and run the code here ===========================
# to tun the file, run svm(sch_flag, T, gamma_0,d)
sch_flag = 1
T_global = 10
T_epoch = 1
gamma_0 = 2
d =1
C =5
tao = 2
num_block =2
w0 =list(np.zeros(len(train_data[0])-1)) 
[wt, loss_dist,dt_1, dt_2, loss_1, loss_2] = main(w0, T_global, tao, num_block, train_data, sch_flag, C, gamma_0, d )
#[wt, loss_cen] = local_update(w0, tao, train_data, sch_flag, C, gamma_0, d)
#plt.plot(loss_1, label ='loss_worker_1')
#plt.plot(loss_2, label ='loss_worker_2')
##plt.plot(loss_cen, label ='central')
#
#plt.gca().legend()
#plt.show()

plt.plot(loss_dist, label ='dist')





