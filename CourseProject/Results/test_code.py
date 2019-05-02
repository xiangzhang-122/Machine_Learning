# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 20:53:43 2019

@author: Larry Cheung
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import random

############################# banknote data ############################
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

train_data = add_cons_feature(polar_label(mylist_train))[0:400]
test_data = add_cons_feature(polar_label(mylist_test))

train_len = len(train_data)   # NO. of samples
test_len = len(test_data)


# ==================================== Nick data ======================================
# real_w = np.array(([1, 1, 1, 1, 1, 1, 1, 0]))/np.sqrt(7)
# 
# # num. points
# P = 500
# print(P)
# P_test = 5000
# 
# # number of atrributes (including bias term)
# num_attr = np.size(real_w)
# 
# # Data 
# train_data =  np.hstack( (2*np.random.rand(P,num_attr-1)-1,np.ones((P,1))))
# train_data = [list(row)  for row in train_data]
# #test_data = np.hstack( (2*np.random.rand(P_test,num_attr-1)-1,np.ones((P_test,1))))
# 
# # labels
# labels = list(np.sign(np.matmul(train_data,real_w)))
# train_len = len(train_data)   # NO. of samples
# 
# def append_label(data, labels):
#     for i in range(len(data)):
#         data[i].append(labels[i])
# 
# append_label(train_data, labels)
# =============================================================================
    

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
def local_update(wt, tao, data, sch_flag, C, gamma_0, d, iter_cnt):
    tao = int(tao)
    data_len = len(data)
    loss = []
    loss_interval =[]
    loss_interval.append(loss_func(wt, C, train_data ))
    for i in range(tao):
        print('tao = ', i)
        permu = np.random.permutation(data_len)
        [wt, iter_cnt, loss_] = svm_single(wt, iter_cnt, permu, data,C, sch_flag, gamma_0, d)
        loss.extend(loss_)
        loss_interval.append(loss_func(wt, C, train_data ))
    return [wt, loss, loss_interval, iter_cnt]
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
    return ll, sub_len, data_1, data_2

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
    data_1 =[]
    data_2 =[]
    for i in range(data_len):
        data_1.append(data[index_1[i]])
        data_2.append(data[index_2[i]])
    return data_1, data_2
    
def main(wt, T, tao, num_block, train_dataset, sch_flag, C, gamma_0, d, frac , shuffle_flag):
    if shuffle_flag == 'shuffle':
        iter_cnt = 1
    #    [ll, sub_len, data_1, data_2] = partition(train_dataset, num_block)
        perm_1, perm_2 = partition_1(train_dataset, 2)
        data_1, data_2 = exchange(train_dataset, frac, perm_1, perm_2)
        loss_interval_dist =[]
        loss_interval_dist.append(loss_func(wt, C, train_dataset))
        for i in range(T):
            print('T=',i)
            [wt_1, loss_1_, loss_intv_1, iter_cnt] = local_update(wt, tao, data_1, sch_flag, C, gamma_0, d, iter_cnt)
            [wt_2, loss_2_, loss_intv_2, iter_cnt] = local_update(wt, tao, data_2, sch_flag, C, gamma_0, d, iter_cnt)
            wt = global_agg([np.array(wt_1), np.array(wt_2)])
            loss_interval_dist.append(loss_func(wt, C, train_data))
            perm_1, perm_2 = partition_1(train_dataset, 2)
            data_1, data_2 = exchange(train_dataset, frac, perm_1, perm_2)
    #        loss_central.append(loss_func(wt, C, train_data))
    if shuffle_flag == 'no_shuffle':
        iter_cnt = 1
    #    [ll, sub_len, data_1, data_2] = partition(train_dataset, num_block)
        _,_, data_1, data_2 = partition(train_dataset, 2)
        loss_interval_dist =[]
        loss_interval_dist.append(loss_func(wt, C, train_dataset))
        for i in range(T):
            print('T=',i)
            [wt_1, _, _, iter_cnt] = local_update(wt, tao, data_1, sch_flag, C, gamma_0, d, iter_cnt)
            [wt_2, _, _, iter_cnt] = local_update(wt, tao, data_2, sch_flag, C, gamma_0, d, iter_cnt)
            wt = global_agg([np.array(wt_1), np.array(wt_2)])
            loss_interval_dist.append(loss_func(wt, C, train_data))       
    return [wt, loss_interval_dist]
        
# =======================   main funciton ===========================
# to tun the file, run svm(sch_flag, T, gamma_0,d)
sch_flag = 1
T_global = 100
T_epoch = 4
gamma_0 = 0.0005
d = gamma_0
C = 250/783
tao = 2
num_block = 2
frac_1 = 0.1
frac_2 = 0.5
#frac_3 = 0.5
#frac_4 = 0.8
w0 =list(np.zeros(len(train_data[0])-1)) 
[ _,loss_interval_dist_sh_1] = main(w0, T_global, tao, num_block, train_data, sch_flag, C, gamma_0, d, frac_1, 'shuffle' )
[ _,loss_interval_dist_sh_2] = main(w0, T_global, tao, num_block, train_data, sch_flag, C, gamma_0, d, frac_2, 'shuffle' )
#[ _,loss_interval_dist_sh_3] = main(w0, T_global, tao, num_block, train_data, sch_flag, C, gamma_0, d, frac_3, 'shuffle' )
#[ _,loss_interval_dist_sh_4] = main(w0, T_global, tao, num_block, train_data, sch_flag, C, gamma_0, d, frac_4, 'shuffle' )
#[ _,loss_interval_dist_sh_5] = main(w0, T_global, tao, num_block, train_data, sch_flag, C, gamma_0, d, frac_5, 'shuffle' )
[ _,loss_interval_dist_nosh] = main(w0, T_global, tao, num_block, train_data, sch_flag, C, gamma_0, d, 0.0, 'no_shuffle' )

iter_cnt = 1
[_, _, loss_interval_cen, _] = local_update(w0, T_global, train_data, sch_flag, C, gamma_0, d, iter_cnt)
plt.plot(loss_interval_dist_sh_1, label ='shuff, frac = 0.1')
plt.plot(loss_interval_dist_sh_2, label ='shuff, frac = 0.5')
#plt.plot(loss_interval_dist_sh_3, label ='shuff, frac = 0.5')
#plt.plot(loss_interval_dist_sh_4, label ='shuff, frac = 0.8')
#plt.plot(loss_interval_dist_sh_5[5: T_global], label ='shuff, frac = 1.0')
plt.plot(loss_interval_dist_nosh, label =' no shuffle')
plt.plot(loss_interval_cen, label= 'central')
#plt.plot(loss_interval_cen, label ='central')
#plt.title('tao = ', tao )
#plt.xlabel('iterations')
#plt.ylabel('loss')
plt.gca().legend()
plt.show()







