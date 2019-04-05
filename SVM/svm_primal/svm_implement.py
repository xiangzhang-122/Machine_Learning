# SVM implement-- primal domain 
# Apr. 2019
# The bank-note data set was preciously used for the Perceptron algorithm 
import math
import statistics
import numpy as np
import matplotlib.pyplot as plot
from numpy.linalg import inv
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
#===========================================================================
#print(train_data)
# rate scheduling     
def rate_func_1(x, gamma_0, d):
    return gamma_0/(1 + gamma_0*x/d)

def rate_func_2(x, gamma_0):
    return gamma_0/(1 + x)
# curr_wt: the current weight vector dim: n+1
# sample: augmentted sample including label
#iter_cnt: number of overall iterations
# sch_flag ==1 rate_1, ==2, rate_2
# return: the updated weight
def sub_grad(curr_wt, sample, iter_cnt, sch_flag,C, gamma_0,d):
#    next_wt = 0
    next_wt = list(np.zeros(len(sample)-1))
    w_0 = curr_wt[0:len(curr_wt)-1];
    w_0.append(0)
    w_00 = w_0
    if sch_flag == 1:
        temp_1 = 1- rate_func_1(iter_cnt, gamma_0, d)
        temp_2 = rate_func_1(iter_cnt, gamma_0, d)
        temp_3 = temp_2*C*train_len*sample[-1]
        if sample[-1]*np.inner(sample[0:len(sample)-1], curr_wt) <= 1: 
            next_wt_1 = [x*temp_1 for x in w_00] 
            next_wt_2 = [x*temp_3 for x in sample[0:len(sample)-1]]
            next_wt = [next_wt_1[i] + next_wt_2[i] for i in range(len(next_wt_1))]
        else:
            next_wt = [x*temp_1 for x in w_00]
    if sch_flag == 2:
        temp_1 = 1- rate_func_2(iter_cnt, gamma_0)
        temp_2 = rate_func_2(iter_cnt, gamma_0)
        temp_3 = temp_2*C*train_len*sample[-1]
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
def svm_single(wt, iter_cnt, permu, train_data,C, sch_flag, gamma_0, d):
    loss_ = [];
    for i in range(train_len):
        wt = sub_grad(wt, train_data[permu[i]], iter_cnt, sch_flag,C,gamma_0,d)
        loss_.append(loss_func(wt, C, train_data))
        iter_cnt = iter_cnt + 1
    return [wt, iter_cnt, loss_]

# multi-epoch svm
# T- # of epochs
def svm_epoch(wt, T, train_data,C, sch_flag, gamma_0, d):
    iter_cnt = 1
    loss =[]
#    ct = 0
    for i in range(T):
        permu = np.random.permutation(train_len)
        [wt, iter_cnt, loss_] = svm_single(wt, iter_cnt, permu, train_data, C, sch_flag, gamma_0,d)
        loss.extend(loss_)
#        ct = ct + 1
#        print('# epoch=',ct)
    return [wt, loss]
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
        
def loss_func(wt, C, train_data):
    temp=[];
    for i in range(train_len):
        temp.append(max(0, 1- train_data[i][-1]*np.inner(wt, train_data[i][0:len(train_data[0])-1])))
    val = 0.5*np.linalg.norm(wt)**2 + C*sum(temp)
    return val
           
# flag =1/2: rate scheduling
def svm(sch_flag, T, gamma_0,d):
    C_global = [ x/873 for x in [1,10,50,100, 300, 500,700]]  # hyper parameter
    for C_glo in C_global:
        wt =list(np.zeros(len(train_data[0])-1)) 
        [ww, loss_val] = svm_epoch(wt, T, train_data, C_glo, sch_flag, gamma_0, d)
        print('LEARNED WEIGHT:',ww)
        err_train= predict(ww, train_data)
        err_test = predict(ww, test_data)
        print('TRAIN ERROR:', err_train)
        print('TEST ERROR:', err_test)
        
        
##----------main funciton-----------
# to tun the file, run svm(sch_flag, T, gamma_0,d)
sch_flag = 2
T = 100
gamma_0 = 2.3
d =1
svm(sch_flag, T, gamma_0,d)  # run file




#plot.plot(loss_val)
#plot.ylabel('loss')
#plot.xlabel('No. of iterations')
#plot.title('T=4')
#plot.show()

#print('loss convergence:',loss_func(ww,C_glo, train_data ))