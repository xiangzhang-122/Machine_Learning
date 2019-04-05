#==============================================================================
#============================== SVM in DUAL FORM ==============================
#==============================================================================
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

# dual_x: the optimal dual variables
def predict_ker(dual_x, data, gamma):
    true_label = [row[-1] for row in data]
    pred_seq = [];
    for row in data:
        ll =[]
        for i in range(len(dual_x)):
            ll.append(dual_x[i] * train_data[i][-1] * g_ker(train_data[i][0:dim_s], row[0:dim_s], gamma ) )
        pred = sign_func(sum(ll))
        pred_seq.append(pred)
    return error_compute(pred_seq, true_label)
    
# Gaussian ker with para. gamma
# s_1 sample excluding label
def g_ker(s_1, s_2, gamma):
    s_1_ = np.asarray(s_1)
    s_2_ = np.asarray(s_2)
    return math.e**(-np.linalg.norm(s_1_ - s_2_)**2/gamma)   

# compute the K_hat matrix
def ker_mat_comp(gamma):
    K_hat_t = np.ndarray([train_len, train_len])
    for i in range(train_len):
        for j in range(train_len):
            K_hat_t[i,j] = g_ker(train_data[i][0:dim_s], train_data[j][0:dim_s], gamma)
    return K_hat_t

# x -alpha here, should be numpy array type
def svm_obj(x):
    tp1 = x.dot(K_mat_) 
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
    sol = minimize(svm_obj, x0, method = 'SLSQP', bounds = bds, constraints = cons)   # MINIMIZER
    return [sol.fun, sol.x]

# count the number of support vectors in dual opt variables dual_x(np.array)
def sup_vec_cnt(dual_x):
    ll = []
    for i in range(len(dual_x)):
        if dual_x[i] != 0.0:
            ll.append(i)
    return [np.count_nonzero(dual_x), set(ll)]
                  

def svm_main(C):
    [sol_f, sol_x] = svm_dual(C)
    [cnt, gg] = sup_vec_cnt(sol_x)
    print(cnt)
    print(gg)
    return [cnt, gg]
# =============================================================================
#     err_1 = predict_ker(sol_x, train_data, gamma)
#     err_2 = predict_ker(sol_x, test_data, gamma)
#     print('train err=', err_1)
#     print('test err=', err_2)
# =============================================================================
    
#------------------------- main function ------------------------
# underline at variable name end means global variable 
label_ = [row[-1] for row in train_data]  
CC_ = [100/873, 500/873, 700/873] 
Gamma_ =[0.01, 0.1, 0.5,1,2,5,10,100]
# =============================================================================
# for C in CC_:
#     for gamma in Gamma_:
#         print('C=',C, 'gamma=', gamma)
#         K_mat_ = ker_mat_comp(gamma)
#         svm_main(C)
# =============================================================================
C = 500/873
ll =[]
for gamma in Gamma_:
    print('C=',C, 'gamma=', gamma)
    K_mat_ = ker_mat_comp(gamma)
    [cnt, gg] = svm_main(C)  # gg is the index set of supp. vectors
    ll.append(gg)   

tt=[]    
for i in range(len(Gamma_)-1):
    tt.append(len(ll[i].intersection(ll[i+1])))
print('# overlaps=',tt)
    
    







#C =5/873
#gamma = 10       
#K_mat_ = ker_mat_comp(gamma)
#svm_main(C)
    
   






