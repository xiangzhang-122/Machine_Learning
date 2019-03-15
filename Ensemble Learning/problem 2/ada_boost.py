# Adaboost algorithm
# only for stumps, the 'used attrs can not be used again ' issue is not fixed here
import math
import statistics
import numpy as np
import matplotlib.pyplot as plot
#-----------------train data process------------------------------------
with open('train.csv',mode='r') as f:
    myList_train=[];
    for line in f:
        terms=line.strip().split(',') # 7*N matrix
        myList_train.append(terms)

num_set={0,5,9,11,12,13,14} # indices of numeric attr   
def str_2_flo(mylist):
    temp_list = mylist
    for k in range(len(temp_list)):
        for i in {0,5,9,11,12,13,14}:
            temp_list[k][i] = float(mylist[k][i])
    return temp_list

mylist_train = str_2_flo(myList_train)

obj={0:0,5:5,9:9,11:11,12:12,13:13,14:14}
for i in obj:
    obj[i] = statistics.median([row[i] for row in mylist_train])
    
for row in mylist_train:
    for i in obj:
        if row[i] >= obj[i]:
            row[i] = 'yes'
        else:
            row[i] = 'no'                     
#--------------------test data process--------------------------------
with open('test.csv',mode='r') as f:
    myList_test=[];
    for line in f:
        terms=line.strip().split(',') # 7*N matrix
        myList_test.append(terms)

mylist_test = str_2_flo(myList_test)
for i in obj:
    obj[i] = statistics.median([row[i] for row in mylist_test])
#binary quantization of numerical attributes
for row in mylist_test:
    for i in obj:
        if row[i] >= obj[i]:
            row[i] = 'yes'
        else:
            row[i] = 'no'  
#------------------------------------------------------------------------------
Attr_dict ={'age':['yes','no'],
             'job':['admin.','unknown','unemployed','management',
                    'housemaid','entrepreneur','student','blue-collar',
                    'self-employed','retired','technician','services'],
                    'martial':['married','divorced','single'],
                    'education':['unknown','secondary','primary','tertiary'],
                     'default':['yes','no'],
                     'balance':['yes','no'],
                     'housing':['yes','no'],
                     'loan':['yes','no'],
                     'contact':['unknown','telephone','cellular'],
                     'day':['yes','no'],
                     'month':['jan', 'feb', 'mar', 'apr','may','jun','jul','aug','sep','oct', 'nov', 'dec'],
                     'duration': ['yes','no'],
                     'campaign':['yes','no'],
                     'pdays':['yes','no'],
                     'previous':['yes','no'],
                     'poutcome':[ 'unknown','other','failure','success']}

Attr_set = set(key for key in Attr_dict)   # the set of all possible attr.s

def pos(attr):
    pos=0
    if attr=='age':
        pos=0
    if attr=='job':
        pos=1
    if attr=='martial':
        pos=2
    if attr=='education':
        pos=3
    if attr=='default':
        pos=4
    if attr=='balance':
        pos=5
    if attr=='housing':
        pos=6
    if attr=='loan':
        pos=7
    if attr=='contact':
        pos=8
    if attr=='day':
        pos=9
    if attr=='month':
        pos=10
    if attr=='duration':
        pos=11
    if attr=='campaign':
        pos=12
    if attr=='pdays':
        pos=13
    if attr=='previous':
        pos=14
    if attr=='poutcome':
        pos=15
    if attr=='y':
        pos=16
    return pos        
 
#----------------------------------------------------------------------------  
def create_list(attr):
    obj={}
    for attr_val in Attr_dict[attr]:
        obj[attr_val]=[]
    return obj   # dict type with list value type

def create_list_0(attr):
    obj={}
    for attr_val in attr:
        obj[attr_val]=0
    return obj    # dict type with float value type   dict=(key,value) 
#----------------------------------------------------------------------------
# =============================================================================
# # weighted expected entropy
# # groups:  dict type
# # classes: ther set of labels contained in 'groups'
# def exp_entropy(groups, classes):
#     #N_ins= float(sum([len(groups[attr_val]) for attr_val in groups])) 
#     Q = 0.0   #total weights
#     tp =0.0
#     for attr_val in groups:
#         tp = sum([row[-1] for row in groups[attr_val]])
#         Q = Q + tp 
#         
#     exp_ent = 0.0
#     for attr_val in groups:
#         size = float(len(groups[attr_val]))
#         if size == 0:
#             score = 0
#         else:
#             score = 0.0
#             q = sum([row[-1] for row in groups[attr_val]])
#             for class_val in classes:
# #            p = [row[-3] for row in groups[attr_val]].count(class_val) / size   ###            
#                 p = sum([row[-1] for row in groups[attr_val] if row[-2] == class_val])/q  #sum up the weights
#                 if p==0:
#                     temp=0
#                 else:
#                     temp=p*math.log2(1/p)
#                 score +=temp 
# #        exp_ent += score* (size / N_ins)
#         exp_ent += score* sum([row[-1] for row in groups[attr_val]])/Q #total weights of a subset
#     return exp_ent          
# =============================================================================
    
def exp_entropy(groups, classes):
    #N_ins= float(sum([len(groups[attr_val]) for attr_val in groups])) 
    Q = 0.0   #total weights
    tp =0.0
    for attr_val in groups:
        tp = sum([row[-1] for row in groups[attr_val]])
        Q = Q + tp        
    exp_ent = 0.0
    for attr_val in groups:
        size = float(len(groups[attr_val]))
        if size == 0:
            continue    # jump this iteration
        score = 0
        q = sum([row[-1] for row in groups[attr_val]])
        for class_val in classes:
#           p = [row[-3] for row in groups[attr_val]].count(class_val) / size   ###            
            p = sum([row[-1] for row in groups[attr_val] if row[-2] == class_val])/q  #sum up the weights
            if p==0:
                temp=0
            else:
                temp=p*math.log2(1/p)
            score +=temp 
#        exp_ent += score* (size / N_ins)
        exp_ent += score* sum([row[-1] for row in groups[attr_val]])/Q #total weights of a subset
    return exp_ent          
#--------------------------------------------------------------------------
#dataset: list type
# return an dict with subsets of samples corres. to vlaues of attr.
def data_split(attr, dataset):
    branch_obj=create_list(attr)  # this may result in empty dict elements 
    for row in dataset:
        for attr_val in Attr_dict[attr]:
           if row[pos(attr)] == attr_val:
               branch_obj[attr_val].append(row)
    return branch_obj
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
def find_best_split(dataset):
    if dataset==[]:
        return 
    label_values = list(set(row[-2] for row in dataset)) 
    metric_obj = create_list_0(Attr_dict)
    for attr in Attr_dict:
        groups = data_split(attr, dataset)
        metric_obj[attr] = exp_entropy(groups, label_values)             # change metric here
    best_attr = min(metric_obj, key=metric_obj.get)
    best_groups = data_split(best_attr, dataset)  
    return {'best_attr':best_attr, 'best_groups':best_groups}
#----------------------------------------------------------------------------    
# returns the majority label within 'group' (list type)
def leaf_node_label(group):
    majority_labels = [row[-2] for row in group]    # we deal with data appended with weight 
    return max(set(majority_labels), key=majority_labels.count)
#----------------------------------------------------------------------------
def if_node_divisible(branch_obj):
    non_empty_indices=[key for key in branch_obj if not (not branch_obj[key])]
    if len(non_empty_indices)==1:
        return False
    else:
        return True
#-------------------------------------------------------------------------
def child_node(node, max_depth, curr_depth):
#    if not if_node_divisible(node['best_groups']):  #only one non-empty branch
#        # what if all elements in node 
#        for key in node['best_groups']:
#            if  node['best_groups'][key]!= []: #and ( not isinstance(node['best_groups'][key],str)): 
#                 node[key] = leaf_node_label(node['best_groups'][key])
#            else:
#                node[key] = leaf_node_label(sum(node['best_groups'].values(),[])) 
#        return
    if curr_depth >= max_depth:
        for key in node['best_groups']:
            if  node['best_groups'][key]!= []: #and ( not isinstance(node['best_groups'][key],str)):
                # extract nonempty branches
                node[key] = leaf_node_label(node['best_groups'][key])   
            else:
                node[key] = leaf_node_label(sum(node['best_groups'].values(),[]))
        return  
    for key in node['best_groups']:
        if node['best_groups'][key]!= []: #and ( not isinstance(node['best_groups'][key],str)):
            node[key] = find_best_split(node['best_groups'][key]) 
            child_node(node[key], max_depth, curr_depth + 1)
        else:
            node[key] = leaf_node_label(sum(node['best_groups'].values(),[]))  

    
#----------------------------------------------------------------------------
def tree_build(train_data, max_depth):
	root = find_best_split(train_data)               #########add weights here
	child_node(root, max_depth, 1)
	return root
#----------------------------------------------------------------------------
#test if an instance belongs to a node recursively
def label_predict(node, inst):
    if isinstance(node[inst[pos(node['best_attr'])]],dict):
        return label_predict(node[inst[pos(node['best_attr'])]],inst)
    else:
        return node[inst[pos(node['best_attr'])]]   #leaf node

#sign function
def sign_func(val):
    if val > 0:
        return 1.0
    else:
        return -1.0 
#------return the true label and predicted result using stump 'tree'-----
def label_return(dataset,tree):
    true_label = []
    pred_seq = []   # predicted sequence
    for row in dataset:
        true_label.append(row[-2])    
        pre = label_predict(tree, row)
        pred_seq.append(pre)
    return [true_label, pred_seq]
    
# create dict with n keys and list element
def list_obj(n):
    obj={}
    for i in range(n):
        obj[i] = []
    return obj
#-------------------------convert label to binaries---------------------
def bin_quan(llist):
    bin_list =[]
    for i in range(len(llist)):
        if llist[i] == 'yes':
            bin_list.append(1.0)
        else:
            bin_list.append(-1.0)
    return bin_list
#bin_true: true lable
#bin_pred: predicted result
def wt_update(curr_wt, vote, bin_true, bin_pred):  # updating weights
    next_wt=[]  # updated wieght
    for i in range(len(bin_true)):
        next_wt.append(curr_wt[i]*math.e**(- vote*bin_true[i]*bin_pred[i]))
    next_weight = [x/sum(next_wt) for x in next_wt]
    return next_weight

#----------------------------------------------------------------------
def wt_append(mylist, weights):
    for i in range(len(mylist)):
        mylist[i].append(weights[i]) 
    return mylist 
# replace the last weight column with 'weight'
def wt_update_2_data(data, weight):
    for i in range(len(data)):
        data[i][-1] = weight[i]
    return data
#-----------------------------------------------------------
# indiv_pred: individual stump prediction result
# data_len
def fin_dec(indiv_pred, vote, data_len, _T):
    fin_pred = []
    for j in range(data_len):
        score = sum([indiv_pred[i][0][j]*vote[i] for i in range(_T)])
        fin_pred.append(sign_func(score))
    return fin_pred
#-----------------------------weighted error------------------     
def wt_error(true_label, predicted, weights):
    count = 0  # correct predication count
    for i in range(len(true_label)):
        if true_label[i] != predicted[i]:
            count += weights[i]
    return count
#    return count / float(len(true_label)) * 100.0
def _error(_true_lb, _pred_lb): 
    count = 0
    size = len(_true_lb)
    for i in range(size):
        if _true_lb[i] != _pred_lb[i]:
            count += 1
    return count/size 
#===================================boosting================================           
delta = 1e-8 # avoid zero training err resulintg infinite vote of a stump
T = 50   # No. of iterations
#--------------------------------------
# _T: NO. of iterations
# _delta : small item added to err to avoid infinite vote
# return: [predicted result of each stump, vote of each stump]
def ada_boost(_T, _delta, train_data):
    pred_result = list_obj(_T)   # +1,-1 dict ele
    vote_say = []
#    W_1 = np.ones(len(train_data))/len(train_data)   # wt initialization
#    train_data = wt_append(train_data, W_1)
    weights = [row[-1] for row in train_data]
    for i in range(_T):
        tree = tree_build(train_data, 1)    # train stumps
        print(tree['best_attr'])
        [pp_true, qq_pred] = label_return(train_data, tree)   # prediction result 'yes or no'
        pred_result[i].append(bin_quan(qq_pred))
        err = wt_error(pp_true, qq_pred, weights)  #+ _delta
        print(err)   # from the 2nd stump err is always clsoe to 0.5
        print(weights[0])
        vote_say.append( 0.5*math.log((1-err)/err ))   #final vote of each stump
        weights = wt_update(weights, 0.5*math.log((1-err)/err ), bin_quan(pp_true), bin_quan(qq_pred))
        train_data = wt_update_2_data(train_data, weights) 
    return [pred_result, vote_say, weights]


W_1 = np.ones(len(mylist_train))/len(mylist_train)   # wt initialization
mylist_train = wt_append(mylist_train, W_1) 
true_label_bin = bin_quan([row[-2] for row in mylist_train]) 

# =============================================================================
#def iteration_error(T_max):
#    ERR =[]
#    for t in range(1,T_max):
#        [aa_pred, bb_vote, weights] = ada_boost(t, .001, mylist_train)
#        fin_pred = fin_dec(aa_pred, bb_vote, len(mylist_train), t)
#        ERR.append(_error(true_label_bin, fin_pred))
#    return ERR
#  
#Err = iteration_error(10)       
#         
#plot.plot(Err)
#plot.ylabel('loss function value')
#plot.xlabel('No. of iterations')
#plot.title('tolerance= 0.000001, # passings =20000 ')
#plot.show()
# =============================================================================
# W_1 = np.ones(len(mylist_train))/len(mylist_train)   # wt initialization
# mylist_train = wt_append(mylist_train, W_1)      
# [aa_pred, bb_vote, weights] = ada_boost(T, delta, mylist_train)
# mylist_train = wt_update_2_data(mylist_train, weights) 
# tree_1 = tree_build(mylist_train, 1)
# [pp, qq] =label_return(mylist_train, tree_1)
# 
# 
# def compare(x,y):
#     count =0
#     for i in range(len(x)):
#         if x[i] != y[i]:
#             count += 1
#     return count
# print(compare(pp,qq))
# 
# print(wt_error(bin_quan(pp), bin_quan(qq), weights))
# =============================================================================

#fin_pred = fin_dec(aa_pred, bb_vote, len(mylist_train), T)
#true_label =bin_quan([row[-2] for row in mylist_train])  
#print(_error(true_label, fin_pred))

        
# =============================================================================
# W_1 = np.random.random(len(mylist_train))
# MM = len(mylist_train)
# for i in range(MM):
#     if i <= 1000:
#         W_1[i] = 100
#     else:
#         W_1[i] = 80
# WW = [x/sum(W_1) for x in W_1]
# #W_1 = [x/sum(W_1) for x in W_1]
# mylist_train = wt_append(mylist_train, WW)
#tree = tree_build(mylist_train, 1)
#print(tree['best_attr'])
# =============================================================================









