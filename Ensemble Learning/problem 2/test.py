 
# HW 2 Problem 2. boosting & bagging algorithm
import math
import statistics
import numpy as np
import matplotlib.pyplot as plot
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
# weighted expected entropy
# groups:  dict type
# classes: ther set of labels contained in 'groups'
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
    if isinstance(node[inst[pos(node['best_attr'])]], dict):
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
        if err == 0.5:
            err+= delta
        print(err)   # from the 2nd stump err is always clsoe to 0.5
        print(weights)
        vote_say.append( 0.5*math.log((1-err)/err ))   #final vote of each stump
        weights = wt_update(weights, 0.5*math.log((1-err)/err ), bin_quan(pp_true), bin_quan(qq_pred))
        train_data = wt_update_2_data(train_data, weights) 
    return [pred_result, vote_say, weights]
    
Attr_dict ={'age':['h','m', 'l'],
             'job':['g','b']}
def pos(attr):
    if attr=='age':
        pos= 0
    if attr=='job':
        pos= 1
    return pos  

data_list = [['h','g','yes', 0.2], ['l','b','yes', 0.2],['l','g','no',0.2] ,['l','b','no', 0.2],
             ['h','b','no', 0.2] ]
true_label_bin =bin_quan([row[-2] for row in data_list]) 






# =============================================================================
# def iteration_error(T_max):
#     ERR =[]
#     for t in range(1,T_max):
#         [aa_pred, bb_vote, weights] = ada_boost(t, .001, data_list)
#         fin_pred = fin_dec(aa_pred, bb_vote, len(data_list), t)
#         ERR.append(_error(true_label_bin, fin_pred))
#     return ERR
#  
# Err = iteration_error(30)       
#         
# plot.plot(Err)
# plot.ylabel('loss function value')
# plot.xlabel('No. of iterations')
# plot.title('tolerance= 0.000001, # passings =20000 ')
# plot.show()
# =============================================================================

 







