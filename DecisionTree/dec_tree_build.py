# -*- coding: utf-8 -*-

# read training data
import math
with open('train.csv',mode='r') as f:
    myList=[];
    for line in f:
        terms=line.strip().split(',') # 7*N matrix
        myList.append(terms)
   # print(myList)
   
# create an object containing n empty lists
def create_list(n):
    obj={}
    for i in range(n):
        obj[i]=[]
    return obj

# column index: 0 1 2 3 4 5 6
#columns:buying,maint,doors,persons,lug_boot,safety,label
# No. values: 4,4,4,3,3,3,label4
# Description--
# split the dataset into multiple subsets with specific values of a given attribute
# return: obj contains V lists, each corresponing to a value v in V
def data_split(index, dataset):
    Num_index=[4, 4, 4, 3, 3, 3] # No. of values for each attribute & label
    obj=create_list(Num_index[index])
    if index==0:  # buying 
        for row in dataset:
           if row[index]=='vhigh':
               obj[0].append(row)
               #obj['vhigh'].append(row)
           if row[index]=='high':
               obj[1].append(row)
           if row[index]=='med':
               obj[2].append(row)
           if row[index]=='low':
               obj[3].append(row) 
        return obj
    if index==1: # maint 
        for row in dataset: 
           if row[index]=='vhigh':
              obj[0].append(row)
           if row[index]=='high':
              obj[1].append(row)
           if row[index]=='med':
              obj[2].append(row)
           if row[index]=='low':
              obj[3].append(row)
        return obj
    if index==2: # doors         
        for row in dataset: 
           if row[index]=='2':
              obj[0].append(row)
           if row[index]=='3':
              obj[1].append(row)
           if row[index]=='4':
              obj[2].append(row)
           if row[index]=='5more':
              obj[3].append(row)
        return obj
    if index==3: # persons        
        for row in dataset: 
           if row[index]=='2':
              obj[0].append(row)
           if row[index]=='4':
              obj[1].append(row)
           if row[index]=='more':
              obj[2].append(row)
        return obj
    if index==4: # lug_boot        
        for row in dataset: 
           if row[index]=='small':
              obj[0].append(row)
           if row[index]=='med':
              obj[1].append(row)
           if row[index]=='big':
              obj[2].append(row)
        return obj
    if index==5: # safty        
        for row in dataset: 
           if row[index]=='low':
              obj[0].append(row)
           if row[index]=='med':
              obj[1].append(row)
           if row[index]=='high':
              obj[2].append(row)
        return obj

#test correctness
#only works for the ex.txt dataset
Num_index=[4, 4, 4, 3, 3, 3]
def data_split_ex(index, dataset):
    obj=create_list(Num_index[index])
    if index==0:  # buying 
        for row in dataset:
           if row[index]=='1':
               obj[0].append(row)
           if row[index]=='2':
               obj[1].append(row)
           if row[index]=='3':
               obj[2].append(row)
           if row[index]=='4':
               obj[3].append(row) 
        return obj
    if index==1:
        for row in dataset:
           if row[index]=='1':
               obj[0].append(row)
           if row[index]=='2':
               obj[1].append(row)
           if row[index]=='3':
               obj[2].append(row)
           if row[index]=='4':
               obj[3].append(row) 
        return obj
        
 # Calculate gini index
 # classes: set of labels
 # groups: subsets after splitting--object containing multiple lists
 # For simplicity, we directly use Gini index instead of Gini gain
def gini_index(groups, classes):
    n_instances = float(sum([len(groups[group]) for group in groups]))
	# sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(groups[group]))
		# avoid divide by zero
        if size == 0:
            continue
        score = 0.0
		# score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in groups[group]].count(class_val) /size
            score += p * p	       
        gini += (1.0 - score) * (size / n_instances)
    return gini          


# calculate expexted entropy instead of info. gain
def exp_entropy(groups, classes):
    n_instances = float(sum([len(groups[group]) for group in groups]))
    exp_ent = 0.0
    for group in groups:
        size = float(len(groups[group]))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in groups[group]].count(class_val) / size
            if p==0:
                temp=0
            else:
                temp=p*math.log2(1/p)
            score +=temp 
		# weight the group score by its relative size
        exp_ent += score* (size / n_instances)
    return exp_ent          
        
# calculate majority error
# ME=1-max(p_1,p_2,...,p_V)
def m_error(groups, classes):
    n_instances = float(sum([len(groups[group]) for group in groups]))
    m_err = 0.0
    for group in groups:
        size = float(len(groups[group]))
        if size == 0:
            continue
        score = 0.0
        temp=0
        for class_val in classes:
            p = [row[-1] for row in groups[group]].count(class_val) / size
            temp=max(temp,p)
            score=1-temp
        m_err += score* (size / n_instances)
    return m_err


# Description
# Find the best available attribute to split the dataset
# returns an (node) object: [best_attribute_index, best(smallest)_metric(gini/Ent/ME)_value, groups_of_subsets]
    # input type: list
def find_best_split(dataset):
    if dataset==[]:
        return
    label_values = list(set(row[-1] for row in dataset)) 
    #print(label_values)
    #find all the possible label values in the input dataset
    #last column is the label
    best_attri_index=1000
    best_metric_vlaue=1000
    best_groups=None  # Nontype object
    Metric=[] # store gini metric for splitting based on attribute index
    for index in range(len(dataset[0])-1):
        #loop all the attribute indices
        #groups = data_split_ex(index, dataset) # test ex.txt
        groups = data_split(index, dataset)  # type: dict (obj)
        gini = gini_index(groups, label_values)  #used gini as metric here
        Metric.append(gini)
    best_metric_value =min(Metric)
    best_attri_index = Metric.index(min(Metric)) 
    #best_groups = data_split_ex(best_attri_index, dataset) #test ex.txt
    best_groups = data_split(best_attri_index, dataset)  # type: object
    return {'best_attri':best_attri_index, 'best_metric_val': best_metric_value, 'best_groups':best_groups}
# =============================================================================
#test of ex.txt
# best_split=find_best_split(myList)
# print(best_split)
# print(len(myList[0]))
# =============================================================================
# a,b,split=find_best_split(myList)
# print(split)
# print(leaf_node_label(split[3]))
# =============================================================================
    
# Determine the label value for a leaf node
# returns the majority label within 'group'
def leaf_node_label(group):
    if group==[]:
        return
    majority_labels = [row[-1] for row in group]
    return max(set(majority_labels), key=majority_labels.count)

#judge if the if there is only one resulting branch after splitting based on the best feature
# return k if k out of len(branch_list) liest elements are non-empty
# Type(brach_list)=list
def if_node_divisible(branch_list):
    score=0
    temp=0
    non_empty_indices=[i for i,x in enumerate(branch_list) if not (not x)] # return the index of non-empty element
    for i in range(len(branch_list)):
        temp=int(not branch_list[i])    # temp==1 if element if empty and 0 if not empty
        score=score+temp
    score=len(branch_list)-score
    non_empty_list=branch_list[non_empty_indices[0]]
    return [score,non_empty_indices,non_empty_list]
# return the 1 st non-empty element (list)
        
    
    

# recursively create child nodes for a node until reaching leaf node 
# node['']
# result: create new child nodes or leaf nodes
    
def dict_2_list(dictionary):
    best_list=[]
    for key, value in dictionary.items():
            temp = [key,value]
            best_list.append(value)
    return best_list

def child_node(node, max_depth, curr_depth):
    #Num_index=[4, 4, 4, 3, 3, 3]
    if node['best_attri']==0 or node['best_attri']==1 or node['best_attri']==2:# attributes deternmine the number of braches (3 or 4)
        # convert dict type to list
        best_list=dict_2_list(node['best_groups'])          
        bran_0, bran_1,bran_2,bran_3 = best_list 
        #obj['best_groups'] represents the best split of a node
        # each brach (brch) represents a split corresponding to a specific attri. value
        del(node['best_groups'])
	    # check for the case no split is needed-- best attri. only takes on one value 
        [score ,index,non_empty_list]=if_node_divisible([ bran_0, bran_1,bran_2,bran_3])  # only one out of 4 branches are non-empty
        if score==1:
            node['bran_0'] = node['bran_1'] =node['bran_2']=node['bran_3'] =leaf_node_label(bran_0 + bran_1 + bran_2 + bran_3) 
            return
        if curr_depth >= max_depth:
            node['bran_0'], node['bran_1'],node['bran_2'],node['bran_3'] = leaf_node_label(bran_0), leaf_node_label(bran_1),leaf_node_label(bran_2),leaf_node_label(bran_3)
            #stop with leaf node is max depth is reached
            return
        # extend branch 0
#        if node['bran_0']!=[]:
            node['bran_0'] = find_best_split(bran_0)
            child_node(node['bran_0'], max_depth, depth+1) # recursive calling
	    # extend branch 1
#        if node['bran_1']!=[]:
            node['bran_1'] = find_best_split(bran_1)
            child_node(node['bran_1'], max_depth, depth+1)
        # extend branch 2
#        if node['bran_2']!=[]:
            node['bran_2'] = find_best_split(bran_2)
            child_node(node['bran_2'], max_depth, depth+1)
        # extend branch 3
#        if node['bran_3']!=[]:
            node['bran_3'] = find_best_split(bran_3)
            child_node(node['bran_3'], max_depth, depth+1)
        # end here for attr. with 4 values
    if node['best_attri']==3 or node['best_attri']==4 or node['best_attri']==5:# attributes deternmine the number of braches (3 or 4)
        best_list=dict_2_list(node['best_groups'])          
        bran_0, bran_1, bran_2 = best_list   
        del(node['best_groups'])
        [score ,index,non_empty_list]=if_node_divisible([ bran_0, bran_1, bran_2])  # only one out of 3 branches are non-empty
        if score==1:
            node['bran_0'] = node['bran_1'] =node['bran_2'] =leaf_node_label(bran_0 + bran_1 + bran_2) 
            return
        if curr_depth >= max_depth:
            node['bran_0'], node['bran_1'],node['bran_2'] = leaf_node_label(bran_0), leaf_node_label(bran_1),leaf_node_label(bran_2)
            #stop with leaf node is max depth is reached
            return
        # extend branch 0
        
            node['bran_0'] = find_best_split(bran_0)
            child_node(node['bran_0'], max_depth, depth+1)
	    # extend branch 1
#        if node['bran_1']!=[]:
            node['bran_1'] = find_best_split(bran_1)
            child_node(node['bran_1'], max_depth, depth+1)
        # extend branch 2
#        if node['bran_2']!=[]:
            node['bran_2'] = find_best_split(bran_2)
            child_node(node['bran_2'], max_depth, depth+1)
        # function ends here
        
# Build the whole tree
def tree_build(train, max_depth):
	root = find_best_split(train)
	child_node(root, max_depth, 1)
	return root

# main()
tree_build(myList,3)     # build the tree