import math
import statistics
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
            
#---------replacing 'unknown' in train data-------------------
major_label=[]
for i in range(16):
    majority_labels = [row[i] for row in mylist_train if row[i]!= 'unknown']
    lb = max(set(majority_labels), key=majority_labels.count)
    major_label.append(lb)
    
for i in range(len(mylist_train)):
    for j in range(16):
        if mylist_train[i][j] == 'unknown':
            mylist_train[i][j] = major_label[j]
#print(mylist_train[0])
            
#--------------------test data process--------------------------------
with open('test.csv',mode='r') as f:
    myList_test=[];
    for line in f:
        terms=line.strip().split(',') # 7*N matrix
        myList_test.append(terms)

mylist_test = str_2_flo(myList_test)
for i in obj:
    obj[i] = statistics.median([row[i] for row in mylist_test])
    
for row in mylist_test:
    for i in obj:
        if row[i] >= obj[i]:
            row[i] = 'yes'
        else:
            row[i] = 'no'
            
#---------replacing 'unknown' in train data-------------------
major_label_test=[]
for i in range(16):
    majority_labels = [row[i] for row in mylist_train if row[i]!= 'unknown']
    lb = max(set(majority_labels), key=majority_labels.count)
    major_label_test.append(lb)
    
for i in range(len(mylist_test)):
    for j in range(16):
        if mylist_test[i][j] == 'unknown':
            mylist_test[i][j] = major_label_test[j]
#==============================================================================
Attr_dict ={'age':['yes','no'],
             'job':['admin.','unknown','unemployed','management','housemaid','entrepreneur','student','blue-collar','self-employed','retired','technician','services'],
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
    return obj

def create_list_0(attr):
    obj={}
    for attr_val in attr:
        obj[attr_val]=0
    return obj
#----------------------------------------------------------------------------
def gini_index(groups, classes):
    # totoal number of instances
    N_ins= float(sum([len(groups[attr_val]) for attr_val in groups])) # attr_val--str type
    gini = 0.0
    for attr_val in groups:   # traverse diff. braches
        size = float(len(groups[attr_val]))
		# avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:  # label values
            p = [row[-1] for row in groups[attr_val]].count(class_val) /size
            score += p * p	       
        gini += (1.0 - score) * (size / N_ins)
    return gini          

def exp_entropy(groups, classes):
    N_ins= float(sum([len(groups[attr_val]) for attr_val in groups])) 
    exp_ent = 0.0
    for attr_val in groups:
        size = float(len(groups[attr_val]))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in groups[attr_val]].count(class_val) / size
            if p==0:
                temp=0
            else:
                temp=p*math.log2(1/p)
            score +=temp 
        exp_ent += score* (size / N_ins)
    return exp_ent          
#----------------------------------------------------------------------------      
# ME=1-max(p_1,p_2,...,p_V)
def m_error(groups, classes):
    N_ins = float(sum([len(groups[attr_val]) for attr_val in groups]))
    m_err = 0.0
    for attr_val in groups:
        size = float(len(groups[attr_val]))
        if size == 0:
            continue
        score = 0.0
        temp=0
        for class_val in classes:
            p = [row[-1] for row in groups[attr_val]].count(class_val) / size
            temp=max(temp,p)
            score=1-temp
        m_err += score* (size / N_ins)
    return m_err
#----------------------------------------------------------------------------
def data_split(attr, dataset):
    branch_obj=create_list(attr)  # this may result in empty dict elements 
    for row in dataset:
        for attr_val in Attr_dict[attr]:
           if row[pos(attr)]==attr_val:
               branch_obj[attr_val].append(row)
    return branch_obj
#----------------------------------------------------------------------------
def find_best_split(dataset):
    if dataset==[]:
        return 
    label_values = list(set(row[-1] for row in dataset)) 
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
    majority_labels = [row[-1] for row in group]
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
    if not if_node_divisible(node['best_groups']):  #only one non-empty branch
        # what if all elements in node 
        for key in node['best_groups']:
            if  node['best_groups'][key]!= []: #and ( not isinstance(node['best_groups'][key],str)): 
                 node[key] = leaf_node_label(node['best_groups'][key])
            else:
                node[key] = leaf_node_label(sum(node['best_groups'].values(),[])) 
        return
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
def tree_build(train, max_depth):
	root = find_best_split(train)
	child_node(root, max_depth, 1)
	return root
#----------------------------------------------------------------------------
#test if an instance belongs to a node recursively
def label_predict(node, inst):
    if isinstance(node[inst[pos(node['best_attr'])]],dict):
        return label_predict(node[inst[pos(node['best_attr'])]],inst)
    else:
        return node[inst[pos(node['best_attr'])]]   #leaf node
#-------------------------------------      
def error(true_label, predicted):
    count = 0  # correct predication count
    for i in range(len(true_label)):
        if true_label[i] != predicted[i]:
            count += 1
    return count / float(len(true_label)) * 100.0


#===================================prediction================================
tree=tree_build(mylist_train, 1)


 
true_label = []
pred_seq = []   # predicted sequence
for row in mylist_test:
    true_label.append(row[-1])
    pre = label_predict(tree, row)
    pred_seq.append(pre)

print(error(true_label, pred_seq))




