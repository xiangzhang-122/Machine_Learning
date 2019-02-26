import math
with open('train.csv',mode='r') as f:
    myList_train=[];
    for line in f:
        terms=line.strip().split(',') # 7*N matrix
        myList_train.append(terms)
        
Attr_dict={'buying': ['vhigh', 'high', 'med', 'low'], 
           'maint':  ['vhigh', 'high', 'med', 'low'],
           'doors':  ['2', '3', '4', '5more' ],
           'persons': ['2', '4', 'more'],
           'lug_boot':[ 'small', 'med', 'big'],
           'safety': ['low', 'med', 'high']}
#----------------------------------------------------------------------------
def pos(attr):
    pos=0
    if attr=='buying':
        pos=0
    if attr=='maint':
        pos=1
    if attr=='doors':
        pos=2
    if attr=='persons':
        pos=3
    if attr=='lug_boot':
        pos=4
    if attr=='safety':
        pos=5
    return pos
#----------------------------------------------------------------------------
# create obj with multipe empty lists   
def create_list(attr):
    obj={}
    for attr_val in Attr_dict[attr]:
        obj[attr_val]=[]
    return obj
# create obj with multipe zero elements (int type)
def create_list_0(attr):
    obj={}
    for attr_val in attr:
        obj[attr_val]=0
    return obj
#----------------------------------------------------------------------------
# groups: diff. braches specidifed by attr values (dict type)
# classes : set of possible labels
# Weighted Gini index is used here, instead of Gini gain
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

# calculate expexted entropy instead of info. gain
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
		# weight the group score by its relative size
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
    #print(label_values)
    metric_obj = create_list_0(Attr_dict)
    for attr in Attr_dict:
#        metric_obj = create_list_0(attr)
        groups = data_split(attr, dataset)
        metric_obj[attr] =  gini_index(groups, label_values)             # change metric here
    best_attr = min(metric_obj, key=metric_obj.get)
    best_groups = data_split(best_attr, dataset)  
#    exist_attr_values = 
    # each group element is associated with an attr. value
    return {'best_attr':best_attr, 'best_groups':best_groups}
#----------------------------------------------------------------------------    
# returns the majority label within 'group' (list type)
def leaf_node_label(group):
#    if group == []:
#        return None   # neither dict type nor str type
    majority_labels = [row[-1] for row in group]
    return max(set(majority_labels), key=majority_labels.count)
#----------------------------------------------------------------------------
#input-- dict type
# if there is only one non-empty branch, then return False(not divisible)
def if_node_divisible(branch_obj):
    non_empty_indices=[key for key in branch_obj if not (not branch_obj[key])]
    if len(non_empty_indices)==1:
        return False
    else:
        return True
#-------------------------------------------------------------------------
def child_node(node, max_depth, curr_depth):
    if not if_node_divisible(node['best_groups']):  #only one non-empty branch
        # 'best_groups' may contain empty branches
        for key in node['best_groups']:
            if  node['best_groups'][key]!= []: #and ( not isinstance(node['best_groups'][key],str)):
#                node[key] = leaf_node_label(sum(node['best_groups'].values(),[]))  
                 node[key] = leaf_node_label(node['best_groups'][key])
            else:
                node[key] = leaf_node_label(sum(node['best_groups'].values(),[])) 
            # concatenate values (lists) corresponding to diff. attr. values 
                #node['best_attr'] = leaf_node_label(sum(node['best_groups'].values(),[])) 
        return
    if curr_depth >= max_depth:
        for key in node['best_groups']:
            if  node['best_groups'][key]!= []: #and ( not isinstance(node['best_groups'][key],str)):
                # extract nonempty branches
                node[key] = leaf_node_label(node['best_groups'][key])   
                #what if branch_obj[''] is empty? 
            else:
                node[key] = leaf_node_label(sum(node['best_groups'].values(),[]))
        return  
    for key in node['best_groups']:
        if node['best_groups'][key]!= []: #and ( not isinstance(node['best_groups'][key],str)):
            node[key] = find_best_split(node['best_groups'][key]) 
            child_node(node[key], max_depth, curr_depth + 1)
        else:
            node[key] = leaf_node_label(sum(node['best_groups'].values(),[]))
           #further dataset splitting
           # node[key] is dict type if non-leaf and str type if leaf node
           
#----------------------------------------------------------------------------
def tree_build(train, max_depth):
	root = find_best_split(train)
	child_node(root, max_depth, 1)
	return root
#----------------------------------------------------------------------------

#temp = 0
#for row in myList:
#    if row[-1] == 'good':
#        temp=temp+1
#print(temp)   
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



with open('test.csv',mode='r') as f:
    myList_test=[];
    for line in f:
        terms=line.strip().split(',') # 7*N matrix
        myList_test.append(terms)

tree=tree_build(myList_train, 6)
 
true_label = []
pred_seq = []   # predicted sequence
for row in myList_train:
    true_label.append(row[-1])
    pre = label_predict(tree, row)
    pred_seq.append(pre)

print(error(true_label, pred_seq))





    





    
    
    
    
    
    
    
    
    
    
   


