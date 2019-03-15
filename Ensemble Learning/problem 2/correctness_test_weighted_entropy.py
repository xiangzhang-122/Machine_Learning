# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:16:43 2019

@author: Larry Cheung
"""
import math
import numpy as np

## test the correctness of the weighted entropy
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
            p = sum([row[-1] for row in groups[attr_val] if row[-2] == class_val])/q  
            if p==0:
                temp=0
            else:
                temp=p*math.log2(1/p)
            score +=temp 
#        exp_ent += score* (size / N_ins)
        exp_ent += score* sum([row[-1] for row in groups[attr_val]])/Q #total weights of a subset
    return exp_ent  

group ={0:[[0,0,1,0,0,1/7+1/8],[0,1,0,0,0,1/7],[0,0,1,1,1,1/7 - 1/8] ,[0,1,1,0,0,1/7],[0,1,0,1,0,1/7] ]  ,
        1:[[1,0,0,1,1,1/7], [1,1,0,0,0,1/7]] }        # x_1 


classes = [0, 1]
print(exp_entropy(group, classes))
   
#dataset = [[0,0,1,0],[0,1,0,0],[0,0,1,1] ,[0,1,1,0],[0,1,0,1],
#           [1,0,0,0], [1,1,0,0] ]
