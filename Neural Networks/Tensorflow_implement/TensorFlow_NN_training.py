# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:38:15 2019

@author: Larry Cheung
"""

from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np


"""--------------------------- data pre-processing --------------------------------"""
with open('train.csv',mode='r') as f:
    myList_train=[]
    for line in f:
        terms=line.strip().split(',') # 7*N matrix
        myList_train.append(terms)
with open('test.csv',mode='r') as f:
    myList_test = []
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
    temp = data
    for i in range(len(data)):
        temp[i][-1] = 1.0
    for i in range(len(data)):
        temp[i].append(label[i])
    return temp

# convert label {0,1} to {-1,1}
def polar_label(data):
    temp = data
    for i in range(len(data)):
        temp[i][-1] = 2*data[i][-1]-1
    return temp
               
mylist_train = str_2_flo(myList_train)  #convert to float  types data 
mylist_test = str_2_flo(myList_test) 
train_data = add_cons_feature(mylist_train)
test_data = add_cons_feature(mylist_test)
#train_data = add_cons_feature(polar_label(mylist_train))
#test_data = add_cons_feature(polar_label(mylist_test))
train_len = len(train_data)   # NO. of samples
test_len = len(test_data)
dim_s = len(train_data[0]) -1   # sample dimension = 5 (including constant feature)
train_labels = np.array([row[-1] for row in train_data ])
test_labels = np.array([row[-1] for row in test_data ])
train_data_unlabeled = [np.array(row[0:dim_s],ndmin =2) for row in train_data]
test_data_unlabeled = [np.array(row[0:dim_s],ndmin =2) for row in test_data]

"""------------------------------- end data processing ---------------------------"""
train_data_array = np.array(train_data_unlabeled)
test_data_array = np.array(test_data_unlabeled)
print('train data dimension:', train_data_array.shape)
print('test data dimension:', test_data_array.shape)

width = 100    # hiden layer width
# he_normal initializer
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1,5)),
    keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer= 'he_normal',
                bias_initializer='zeros'),
    keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer= 'he_normal',
                bias_initializer='zeros'),
    keras.layers.Dense(2, activation=tf.nn.softmax)  
    # outputs two confidence levels for label = 0 and label =1
])
    
# =============================================================================
# #  Xavier normal initializer, i.e., the glorot_normal initializer 
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(1,5)),
#     keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal',
#                 bias_initializer='zeros'),
#     keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal',
#                 bias_initializer='zeros'),
#     keras.layers.Dense(2, activation=tf.nn.softmax)  
# ])
# =============================================================================
    

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# ----------   model training  ----------------
model.fit(train_data_array, train_labels, epochs= 20)

# -----------------------  prediction------------------------
test_loss, test_acc = model.evaluate(test_data_array, test_labels)


