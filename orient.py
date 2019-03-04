#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 16:24:20 2018

@author: Vijaya Krishna, Gopal Seshadri, Raja Rajeshwari
"""

import time
import numpy as np
from collections import Counter
import sys
from scipy import stats
import math
import pickle
import pandas as pd

function = sys.argv[1]
file1 = sys.argv[2]
file2 = sys.argv[3]
model = sys.argv[4]

'''
K Nearest Neighbor
'''

#This function implements the K nearest neighbours and returns the list containing the predicted elements.
def nearest(train_list, test_list):
    
    train_matrix = np.matrix([np.array(each[2:]).astype(int) for each in train_list])
    test_matrix = np.matrix([np.array(each[2:]).astype(int) for each in test_list])
        
    orientation_array = np.array([int(each[1]) for each in train_list])
    
    distance_array = []
    K = 16
    for each in test_matrix:
        
        temp = np.argpartition(np.linalg.norm(train_matrix - each, axis = 1), K)[:K]
        distance_array.append(list(temp))
    
    

    predicted_array = [int(stats.mode(orientation_array[d], axis = None)[0]) for d in distance_array]
    
    return predicted_array

'''
Random Forest
'''

# Referred https://machinelearningmastery.com/implement-random-forest-scratch-python/ for building random forest.
# Few lines of the code that used to build Random Forest were taken from this link.

#This function creates a random forest by calling the build_tree function multiple times. It has the hyperparameters we need to create a random forest.    
def forest(train_list):
    
    train_matrix = np.array([np.array(each[2:]).astype(int) for each in train_list])
        
    orientation_array = np.array([int(each[1]) for each in train_list])
        
    n_features = int(math.sqrt(len(train_matrix[0])))
    
    s_size = 300
    
    n_trees = 300
    
    min_leaf = 10
    
    max_depth = 10
    
    forest = []
    
    column_index = []
    
    for i in range(n_trees):
        
        print(i)
        
        tree, col_index = build_tree(s_size, n_features, min_leaf, max_depth, train_matrix, orientation_array)
        
        forest.append(tree)
        
        column_index.append(col_index)
        
    return (forest, column_index)

#This function calculates the gini entropy   
def gini_entropy(groups, classes, orientation_array):

    n_instances = float(sum([len(group) for group in groups]))
	
    gini = 0.0

    for group in groups:
        size = float(len(group))
        
        if size == 0:
            continue
		
        score = 0.0

        for class_val in classes:
            
            p = [orientation_array[row] for row in group].count(class_val) / size
            score += p * p

    gini += (1.0 - score) * (size / n_instances)
	
    return gini

#This function creates splits based on the row and column index. It returns the row indexed for the left and right
def create_branches(r_index, c_index, train_matrix, row_index):
    left = []
    right = []
    
    for index in row_index:
        
        if train_matrix[index][c_index] < train_matrix[r_index][c_index]:
            left.append(index)
        
        else:
            right.append(index)
    
    return left, right

#This function finds the best split at a node.
def best_split(row_index, col_index, train_matrix, orientation_array):
    
    class_values = list(set(orientation_array))
    
    best_index, best_value, best_score, best_groups = 999, 999, 999, None
    
    for c_index in col_index:
        
        for r_index in row_index:
            
            groups = create_branches(r_index, c_index, train_matrix, row_index)
            
            gini = gini_entropy(groups, class_values, orientation_array)
        
        if gini < best_score:
            
            best_index, best_value, best_score, best_groups = c_index, train_matrix[r_index, c_index] , gini, groups
    
    return {'index':best_index, 'value':best_value, 'groups':best_groups}
            
 #This function build a tree based on the hyperparameters.
def build_tree(s_size, n_features, min_leaf, max_depth, train_matrix, orientation_array):
    
    row_index = np.random.permutation(len(train_matrix))[:s_size]
    
    col_index = np.random.permutation(len(train_matrix[0]))[:n_features]
    
    root = best_split(row_index, col_index, train_matrix, orientation_array)
     
    split_node(root, row_index, col_index, train_matrix, orientation_array, min_leaf, max_depth, 1)
    
    return root, list(col_index)
    
#This function updates the node dict based on the values of best split. It stops spliting if the count at the leaf in less
#than min_leaf or if we reach the max_depth.    
def split_node(node, row_index, col_index, train_matrix, orientation_array, min_leaf, max_depth, depth):
    
    left, right = node['groups']
    del(node['groups'])
    
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right, orientation_array)
        return
    
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left, orientation_array), to_terminal(right, orientation_array)
        return
    
    if len(left) <= min_leaf:
        node['left'] = to_terminal(left, orientation_array)
    else:
        node['left'] = best_split(left, col_index, train_matrix, orientation_array)
        split_node(node['left'], left, col_index, train_matrix, orientation_array, min_leaf, max_depth, depth+1)
        
    if len(right) <= min_leaf:
        node['right'] = to_terminal(right, orientation_array)
    
    else:
        node['right'] = best_split(right, col_index, train_matrix, orientation_array)
        split_node(node['right'], right, col_index, train_matrix, orientation_array, min_leaf, max_depth, depth+1)
    
#Finds the mode for each leaf node    
def to_terminal(group, orientation_array):

    outcomes = [orientation_array[row] for row in group]

    return max(set(outcomes), key=outcomes.count)

#Predict the label for rows from test set.
def predict_forest(node, row):

    if row[node['index']] < node['value']:
        
        if isinstance(node['left'], dict):
            return predict_forest(node['left'], row)
        else:
            return node['left']
    else:
        
        if isinstance(node['right'], dict):
            return predict_forest(node['right'], row)
        else:
            return node['right']

'''
Adaboost
'''
def train_pd_read(filename):# training data stored as pandas
    tr=pd.read_csv(filename, header=None)
    train=[]
    trainimg=[]
    for i in range(0,tr.shape[0]):
        row=tr[0][i].split(' ')    
        train.append(pd.to_numeric(row[1:]))
        trainimg.append(row[0])
    col_x=['outputY']
    for i in range(1,65):
        col_x+=['red'+str(i),'green'+str(i),'blue'+str(i)]
    pd_train=pd.DataFrame(train,columns=col_x)
    print('\n read train in pandas')
    return pd_train


def train_dict_read(filename): # training data stored as dictionary 
    tr=pd.read_csv(filename, header=None)
    row_num=tr.shape[0]
    dict_train={}
    trainimg=[]
    col_x=['outputY']
    for i in range(1,65):
        col_x+=['red'+str(i),'green'+str(i),'blue'+str(i)]

    dict_train={i : {col:0 for col in col_x} for i in range(0,row_num)}
    dict_train0={}
    dict_train90={}
    dict_train180={}
    dict_train270={}
    print('\n read train in dictionary')
    for i in range(0,row_num):
        row=tr[0][i].split(' ')[1:]
        row_spec={}
        for j in range(0,len(col_x)):
            dict_train[i][col_x[j]]=int(row[j])
            row_spec.update({col_x[j]:int(row[j])})
        if row[0]=='0':
        
            dict_train0.update({i:row_spec})
            
        if row[0]=='90':
            dict_train90.update({i:row_spec})
            
        if row[0]=='180':
            dict_train180.update({i:row_spec})
            
        if row[0]=='270':
            dict_train270.update({i:row_spec})   
            #dict_train0 has training data with orientation 0 and so on
    return(dict_train,dict_train0,dict_train90,dict_train180,dict_train270)

def test_dict_read(filename):
    test=pd.read_csv(filename, header=None)
    row_num=test.shape[0]
    image_name=[]
    dict_test={}
    col_x=['outputY']
    for i in range(1,65):
        col_x+=['red'+str(i),'green'+str(i),'blue'+str(i)]
    col_x.append('model_y') # model_y stores the predicted orientation
    dict_test={i : {col:0 for col in col_x} for i in range(0,row_num)}
    for i in range(0,row_num):
        row=test[0][i].split(' ')
        image_name.append(row[0])
        row=row[1:]
        for j in range(0,len(col_x)-1):
            dict_test[i][col_x[j]]=int(row[j])
        j+=1
        dict_test[i][col_x[j]]=270
    return(image_name,dict_test)
    
def find_features(pd_train,less_deg,more_deg):
    col=[]
    for i in range(1,65):
        col+=['red'+str(i),'green'+str(i),'blue'+str(i)]
    print('finding features')
    feature=[]
    while(len(feature))<=50: # 50 decision stumps
        col1=np.random.randint(0,192)
        col2=np.random.randint(0,192)
        pd_train['new_col']=pd_train[col[col1]]-pd_train[col[col2]]
        pd_less=pd_train[pd_train['new_col']<0] # keep the split at 0 and find the best features
        pd_more=pd_train[pd_train['new_col']>=0]
        count_less=pd_less[pd_less['outputY']==less_deg].shape[0]
        count_more=pd_more[pd_more['outputY']==more_deg].shape[0]
        if count_less>int(pd_train.shape[0]*75/200) and (count_less+count_more)>int(pd_train.shape[0]/2):                                    
            # count_less must be at least 75 percent i.e. for say 90 vs 180 combiantion there must be at least 7000 samples for 90 for split less than 0
            # count_less+count_more must be more than half of the sample size, so the decision stump is more right than it is wrong
            feature.append((col[col1],col[col2]))
    return(feature)
    
def hypo(x,feat): 
    if x[feat[0]]-x[feat[1]]<0:
        return -1
    else:
        return 1
def normalize(x):
    sum_x=sum(x)
    diff=1-sum_x
    each_add=diff/len(x)
    x=[each_x+each_add for each_x in x]    
    return(x)

def adaboost(dict_trainy,features,negdeg,posdeg):
    print('adaboost')
    N=len(list(dict_trainy.keys()))
    w=pd.DataFrame({'wt':[1/N]*N},index=list(dict_trainy.keys())) 
    a={f:0 for f in features}
    for feat in features:                
        error=0
        for i in list(w.index):            
            hx=hypo(dict_trainy[i],feat)            
            if hx==-1 and dict_trainy[i]['outputY']!=negdeg:               
                error+=w['wt'][i]        
            elif hx==1 and dict_trainy[i]['outputY']!=posdeg:
                error+=w['wt'][i]
        if error>=0.5: #ignore the classifiers with error more than 50 percent
            #i.e. it does not classify the samples incorrectly classified by the previous classfiers
            continue
        for j in list(w.index):
            hx=hypo(dict_trainy[i],feat) 
            if hx==1 and dict_trainy[i]['outputY']==posdeg:
                w['wt'][j]=w['wt'][j]*error/(1-error)
            elif hx==-1 and dict_trainy[i]['outputY']==negdeg:               
                w['wt'][j]=w['wt'][j]*error/(1-error)
        w['wt']=normalize(w['wt'])      
        
        a[feat]=math.log((1-error)/error)  
    return(a)
    
def write_model(wts,filename):    
    file=open(filename,'w')
    
    for each_0_90_180_270 in wts:
        for feat_wt in each_0_90_180_270:            
            file.write(str(feat_wt)+'->'+str(each_0_90_180_270[feat_wt])+'\n')
        file.write(';;') # the weights of each classifier is seperated by ;;
    file.close()
    
    
def model_test(test,model_wt):
    wt0_90=model_wt[0]
    wt0_180=model_wt[1]
    wt0_270=model_wt[2]
    wt90_0=model_wt[3]
    wt90_180=model_wt[4]
    wt90_270=model_wt[5]
    wt180_0=model_wt[6]
    wt180_90=model_wt[7]
    wt180_270=model_wt[8]
    wt270_0=model_wt[9]
    wt270_90=model_wt[10]
    wt270_180=model_wt[11]
    row_num=len(list(test.keys()))    
    feature0_90=list(wt0_90.keys())
    feature0_270=list(wt0_270.keys())
    feature0_180=list(wt0_180.keys())
    feature90_180=list(wt90_180.keys())
    feature90_270=list(wt90_270.keys())
    feature90_0=list(wt90_0.keys())
    feature180_270=list(wt180_270.keys())
    feature180_0=list(wt180_0.keys())
    feature180_90=list(wt180_90.keys())
    feature270_180=list(wt270_180.keys())
    feature270_0=list(wt270_0.keys())
    feature270_90=list(wt270_90.keys())
    for i in range(0,row_num):
        #res stores the weighted sum of after classification by each classifier
        res0_1=0
        res0_2=0
        res0_3=0
        res90_1=0
        res90_2=0
        res90_3=0
        res180_1=0
        res180_2=0
        res180_3=0
        res270_1=0
        res270_2=0
        res270_3=0
        
        result0_90={k:0 for k in feature0_90}        
        for feat in feature0_90:            
            result0_90[feat]=hypo(test[i],feat)*wt0_90[feat]        
        if sum(list(result0_90.values()))<0:
            res0_1=abs(sum(list(result0_90.values())))        
        
        result0_180={k:0 for k in feature0_180}        
        for feat in feature0_180:            
            result0_180[feat]=hypo(test[i],feat)*wt0_180[feat]        
        if sum(list(result0_180.values()))<0:
            res0_2=abs(sum(list(result0_180.values())))      
            
                
        result0_270={k:0 for k in feature0_270}        
        for feat in feature0_270:            
            result0_270[feat]=hypo(test[i],feat)*wt0_270[feat]        
        if sum(list(result0_270.values()))<0:
            res0_3=abs(sum(list(result0_270.values())))
            
        result90_180={k:0 for k in feature90_180}        
        for feat in feature90_180:            
            result90_180[feat]=hypo(test[i],feat)*wt90_180[feat]        
        if sum(list(result90_180.values()))<0:
            res90_1=abs(sum(list(result90_180.values())))
       
        result90_270={k:0 for k in feature90_270}        
        for feat in feature90_270:            
            result90_270[feat]=hypo(test[i],feat)*wt90_270[feat]        
        if sum(list(result90_270.values()))<0:
            res90_2=abs(sum(list(result90_270.values())))
      
        result90_0={k:0 for k in feature90_0}        
        for feat in feature90_0:            
            result90_0[feat]=hypo(test[i],feat)*wt90_0[feat]        
        if sum(list(result90_0.values()))<0:
            res90_3=abs(sum(list(result90_0.values())))
            
        result180_270={k:0 for k in feature180_270}        
        for feat in feature180_270:            
            result180_270[feat]=hypo(test[i],feat)*wt180_270[feat]        
        if sum(list(result180_270.values()))<0:
            res180_1=abs(sum(list(result180_270.values())))
            
        result180_0={k:0 for k in feature180_0}        
        for feat in feature180_0:            
            result180_0[feat]=hypo(test[i],feat)*wt180_0[feat]        
        if sum(list(result180_0.values()))<0:
            res180_2=abs(sum(list(result180_0.values())))
        
        result180_90={k:0 for k in feature180_90}        
        for feat in feature180_90:            
            result180_90[feat]=hypo(test[i],feat)*wt180_90[feat]        
        if sum(list(result180_90.values()))<0:
            res180_3=abs(sum(list(result180_90.values())))
            
        result270_180={k:0 for k in feature270_180}        
        for feat in feature270_180:            
            result270_180[feat]=hypo(test[i],feat)*wt270_180[feat]        
        if sum(list(result270_180.values()))<0:
            res270_1=abs(sum(list(result270_180.values())))
            
        result270_0={k:0 for k in feature270_0}        
        for feat in feature270_0:            
            result270_0[feat]=hypo(test[i],feat)*wt270_0[feat]        
        if sum(list(result270_0.values()))<0:
            res270_2=abs(sum(list(result270_0.values())))
        
        result270_90={k:0 for k in feature270_90}        
        for feat in feature270_90:            
            result270_90[feat]=hypo(test[i],feat)*wt270_90[feat]        
        if sum(list(result270_90.values()))<0:
            res270_3=abs(sum(list(result270_90.values())))
    # take the maximum value of the respective weighted sums
        max_val=max(res0_1,res0_2,res0_3,res90_1,res90_2,res90_3,res180_1,res180_2,res180_3,res270_1,res270_2,res270_3)
        if res0_1==max_val or res0_2==max_val or res0_3==max_val:
            test[i]['model_y']=0
        
        if res90_1==max_val or res90_2==max_val or res90_3==max_val:
            test[i]['model_y']=90
            
        if res180_1==max_val or res180_2==max_val or res180_3==max_val:
            test[i]['model_y']=180
            
        if res270_1==max_val or res270_2==max_val or res270_3==max_val:
            test[i]['model_y']=270
    return(test)
        
def read_model(filename): # read the model file
    weights=[]
    with open(filename, 'r') as myfile:
        all_data = myfile.read()    
        all_list=all_data.split(';;')    
        for data in all_list:
            wt_dict={}
            list_items=data.split('\n')    
            list_items=list_items[0:len(list_items)-1]
            for each in list_items:
                items=each.split('->')
                key1=items[0]
                key1=key1.strip("()'' ")                 
                key1=key1.split(',')                
                key=(tuple([key1[0].strip("''"),key1[1][2:]]))
                values=float(items[1])
                wt_dict[key]=values
            weights.append(wt_dict)
    return(weights)
    
def gen_output(filename,img_list,dict_test): # generate the output file
    file=open(filename,'w')
    c=0
    for row in dict_test:
        file.write(str(img_list[row])+' '+str(dict_test[row]['model_y'])+'\n')
        if(dict_test[row]['model_y']==dict_test[row]['outputY']):
            c+=1
    print('Accuracy of Adaboost --> '+str(c*100/len(list(dict_test.keys()))))

if function == "train":
    
    train = open(file1,"r")
    train_data = train.readlines()
    train_list = [f.strip().split(" ") for f in train_data]  
    train.close()
    
    if model == "nearest":

        model_file = open(file2, "w")
        for each in train_data:
            model_file.write("%s" % each) 
        model_file.close()
    
    elif model == "forest":
        
        forest = forest(train_list)
        model_file = open(file2, "wb")
        pickle.dump(forest ,model_file)                      
        model_file.close() 
        
    elif model=='adaboost':
        
        to_do=function
        train_test_fname=file1
        model_file_name=file2
    
        pd_train=train_pd_read(train_test_fname)
        dict_train,dict_train0,dict_train90,dict_train180,dict_train270=train_dict_read(train_test_fname)

        pd_train_0_90=pd_train[pd_train['outputY']!=270]
        pd_train_0_90=pd_train_0_90[pd_train_0_90['outputY']!=180]    

        pd_train_0_270=pd_train[pd_train['outputY']!=90]
        pd_train_0_270=pd_train_0_270[pd_train_0_270['outputY']!=180]   
        
        pd_train_0_180=pd_train[pd_train['outputY']!=90]
        pd_train_0_180=pd_train_0_180[pd_train_0_180['outputY']!=270]
        
        pd_train_90_270=pd_train[pd_train['outputY']!=0]
        pd_train_90_270=pd_train_90_270[pd_train_90_270['outputY']!=180]
        
        pd_train_90_180=pd_train[pd_train['outputY']!=0]
        pd_train_90_180=pd_train_90_180[pd_train_90_180['outputY']!=270]
        
        pd_train_180_270=pd_train[pd_train['outputY']!=0]
        pd_train_180_270=pd_train_180_270[pd_train_180_270['outputY']!=90]
    # create the dataset for each of the 12 combinations
        dict_train_0_90=dict_train0.copy()
        dict_train_0_90.update(dict_train90)
        dict_train_0_180=dict_train0.copy()
        dict_train_0_180.update(dict_train180)
        dict_train_0_270=dict_train0.copy()
        dict_train_0_270.update(dict_train270)
        dict_train_90_180=dict_train90.copy()
        dict_train_90_180.update(dict_train180)
        dict_train_90_270=dict_train90.copy()
        dict_train_90_270.update(dict_train270)
        dict_train_180_270=dict_train180.copy()
        dict_train_180_270.update(dict_train270)    
    # range has the two features eg. red11 and blue11 which will be compared
        range0_90=find_features(pd_train_0_90,0,90)
        range0_270=find_features(pd_train_0_270,0,270)
        range0_180=find_features(pd_train_0_180,0,180)
        range90_180=find_features(pd_train_90_180,90,180)
        range90_270=find_features(pd_train_90_270,90,270)
        range90_0=find_features(pd_train_0_90,90,0)
        range180_270=find_features(pd_train_180_270,180,270)
        range180_90=find_features(pd_train_90_180,180,90)
        range180_0=find_features(pd_train_0_180,180,0)
        range270_180=find_features(pd_train_180_270,270,180)
        range270_90=find_features(pd_train_90_270,270,90)
        range270_0=find_features(pd_train_0_270,270,0)
    # weights for each of 50 classifier for each combination
        wt0_90=adaboost(dict_train0,range0_90,0,90)    
        wt0_270=adaboost(dict_train0,range0_270,0,270)    
        wt0_180=adaboost(dict_train0,range0_180,0,180)    
        wt90_180=adaboost(dict_train90,range90_180,90,180)    
        wt90_270=adaboost(dict_train90,range90_270,90,270)
        wt90_0=adaboost(dict_train90,range90_0,90,0)    
        wt180_270=adaboost(dict_train180,range180_270,180,270)    
        wt180_0=adaboost(dict_train180,range180_0,180,0)    
        wt180_90=adaboost(dict_train180,range180_90,180,90)    
        wt270_180=adaboost(dict_train270,range270_180,270,180)    
        wt270_0=adaboost(dict_train270,range270_0,270,0)    
        wt270_90=adaboost(dict_train270,range270_90,270,90)
    
        all_wts=[wt0_90,wt0_180,wt0_270,wt90_0,wt90_180,wt90_270,wt180_0,wt180_90,wt180_270,wt270_0,wt270_90,wt270_180]
        write_model(all_wts,model_file_name)
        print('train done')
        
    elif model == "best":
        
        # This function depends on the output from other models
        nearest_file = open("nearest_output.txt","r")
        nearest_data = nearest_file.readlines()
        picture_array = [line.strip().split(" ")[0] for line in nearest_data]
        nearest_array = [int(line.strip().split(" ")[1]) for line in nearest_data]
        nearest_file.close()
        
        adaboost_file = open("adaboost_output.txt","r")
        adaboost_data = adaboost_file.readlines()
        adaboost_array = [int(line.strip().split(" ")[1]) for line in adaboost_data]
        adaboost_file.close()
        
        forest_file = open("forest_output.txt","r")
        forest_data = forest_file.readlines()
        forest_array = [int(line.strip().split(" ")[1]) for line in forest_data]
        forest_file.close()
        
        best_array = []
        for i in range(len(picture_array)):
            best_array.append(str(nearest_array[i]) + " " + str(adaboost_array[i]) + " " + str(forest_array[i]) + "\n")
        
        model_file = open(file2, "w")
        for each in best_array:
            model_file.write("%s" % each) 
        model_file.close()
        
elif function == "test":
    test = open(file1,"r")
    test_data = test.readlines()
    test_list = [f.strip().split(" ") for f in test_data]  
    test.close()

    actual_array = [int(each[1]) for each in test_list]
    picture_array = [each[0] for each in test_list]
    
    
    if model == "nearest":

        train = open(file2,"r")
        train_data = train.readlines()
        train_list = [f.strip().split(" ") for f in train_data]  
        train.close()

        predicted_array = nearest(train_list, test_list)
        
        count = 0
        for i in range(len(actual_array)):
            if actual_array[i] == predicted_array[i]:
                count += 1
        print("Accuracy of Nearest --> ", (count * 100)/ len(actual_array))
        
        output_file = open("nearest_output.txt", "w")
        for i in range(len(picture_array)):
            line = str(picture_array[i]) + " " + str(predicted_array[i]) + "\n"
            output_file.write(line) 
        output_file.close()
        
    elif model == "forest":
        
        model_file = open(file2, "rb")
        forest = pickle.load(model_file)
        test_matrix = np.matrix([np.array(each[2:]).astype(int) for each in test_list])

        pred_dict = {}
        
        tag = 1
        for row in test_matrix:
            preds = []
            row = np.array(row).reshape(192, )

            for i in range(len(forest[0])):   
                pred = predict_forest(forest[0][i], row)
                preds.append(pred)
            pred_dict[tag] = preds
            tag += 1
        
        predicted_array = [int(stats.mode(pred_dict[key], axis = None)[0]) for key in pred_dict.keys()]
        
        count = 0
        for i in range(len(actual_array)):
            if actual_array[i] == predicted_array[i]:
                count += 1
        print("Accuracy of Forest --> ", (count * 100)/ len(actual_array))
        
        output_file = open("forest_output.txt", "w")
        for i in range(len(picture_array)):
            line = str(picture_array[i]) + " " + str(predicted_array[i]) + "\n"
            output_file.write(line) 
        output_file.close()
        
    elif model == 'adaboost':
        to_do=function
        train_test_fname=file1
        model_file_name=file2
        img_lst,dict_test=test_dict_read(train_test_fname)   
        model_wt=read_model(model_file_name)
        new_test=model_test(dict_test,model_wt)
        gen_output('adaboost_output.txt',img_lst,new_test)
    
    elif model == "best":
        
        best_model = open(file2,"r")
        model_data = best_model.readlines()
        model_list = [each.strip().split(" ") for each in model_data]

        best_model.close()
        
        predicted_array = [int(stats.mode(each, axis = None)[0]) for each in model_list]  

        count = 0
        for i in range(len(actual_array)):
            if actual_array[i] == predicted_array[i]:
                count += 1
        print("Accuracy of Best --> ", (count * 100)/ len(actual_array))
        
        output_file = open("best_output.txt", "w")
        for i in range(len(picture_array)):
            line = str(picture_array[i]) + " " + str(predicted_array[i]) + "\n"
            output_file.write(line) 
        output_file.close()
