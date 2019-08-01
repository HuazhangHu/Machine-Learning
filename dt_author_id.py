#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from tools.email_preprocess import preprocess
from sklearn import tree
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
print(len(features_train),len(features_train[0]))#打印训练集数据个数和特征数
min_samples_split=40
# print(type(labels_train))
#########################################################
### your code goes here ###
clf=tree.DecisionTreeClassifier(min_samples_split=min_samples_split)
t0=time()
clf.fit(features_train,labels_train)
print('training time:',round(time()-t0),'s')
pred=clf.predict(features_test)
accuracy=accuracy_score(labels_test,pred)
print(accuracy)
# print(len(features_train[0]))
#########################################################


