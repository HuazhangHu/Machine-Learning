#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import svm
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

clf_svm=svm.SVC(kernel='rbf',C=2.0)#C越大边界越复杂


#########################################################
### your code goes here ###
def classify():
    clf=GaussianNB()
    t0=time()
    clf.fit(features_train,labels_train)
    print("training time:", round(time() - t0, 3), "s")
    t1=time()
    pred=clf.predict(features_test)
    print('predict time:',round(time()-t1,3),'s')
    accuracy=accuracy_score(labels_test,pred)
    return accuracy

accuracy=classify()
print(accuracy)

#########################################################


