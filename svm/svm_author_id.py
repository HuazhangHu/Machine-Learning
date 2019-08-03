#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
print(round(len(features_train)/100),round(len(labels_train)/100))
features_train = features_train[:len(features_train)//100]
labels_train = labels_train[:len(labels_train)//100]
C=10000
kennel='rbf'#这个参数效果太好了
gamma='auto'



clf=SVC(kernel=kennel,C=C,gamma=gamma)
t0=time()
clf.fit(features_train,labels_train)
print('training time',round(time()-t0),'s')
pred=clf.predict(features_test)
# print(pred[:10],pred[:26],pred[:50])
print(accuracy_score(labels_test,pred))



#########################################################
### your code goes here ###


#########################################################


