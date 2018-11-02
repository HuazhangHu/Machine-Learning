import logRegres
from numpy import *

data=open('C:\\Users\ASUS\Desktop\classifyTest.txt')
dataLine=data.readlines()[0].strip().split('\t')
dataArr=[]
for i in range(21):
    dataArr.append(float(dataLine[i]))
print(dataArr)
labels=["未死","已死"]
logRegres.classifyClass(dataArr,labels)

