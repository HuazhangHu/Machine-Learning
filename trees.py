from math import log
import operator

def createDataSet():#够着数据
    dataSet=[[1,1,'yes'],
             [1, 1, 'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels


def calcShannonEnt(dataSet):#计算香农熵
    numEntries=len(dataSet)
    labelCounts={}
    for featVec in dataSet:
        currentLabels=featVec[-1]#最后一列的类别作为键值
        if currentLabels not in  labelCounts.keys():
            labelCounts[currentLabels]=0
        labelCounts[currentLabels]+=1#统计出线的次数
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries#计算频率，频率=概率
        shannonEnt-=prob*log(prob,2)
    return shannonEnt

def splitDataSet(dataSet,axis,value):#按照给定特征划分数据集，dataSet为数据集，axis为划分的特征（0表示第一列），value为提取出来的值，最终划分的结果会剔除value
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatvec=featVec[:axis]#提取前面的元素构成列表[...,self]
            reducedFeatvec.extend(featVec[axis+1:])#提取后面的元素[self+1,...,end]加入列表，extend()函数将列表中的元素以的单个元素的形式加入列表
            retDataSet.append(reducedFeatvec)#矩阵
    return retDataSet

def chooseBestFeatureToSplit(dataSet):#选择最好的数据划分方式，即得到以哪个列划分能将数据划分的最清晰
    numFeatures=len(dataSet[0])-1
    baseEntropy=calcShannonEnt(dataSet)#旧的香农熵
    bestInfoGain=0.0#信息增益
    bestFeature=-1
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet]#循环取出dataset中的每个value放入列表
        uniqueVals=set(featList)
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)#以不同的特征循环分割
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)#得到新的香农熵
        infoGain=baseEntropy-newEntropy#新的信息增益
        if (infoGain>bestInfoGain):
            bestFeature=i
            bestInfoGain=infoGain#通过不断比较得到使信息增益最大的特征索引
    return bestFeature

def majorityCnt(classList):#统计得出出现次数最的分类的名称
    classCount={}
    for vote in classList:#统计次数
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)#降序排列，取出出现次数最多的分类名称
    return sortedClassCount[0][0]

def createTree(dataSet,labels):#创建决策树,返回嵌套列表
    classList=[example[-1] for example in dataSet]#循环取出类标签（最后一列）放入列表中
    if classList.count(classList[0])==len(classList):#如果第一个标签的数目=所以标签数目，即所有标签完全相同时，直接返回该标签
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)#如何dataSet中只含有标签，则统计标签的个数选出出现次数最多分类名称
    #开始创建树
    bestFeat=chooseBestFeatureToSplit(dataSet)#最好的分割列
    bestFeatLabel=labels[bestFeat]#最好分割列对应的标签
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])#递归中不断删除最好的分割特征对应的标签
    featValues=[example[bestFeat] for example in dataSet]#循环取出datatSet中bestFeat(最好的分割列）对应的值放入列表
    uniqueVals=set(featValues)#转换为集合
    for value in uniqueVals:
        subLabels=labels[:]#把所有标签赋给他
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree#最终形成嵌套字典

def classify(inputTree,featLabels,testVec):#决策树的分类函数
    firstStr = list(inputTree.keys())[0]#dict_key can not be index
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

#将分类器存储在硬盘上
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)








