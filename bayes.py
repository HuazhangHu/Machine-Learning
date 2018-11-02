from numpy import *
import re
import random

def loadDataSet():#定义的侮辱言论库
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],#非侮辱
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],#侮辱，后面循环
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]#1代表有侮辱性言论，0代表非侮辱性言论
    return postingList,classVec

def createVocabList(dataSet):#传入的是词汇矩阵。这步相当于统计共有哪些词汇
    vocabSet = set([])#创造一个空集
    for document in dataSet:
        vocabSet = vocabSet | set(document) #创造两个集合的并集，通过循环使dataSet的聚合不断合并
    return list(vocabSet)#返回的是总的词汇列表

def setOfWords2Vec(vocabList, inputSet):#转换为文档向量,vacablist 词汇列表,inputset 文档词条
    returnVec = [0]*len(vocabList)#创建一个与文本长度相等的零向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1#在vocablist中找到每个单词的下标
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec#返回向量

def trainNB0(trainMatrix,trainCategory):#计算概率，trainmatrix文档矩阵，trainCategrory标签向量
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)#是1的概率
    p0Num = ones(numWords); p1Num = ones(numWords)#单位矩阵
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:#如果是侮辱言论
            p1Num += trainMatrix[i]#数目+1
            p1Denom += sum(trainMatrix[i])#侮辱词汇总和增加
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)          #change to log(),取自然对数可以避免下溢或者错误,不影响最终的答案
    p0Vect = log(p0Num/p0Denom)          #change to log()
    return p0Vect,p1Vect,pAbusive#pAbusive为是1的概率


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):#朴素贝叶斯分类函数:vec2Classifys是要分类的向量
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():#调用函数
    listOPosts,listClasses=loadDataSet()
    myVocabList=createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0v,p1v,pAb=trainNB0(trainMat,listClasses)
    testEntry = ['love', 'my', 'dalmation']#测试样本
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc,p0v,p1v,pAb))
    testEntry = ['stupid', 'cute']#测试样本
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc,p0v,p1v,pAb))


def textParse(bigString):    #input is big string, #output is word list 转换为词列表
    listOfTokens = re.split(r'\W*', bigString)#通过正则化提取后分割
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]#转换为小写

def spamTest():#测试算法
    #解析为词列表
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open('C:\\Users\\ASUS\Desktop\机器学习实战源代码\machinelearninginaction\Ch04\email/spam/%d.txt' % i,'r').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('C:\\Users\\ASUS\Desktop\机器学习实战源代码\machinelearninginaction\Ch04\email/ham/%d.txt' % i,'r').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    #转换为测试集和训练集
    vocabList = createVocabList(docList)#create vocabulary
    trainingSet = list(range(50)); testSet=[]           #create test set
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))#随机抽取放入测试集
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex] #把他从训练集中删除

    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))#转换为向量
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))#求得概率
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:#测试错误率
            errorCount += 1
    print('the error rate is: ',float(errorCount)/len(testSet))
    #return vocabList,fullText

spamTest()