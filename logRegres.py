import math
from numpy import *

def loadDataSet():#处理数据得到数据和标签类别
    dataMat=[]
    labelMat=[]
    fr=open('F:\\python库包\机器学习实战源代码\machinelearninginaction\Ch05\\testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])#[1.0,特征一，特征二]
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):#sigmoid()函数，用于分类，>0.5分为1类,<0.5分为0类
    return longfloat(1.0/(1+exp(-inX)))#1.0/(1+e的-inX次方），解决数据溢出

def gradAscent(dataMatIn,classLabels):#用梯度上升算法找到最佳参数
    dataMatrix=mat(dataMatIn)#转换为矩阵100*3
    labelMat=mat(classLabels).transpose()#transpose()转置,100*1
    m,n=shape(dataMatrix)#m=100,n=3
    alpha=0.001#移动的步长，意义在于以局部最优通过不断的迭代达到全局最优
    maxCycles=500#迭代次数
    weights=ones((n,1))#3*1，回归系数组
    for k in range(maxCycles):#不断迭代，不断更新回归系数
        h=sigmoid(dataMatrix*weights)#（100*3）*（3*1）=（100*1）每一个特征值乘以一个回归系数，然后把所有的都相加再代入sigmoid函数中
        error=(labelMat-h)#100*1
        weights=weights+alpha*dataMatrix.transpose()*error#3*1+步长*3*100 *100*1
    return weights.getA()#返回最佳参数

def plotBestFit(weights):#绘制决策边界
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):#随机梯度上升，，每次仅用一个样本点来更新数据
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))#每次选取一个样本数据
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):#随机梯度上升算法，1.步长是变化的，2.随机选取样本数据迭代，减少波动
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))#python3中range()返回的是range对象，而不是数组对象
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001 #步长在每次迭代时都会变化，不断减小，最小为0.0001
            randIndex = int(random.uniform(0,len(dataIndex)))#随机选出数据样本来更新回归系数，避免回归系数周期性的波动
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])#然后把选过的删去
    return weights

def classifyVector(inX,weight):#通过sigmoid（）函数分类,inx为列表
    prob=sigmoid(sum(inX*weight))
    if prob>0.5:
        return 1.0
    else:
        return 0.0

def classifyClass(dataArr,labels):#分类
    frTrain = open('C:\\Users\ASUS\Desktop\机器学习实战源代码\machinelearninginaction\Ch05\horseColicTraining.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)  # 形成矩阵
        trainingLabels.append(float(currLine[21]))  # 列表
    traininWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)  # 调用改进随机梯度上升算法求的最佳参数
    classNum=classifyVector(dataArr,traininWeights)
    if classNum==0.0: num=0
    else:
        num =1
    print("分类器训练后,根据传入的数据判断类别为:{}".format(labels[num]))


def colicTest():
    frTrain=open('C:\\Users\ASUS\Desktop\机器学习实战源代码\machinelearninginaction\Ch05\horseColicTraining.txt')
    frTest=open('C:\\Users\ASUS\Desktop\机器学习实战源代码\machinelearninginaction\Ch05\horseColicTest.txt')
    trainingSet=[]
    trainingLabels=[]
    for line in frTrain.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)#形成矩阵
        trainingLabels.append(float(currLine[21]))#列表
    traininWeights=stocGradAscent1(array(trainingSet),trainingLabels,500)#调用改进随机梯度上升算法求的最佳参数
    #训练集
    errorCount=0;numTestVec=0.0
    for line in frTest.readlines():#测试集
        numTestVec+=1.0#测试数量
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),traininWeights))!=int(currLine[21]):#如果测试的结果于实际结果不符
            errorCount+=1
    errorRate=float(errorCount)/numTestVec
    print("错误率为:%f"%errorRate)
    return errorRate

def multTest():
    numTests=50;errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print("经过了{}次测试,平均错误率为{}".format(numTests,errorSum/float(numTests)))




