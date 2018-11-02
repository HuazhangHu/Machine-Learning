import matplotlib.pyplot as plt
import trees

#定义文本框和箭头格式
decisionNode=dict(boxstyle='sawtooth',fc='0.8')#决策节点文本框为波浪框
leafNode=dict(boxstyle="round4",fc="0.8")#叶子节点文本框为圆边框
arrow_args=dict(arrowstyle='<-')#箭头格式

def plotNode(nodeText,centerPt,parentPt,nodeType):#绘制节点
    createPlot.ax1.annotate(nodeText, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

'''
def createPlot():
    fig=plt.figure(1,facecolor="white")
    fig.clf()
    createPlot.ax1=plt.subplot(111,frameon=False)#创建了一个新图形并清空绘图区
    plotNode('a decision node',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('a leaf node',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()
'''

def getNumLeafs(myTree):#通过递归进栈的方法计算叶子节点个数，从而确定x轴长度
    numLeafs=0
    firstStr=list(myTree.keys())[0]#根节点
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs+=getNumLeafs(secondDict[key])#递归进栈
        else:
            numLeafs+=1
    return numLeafs

def getTreeDepth(myTree):#深度遍历计算树的层数，从而计算y轴高度,也可以知道事情最坏的情况下的复杂度
    maxDepth=0
    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#如果类型还是字典则层数增加1
            thisDepth=1+getTreeDepth(secondDict[key])
        else:
            thisDepth=1
        if thisDepth>maxDepth:maxDepth=thisDepth
    return maxDepth

def retrieveTree(i):#预先储存树的信息
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

def plotMidText(cntrPt, parentPt, txtString):#在父子节点计算中间位置并填充文字信息
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  #叶子节点的个数，x轴宽度
    depth = getTreeDepth(myTree)#树的深度，y轴长度
    firstStr = list(myTree.keys())[0] #根节点的标签
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)#按比例绘制图
    plotMidText(cntrPt, parentPt, nodeTxt)#绘出子节点的特征值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#如何不是叶子节点则递归调用
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #如果是叶子节点则画出节点
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumLeafs(inTree))#树的宽度
    plotTree.totalD = float(getTreeDepth(inTree))#树的深度
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0
    plotTree(inTree, (0.5,1.0), '')
    plt.show()

