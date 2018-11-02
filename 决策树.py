import trees
import treePlotter

#将分类器存储到硬盘上，使其持久化
myDat,labels=trees.createDataSet()
myTree=treePlotter.retrieveTree(0)
trees.storeTree(myTree,'F:\\python库包\机器学习实战源代码\machinelearninginaction\Ch03\classifierStorage.txt')
trees.grabTree('F:\\python库包\机器学习实战源代码\machinelearninginaction\Ch03\classifierStorage.txt')

fr=open('F:\\python库包\机器学习实战源代码\machinelearninginaction\Ch03\lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels=['age','prescript','astigmatic','tearRate']
lensesTree=trees.createTree(lenses,lensesLabels)#创建决策树
print(lensesTree)
treePlotter.createPlot(lensesTree)#画图

