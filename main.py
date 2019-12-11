import random
import numpy as np
import operator


# KNN算法
def classify0(inx, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inx, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 数据归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest(dataset, data_lable):
    datingDataMat = dataset
    datingLabels = data_lable
    hoRatio = 0.10
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 4)
        print("分类结果:%d\t真实类别:%d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("准确率:%f%%" % (100 - (errorCount / float(numTestVecs) * 100)))


# 总数据集
dataset = []

# 标签集
data_lable = []

# 填充数据集
for i in range(1000):
    # 临时数据，存放一条数据
    data_temp = []
    # 甜度
    sweet = random.randint(0, 100)
    data_temp.append(sweet)
    # 酸度
    sour = random.randint(0, 100)
    data_temp.append(sour)
    # 水分
    water = random.randint(0, 100)
    data_temp.append(water)
    # 脆度
    crisp = random.randint(0, 100)
    data_temp.append(crisp)
    # 填充至总数据集
    dataset.append(data_temp)
    # 生成标签数据
    if sour >= 60:
        data_lable.append(0)
    elif (sweet >= 50 and water >= 50) or (sweet >= 50 and crisp >= 50):
        data_lable.append(2)
    else:
        data_lable.append(1)

# 将数据集转换成numpy.array
dataset = np.array(dataset)

# 数据归一化
norm_dataset, ranges, minvals = autoNorm(dataset)

# 测试算法在本项目中的应用
datingClassTest(norm_dataset, data_lable)
