import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import csv

def opencsv():
    #使用pandas加载csv文件
    train = pd.read_csv('F:/Kaggle/DigitRecognizer/data/train.csv')
    test = pd.read_csv('F:/Kaggle/DigitRecognizer/data/test.csv')

    train_data = train.values[0:,1:]
    train_lable = train.values[0:,1]
    test_data = test.values[0:, 0:]
    return train_data, train_lable, test_data

trainData, trainLabel, testData = opencsv()

# 模型训练
def knnClassify(trainData, trainLabel):
    knnClf = KNeighborsClassifier()   # default:k = 5,defined by yourself:KNeighborsClassifier(n_neighbors=10)
    knnClf.fit(trainData, np.ravel(trainLabel))
    return knnClf

knnClf = knnClassify(trainData, trainLabel)

# 结果预测
testLabel = knnClf.predict(testData)
print(testLabel)
def saveResult(result, csvName):
    with open(csvName, 'w', newline="") as myFile:
        myWriter = csv.writer(myFile)
        myWriter.writerow(["ImageId", "Label"])
        index = 0
        for i in result:
            tmp = []
            index = index+1
            tmp.append(index)
            # tmp.append(i)
            tmp.append(int(i))
            myWriter.writerow(tmp)

# 结果的输出
saveResult(testLabel, 'F:/Kaggle/DigitRecognizer/data/Result_sklearn_knn.csv')
