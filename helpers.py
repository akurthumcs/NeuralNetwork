import numpy as np
import random
import math

def normalize(data):
    for column in data:
        if column == 'class':
            continue
        else:
            max = data[column].max()
            min = data[column].min()

            data[column] = (data[column] - min)/(max - min)

    return data

def k_fold(D, k):
    folds = []
    Dcopy = np.copy(D)
    foldSize = math.ceil(len(D)/k)
    for j in range(k):
        fold = []
        for i in range(foldSize):
            index = random.randrange(len(Dcopy))
            fold.append(Dcopy[index])
            np.delete(Dcopy, index)
        folds.append(fold)
    return folds

def k_foldData(folds, i):
    test = folds[i]
    train = []
    for j in range(len(folds)):
        if j == i:
            continue
        else:
            for k in range(len(folds[j])):
                train.append(folds[j][k])
    return train, test
def accuracy(results):
    count = 0
    correct = 0
    for result in results:
        count += 1
        if result[0] == result[1]:
            correct += 1
    
    return correct/count

def F1(results, classes):
    f1s = []
    for i in classes:
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        positive_class = i
        for result in results:
            isPositive = (result[0] == positive_class)
            isSame = (result[0] == result[1])
            if isPositive and isSame:
                TP += 1
            elif isPositive and not isSame:
                FP += 1
            elif not isPositive and isSame:
                TN += 1
            elif not isPositive and not isSame:
                FN += 1
        f1 = TP/(TP + (1/2)*(FP + FN))
        f1s.append(f1)
    
    return np.mean(f1s)

def costFunction(predicted, actual):
        return (-1 * np.dot(np.log(predicted), actual)) - np.dot(np.log(1 - predicted), (1 - actual))