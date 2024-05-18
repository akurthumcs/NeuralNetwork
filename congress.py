import pandas as pd
import numpy as np
import neuralnetwork
import helpers
from matplotlib import pyplot as plt

congress = pd.read_csv('hw3_house_votes_84.csv')
congress.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

congress = congress.to_numpy()
folds = helpers.k_fold(congress, 10)

formatted_folds = []
for fold in folds:
    formatted_fold = []
    for i in range(len(fold)):
        actual = fold[i][16]
        if actual == 0:
            d_actual = np.array([1, 0])
        if actual == 1:
            d_actual = np.array([0, 1])
        data = ((fold[i][0:16], d_actual))
        formatted_fold.append(data)
    formatted_folds.append(formatted_fold)

layer_dims = []
regs = []
alpha = 0.5
layer_dims = []
regs = []
alpha = 0.5
layer_dims1 = [16, 32, 2]
reg1 = .01
regs.append(reg1)
layer_dims.append(layer_dims1)
layer_dims2 = [16, 2, 24, 2]
reg2 = .04
regs.append(reg2)
layer_dims.append(layer_dims2)
layer_dims3 = [16, 16, 8, 16, 2]
layer_dims.append(layer_dims3)
reg3 = .03
regs.append(reg3)
layer_dims4 = [16, 2, 2]
layer_dims.append(layer_dims4)
reg4 = .025
regs.append(reg4)
layer_dims5 = [16, 64, 2]
layer_dims.append(layer_dims5)
reg5 = .05
regs.append(reg5)
layer_dims6 = [16, 16, 32, 2]
layer_dims.append(layer_dims6)
reg6 = .035
regs.append(reg6) 

for netNum in range(0, 6):
    accuracies = []
    f1s = []
    for i in range(len(folds)):
        train, test = helpers.k_foldData(formatted_folds, i)
        net = neuralnetwork.NeuralNet(layer_dims[netNum], train, False, False)
        
        last_layer = len(layer_dims[netNum]) - 1
        net.backPropogate(regs[netNum], alpha, 100)
        results = []
        for testdata in test:
            a = net.forwardpropogate(testdata[0])[last_layer]
            label = np.argmax(a)
            results.append((label, np.argmax(testdata[1])))
        accuracies.append(helpers.accuracy(results))
        f1s.append(helpers.F1(results, [0,1]))

    print('Accuracy of Neural Net ' + str(netNum + 1) + ': ')
    print(np.mean(accuracies))
    print('F1 score of Neural Net ' + str(netNum + 1) + ': ')
    print(np.mean(f1s))

"""Neural net 5 wins!"""
for i in range(len(folds)):
    train, test = helpers.k_foldData(formatted_folds, i)
    net = neuralnetwork.NeuralNet(layer_dims[4], train, False, False)
    
    totalCosts = []
    numSamples = []

    last_layer = len(layer_dims[4]) - 1
    for trainingsamples in range(1, len(train), 5):
        #training with trainingsamples num samples
        numSamples.append(trainingsamples)
        net = neuralnetwork.NeuralNet(layer_dims[4], train[0: trainingsamples], False, False)
        net.backPropogate(regs[4], alpha, 80)
        costs = []
        for testdata in test:
            a = net.forwardpropogate(testdata[0])[last_layer]
            cost = net.costFunction(a, testdata[1])
            costs.append(cost)
        
        totalCosts.append(net.totalCost(costs, regs[4]))

plt.plot(numSamples, totalCosts)
plt.show()
