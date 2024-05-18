import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import neuralnetwork
import helpers

wine = pd.read_csv('hw3_wine.csv', delim_whitespace=True, float_precision=None)
wine.columns = ['class', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
wine = wine.drop(columns=[14], axis=1)

helpers.normalize(wine)

wine = wine.to_numpy()
folds = helpers.k_fold(wine, 10)


#Format training data
formatted_folds = []
for fold in folds:
    formatted_fold = []
    for i in range(len(fold)):
        actual = fold[i][0]
        d_actual = None
        if actual == 1:
            d_actual = np.array([1, 0, 0])
        if actual == 2:
            d_actual = np.array([0, 1, 0])
        if actual == 3:
            d_actual = np.array([0, 0, 1])
        data = ((fold[i][1:14], d_actual))
        formatted_fold.append(data)
    formatted_folds.append(formatted_fold)

#hyper-params, layer_dim = net structure, reg = regular constant, alpha, learn rate
layer_dims = []
regs = []
alpha = 0.5
layer_dims1 = [13, 16, 3]
reg1 = .001
regs.append(reg1)
layer_dims.append(layer_dims1)
layer_dims2 = [13, 8, 4, 3]
reg2 = .001
regs.append(reg2)
layer_dims.append(layer_dims2)
layer_dims3 = [13, 16, 8, 4, 3]
layer_dims.append(layer_dims3)
reg3 = .1
regs.append(reg3)
layer_dims4 = [13, 2, 3]
layer_dims.append(layer_dims4)
reg4 = .0001
regs.append(reg4)
layer_dims5 = [13, 64, 3]
layer_dims.append(layer_dims5)
reg5 = .05
regs.append(reg5)
layer_dims6 = [13, 16, 16, 3]
layer_dims.append(layer_dims6)
reg6 = .025
regs.append(reg6)

#k-fold stuff
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
        f1s.append(helpers.F1(results, [0,1,2]))
    
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
        net.backPropogate(regs[4], alpha, 100)
        costs = []
        for testdata in test:
            a = net.forwardpropogate(testdata[0])[last_layer]
            cost = net.costFunction(a, testdata[1])
            costs.append(cost)
        
        totalCosts.append(net.totalCost(costs, regs[4]))

plt.plot(numSamples, totalCosts)
plt.show()




            
            

            
            
    