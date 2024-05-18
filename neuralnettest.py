import neuralnetwork
import numpy as np
from collections import defaultdict

def testExample1():
    layer_dim = [1, 2, 1]
    training_data = [(np.array([0.13]), np.array([0.9])), (np.array([0.42]), np.array([0.23]))]
    neuralnet = neuralnetwork.NeuralNet(layer_dim, training_data, True, False)

    neuralnet.backPropogate(0, .3, 1)

def testExample2():
    layer_dim = [2, 4, 3, 2]
    training_data = [(np.array([0.32, 0.68]), np.array([0.75, 0.98])), (np.array([0.83, 0.02]), np.array([0.75, 0.28]))]
    neuralnet = neuralnetwork.NeuralNet(layer_dim, training_data, False, True)
    
    neuralnet.backPropogate(0.25, .3, 2)

print('Example 1: ')
testExample1()
print('Example 2: ')
testExample2()


