"""Neural net class"""
import numpy as np
from collections import defaultdict

class NeuralNet:
    #Initialize the neural network, we need to initialize weights and bias
    def __init__(self, layer_dims, training_data, isExample1, isExample2) -> None:
        #We need a matrix in between each layer representing edge weights
        self.layer_dims = layer_dims
        self.training = training_data
        self.weights = defaultdict(int)
        #Following lines are manually set weights for example 1, comment out unless running
        #example 1 in neuralnettest1.py
        if isExample1:
            self.weights[1] = np.array([[0.4, 0.1],[0.3, 0.2]])
            self.weights[2] = np.array([0.7, 0.5, 0.6])
        elif isExample2:
            self.weights[1] = np.array([[0.42, 0.15, 0.4], [0.72, 0.10, 0.54], [0.01, 0.19, 0.42], [0.30, 0.35, 0.68]])
            self.weights[2] = np.array([[0.21, 0.67, 0.14, 0.96, 0.87], [0.87, 0.42, 0.2, 0.32, 0.89],
                                       [0.03, 0.56, 0.80, 0.69, 0.09]])
            self.weights[3] = np.array([[0.04, 0.87, 0.42, 0.53], [0.17, 0.10, 0.95, 0.69]])
        else:
            for i in range(1, len(layer_dims)):
                self.weights[i] = np.random.randn(self.layer_dims[i], self.layer_dims[i - 1] + 1)

    def sigmoid(self, output):
        return 1/(1 + np.exp(np.dot(-1, output)))
    
    #Forward propogates one instance (can also use for testing!)
    def forwardpropogate(self, input):
        activations = defaultdict(int)
        curr_layer = input
        curr_layer = np.insert(curr_layer, 0, 1)
        print('a_1: ')
        print(curr_layer)
        activations[0] = curr_layer
        for i in range(1, len(self.layer_dims)):
            output = np.dot(self.weights[i], curr_layer) 
            print('z' + str(i + 1) + ': ')
            print(output)

            curr_layer = self.sigmoid(output)
            print('a' + str(i + 1) + ': ')

            if(not i == len(self.layer_dims) - 1):
                curr_layer = np.insert(curr_layer, 0, 1)
            activations[i] = curr_layer
            print(curr_layer)
            
        return activations
    
    #Calculates cost for one instance
    def costFunction(self, predicted, actual):
        return (-1 * np.dot(np.log(predicted), actual)) - np.dot(np.log(1 - predicted), (1 - actual))

    def totalCost(self, costs, reg):
        sum = np.sum(costs)
        cost = sum / len(costs)
        
        weightsum = 0
        for i in range(1, len(self.layer_dims)):
            if self.weights[i].ndim == 1:
                for j in range(1, len(self.weights[i])):
                    weightsum += self.weights[i][j] ** 2
            else:
                for j in range(0, len(self.weights[i])):
                    for k in range(1, len(self.weights[i][j])):
                        weightsum += self.weights[i][j][k] ** 2
            
        regularizer = (reg/ (2 * len(costs))) * weightsum
        return cost + regularizer
                    
    def backPropogate(self, reg, alpha, epochs):
        last_layer = len(self.layer_dims) - 1
        for epoch in range(epochs):
            print('epoch: ' + str(epoch))
            count = 0
            gradients = defaultdict(int)
            deltas = defaultdict(int)
            costs = []
            for instance in self.training:
                count += 1
                print('training instance ' + str(count))
                attr, actual = instance

                activations = self.forwardpropogate(attr)
                
                cost = self.costFunction(activations[last_layer], actual)
                
                costs.append(cost)
                print('cost for instance ' + str(count) + ': ')
                print(cost)
                error_last = activations[last_layer] - actual

                deltas[last_layer] = error_last
                print('delta, layer: ' + str(last_layer + 1))
                print(deltas[last_layer])
                delta_prev = error_last

                for k in range(last_layer - 1, 0, -1):  
                    weight_t = np.transpose(self.weights[k + 1])
                    if weight_t.ndim == 1:
                        weight_t.shape = (len(weight_t), 1)
               
                    delta = (weight_t @ delta_prev) * activations[k] * (1 - activations[k])

                    deltas[k] = delta[1:]
                    print('delta, layer: ' + str(k + 1))
                    print(deltas[k])
                    delta_prev = delta[1:]

                for k in range(last_layer - 1, -1, -1):
                    gradients[k] = gradients[k] + np.outer(deltas[k + 1], activations[k])
                    print('Gradient layer ' + str(k + 1))
                    print(np.outer(deltas[k + 1], activations[k]))

            
            print('Total cost for all instances: ' + str(self.totalCost(costs, reg)))

            for k in range(last_layer - 1, -1, -1):
                regularizer = reg * self.weights[k + 1]
           
                if not regularizer.ndim == 1: 
                    for i in range(len(regularizer)):
                        regularizer[i][0] = 0
                else:
                    regularizer[0] = 0
                gradients[k] = (1/len(self.training)) * (gradients[k] + regularizer)

            for k in range(last_layer - 1, -1, -1):
                self.weights[k + 1] = self.weights[k + 1] - alpha * gradients[k]

        return activations