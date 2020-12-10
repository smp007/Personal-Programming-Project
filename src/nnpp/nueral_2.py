import os
import numpy as np 

def sigmoid(z):
    s = 1/(1+np.exp(-1*z))
    return s 

def ReLU(z):
    return np.maximum(0,z)

def tanh(z):
    val = (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
    return val

def linear(z):
    return z
#-------------------------------------------------------------------------------
class NeuralNetwork:
    def __init__(self,nodelist,activations):
        '''Initialisation of neural network parameters'''
        self.input_nodes = nodelist[0]
        self.hidden_layer1_nodes = nodelist[1]        
        self.hidden_layer2_nodes = nodelist[2]        
        self.output_nodes = nodelist[-1]
        self.layer1_activation = activations[0]
        self.layer2_activation = activations[1]
        self.output_activation = activations[-1]

        self.weights1 = np.random.randn(self.input_nodes,self.hidden_layer1_nodes)
        #print(self.weights1)
        self.bias1 = 0
        self.weights2 = np.random.randn(self.hidden_layer1_nodes,self.hidden_layer2_nodes)
        #print(self.weights2)
        self.bias2 = 0
        self.weights3 = np.random.randn(self.hidden_layer2_nodes,self.output_nodes)
        self.bias3 = 0

    def __repr__(self):
        return f"This is a {self.input_nodes}-{self.hidden_layer1_nodes}-{self.hidden_layer2_nodes}-{self.output_nodes} neural network"

        
    def forward_prop(self,x):
        self.layer1 = self.layer1_activation(np.dot(x,self.weights1)+self.bias1)
        self.layer2 = self.layer2_activation(np.dot(self.layer1,self.weights2)+self.bias2)
        output      = self.output_activation(np.dot(self.layer2,self.weights3)+self.bias3)
        return output

x1 = np.array([1,2,3,4])
node_list1 = [4,3,3,1]          #contains the layer sizes
activations = [sigmoid,sigmoid,sigmoid] 
nn_Ti = NeuralNetwork(node_list1,activations)
o1 = nn_Ti.forward_prop(x1)
#o2 = nn_O.forward_prop(x2)

print(o1)