import pytest
import numpy as np
from src.nnpp.neural_network import(
    NeuralNetwork,
    sigmoid,
    ReLU,
    tanh,
    linear,
)

def weights_init(nn):
    '''Setting weights to constant value = 1'''
    nn.weights1 = np.ones(nn.weights1.shape)
    nn.weights2 = np.ones(nn.weights2.shape)
    nn.weights3 = np.ones(nn.weights3.shape)
    return 0


def test_forward_prop_small_network():
    '''testing forward prop in 4-3-3-1 network'''
    x = np.array([1,1,1,1])          #input
    y = [0.94533048]                 #preknown output
    node_list = [4,3,3,1]
    activations = [sigmoid,sigmoid,sigmoid]
    nn_test1 = NeuralNetwork(node_list,activations)
    weights_init(nn_test1)          #Setting weights to constant value = 1
    y_test = nn_test1.forward_prop(x)
    assert np.isclose(y_test,y)                

def test_forward_prop_big_network_1():
    '''testing forward prop in 70-10-10-1 network
    sigmoid--sigmoid--sigmoid'''

    x = np.ones((1,70))             #input
    y = [0.99995458]                #preknown output
    node_list = [70,10,10,1]    
    activations = [sigmoid,sigmoid,sigmoid]
    nn_test2 = NeuralNetwork(node_list,activations)
    weights_init(nn_test2)          #Setting weights to constant value = 1
    y_test = nn_test2.forward_prop(x)
    assert np.isclose(y_test,y)  

def test_forward_prop_big_network_2():
    '''testing forward prop in 70-10-10-1 network
    sigmoid--sigmoid--linear'''
    x = np.ones((1,70))             #input
    y = [9.99954602]                #preknown output
    node_list = [70,10,10,1]
    activations = [sigmoid,sigmoid,linear]
    nn_test3 = NeuralNetwork(node_list,activations)
    weights_init(nn_test3)          #Setting weights to constant value = 1
    y_test = nn_test3.forward_prop(x)
    assert abs(y_test-y) < 1e-8

def test_forward_prop_big_network_3():
    '''testing forward prop in 70-2-2-1 network
    sigmoid--sigmoid--linear'''
    x = np.ones((1,70))             #input
    y = [1.76159416]                #preknown output
    node_list = [70,2,2,1]    
    activations = [sigmoid,sigmoid,linear]
    nn_test4 = NeuralNetwork(node_list,activations)
    weights_init(nn_test4)          #Setting weights to constant value = 1
    y_test = nn_test4.forward_prop(x)
    assert np.isclose(y_test,y)  



def test_forward_prop_big_network_4():
    '''testing forward prop in 70-10-10-1 network
    linear--linear--linear'''
    x = np.ones((1,70))             #input
    y = [7000]                      #preknown output(when input and weights are array of ones,the output will be the product of number of nodes in each layer)
    node_list = [70,10,10,1]    
    activations = [linear,linear,linear]
    nn_test5 = NeuralNetwork(node_list,activations)
    weights_init(nn_test5)          #Setting weights to constant value = 1
    y_test = nn_test5.forward_prop(x)
    assert np.isclose(y_test,y)  