import pytest
import numpy as np
from src.nnpp.neural_network2 import(
    NeuralNetwork,
    sigmoid,
    ReLU,
    tanh,
    linear,
    numerical_gradients
)


#Gradient checking--------------------------------------------------------------
def test_gardient_checking_example():

    def f(x):
        return x**2
    pertubation = 0.0001
    x = 2

    numeric_gradient = (f(x+pertubation)-f(x-pertubation))/(2*pertubation)
    analytical_gradient = 2*x

    assert np.isclose(numeric_gradient,analytical_gradient)

#-------------------------------------------------------------------------------
#
def test_gradient_checking_small_neural_network_1():
    #declare structure of the neural network(no of nodes,activations of layers)
    node_list = [2,3,3,1]
    activations = ['ReLU','ReLU','ReLU']  
    #set inputs and output
    x = np.array([1,2]).reshape(1,node_list[0])
    y = np.array([10]).reshape(1,node_list[-1])
    nn_test_1 = NeuralNetwork(node_list,activations)
    e_nn = nn_test_1.forward_prop(x)
    derivative_analytical = nn_test_1.analytical_gradients(x,e_nn,y)
    derivative_numerical = numerical_gradients(nn_test_1,x,e_nn,y)
    print(derivative_analytical,'\n',derivative_numerical)
    assert np.isclose(derivative_analytical,derivative_numerical).all()


def test_gradient_checking_small_neural_network_2():
    node_list = [2,2,2,1]
    activations = ['sigmoid','sigmoid','sigmoid']  
    x1 = np.array([2,20]).reshape(1,node_list[0])
    y1 = np.array([1000]).reshape(1,node_list[-1])
    nn_test_2 = NeuralNetwork(node_list,activations)
    e_nn = nn_test_2.forward_prop(x1)
    derivative_analytical = nn_test_2.analytical_gradients(x1,e_nn,y1)
    derivative_numerical = numerical_gradients(nn_test_2,x1,e_nn,y1)
    print(derivative_analytical,'\n',derivative_numerical)
    assert np.allclose(derivative_analytical,derivative_numerical,atol = 1e-5)



