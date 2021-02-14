import pytest
import os
import numpy as np
from src.nnpp.neural_network2 import(
    NeuralNetwork,
    sigmoid,
    ReLU,
    tanh,
    linear,
    numerical_gradients,
    nn_switcher,
    structure_forward_prop,
    structure_backward_prop,
    train,
    MSE,
    stochastic_gradient_descent,
)

def test_whether_HDNN_learns_for_very_small_dataset():
    node_list = [70,10,9,1]          #contains the layer sizes
    activations = ['sigmoid','sigmoid','linear']    
    nn_Ti_test5 = NeuralNetwork(node_list,activations)
    nn_O_test5  = NeuralNetwork(node_list,activations)
    print('Ti --',nn_Ti_test5,'\n','O --',nn_O_test5)

    file_name_list = ['structure0005.txt','structure0004.txt','structure0003.txt','structure0002.txt','structure0001.txt']#,'structure1249.txt']
    X_list = [np.loadtxt(os.path.join('./symmetry_functions_demo','%s') %file_name) for file_name in file_name_list]
    A_list = [x.reshape(len(x),1,70) for x in X_list]
    E_ref_list = [[-19960.74194513],[-19960.78597929],[-19960.75811714],[-19960.69526834],[-19960.66173260]]#,[-4987.12739129]]
    
    cost_variation1,_ = stochastic_gradient_descent(nn_Ti_test5,nn_O_test5,A_list,E_ref_list,learning_rate=5e-5,epochs=50)
    assert (abs(cost_variation1[-1])< 0.5 )