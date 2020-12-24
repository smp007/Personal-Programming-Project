import pytest
import os
import numpy as np
from src.nnpp.neural_network import(
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
    train2,
    MSE,
    stochastic_gradient_descent,
)

def test_whether_HDNN_learns_for_very_small_dataset():
    file_name_list = ['structure0005.txt','structure0004.txt','structure0003.txt','structure0002.txt','structure0001.txt']#,'structure1249.txt']
    X_list = [np.loadtxt(os.path.join('./symmetry_functions','%s') %file_name) for file_name in file_name_list]
    A_list = [x.reshape(len(x),1,70) for x in X_list]
    E_ref_list = [[-19960.74194513],[-19960.78597929],[-19960.75811714],[-19960.69526834],[-19960.66173260]]#,[-4987.12739129]]

    cost_variation = stochastic_gradient_descent(A_list,E_ref_list,learning_rate=1e-5,epochs=50)
    assert (abs(cost_variation[-1])< 0.1 )