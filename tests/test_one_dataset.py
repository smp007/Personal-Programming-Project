"""
====================================================================================================
Test to check whether the NN overfits on a single data set
----------------------------------------------------------------------------------------------------
The NN assembly is trained using a single dataset for a particular number of epoch and after that,the
same NNs are used to predict the energy of the same dataset.If the learning capability of the NN is okay,
Then it should predict the reference output of the dataset.
====================================================================================================
"""
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
    MSE_basic
)

learning_rate = 5e-5

def test_whether_NN_trained_on_a_single_dataset_overfits_1():
    file_name = 'structure1249.txt'
    #learning_rate = 0.001
    x = np.loadtxt(os.path.join('./symmetry_txt','%s') %file_name)
    n = len(x)
    a = x.reshape(n,1,70)
    E_ref = [[-4987.12739129]]
    node_list = [70,11,11,1]          #contains the layer sizes
    activations = ['sigmoid','sigmoid','linear']    
    nn_Ti_1a = NeuralNetwork(node_list,activations)
    nn_O_1a  = NeuralNetwork(node_list,activations)

    index = nn_switcher(a)
    #training hte NN
    for i in range(1670):
        train(nn_Ti_1a,nn_O_1a,a,E_ref,learning_rate)
    #making the NN to predict the energy of same dataset
    predicted_energy = sum(structure_forward_prop(a,nn_Ti_1a,nn_O_1a))
    cost = MSE_basic(predicted_energy,E_ref)
    assert np.isclose(predicted_energy,E_ref)



def test_whether_NN_trained_on_a_single_dataset_overfits_2():
    file_name = 'structure0001.txt'
    x = np.loadtxt(os.path.join('./symmetry_txt','%s') %file_name)
    n = len(x)
    a = x.reshape(n,1,70)
    E_ref = [[-19960.66173260]]
    node_list = [70,13,9,1]          #contains the layer sizes
    activations = ['sigmoid','sigmoid','linear']    
    nn_Ti_test2 = NeuralNetwork(node_list,activations)
    nn_O_test2  = NeuralNetwork(node_list,activations)
    index = nn_switcher(a)
    for i in range(1450):
        train(nn_Ti_test2,nn_O_test2,a,E_ref,learning_rate)

    predicted_energy = sum(structure_forward_prop(a,nn_Ti_test2,nn_O_test2))

    assert np.isclose(predicted_energy,E_ref)

def test_whether_NN_trained_on_a_single_dataset_overfits_3():
    
    file_name = 'structure7501.txt'

    x = np.loadtxt(os.path.join('./symmetry_txt','%s') %file_name)
    n = len(x)
    a = x.reshape(n,1,70)
    E_ref = [[-78964.89340133]]
    node_list = [70,17,10,1]          #contains the layer sizes
    activations = ['sigmoid','sigmoid','linear']    
    nn_Ti_test3 = NeuralNetwork(node_list,activations)
    nn_O_test3  = NeuralNetwork(node_list,activations)
    index = nn_switcher(a)
    for i in range(1500):
        train(nn_Ti_test3,nn_O_test3,a,E_ref,learning_rate)

    predicted_energy = sum(structure_forward_prop(a,nn_Ti_test3,nn_O_test3))

    assert np.isclose(predicted_energy,E_ref)