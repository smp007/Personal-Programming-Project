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
    train,
)
np.random.seed(5)
learning_rate = 0.001

def test_whether_NN_trained_on_a_single_dataset_overfits_1():
    file_name = 'structure1249.txt'

    x = np.loadtxt(os.path.join('./symmetry_functions','%s') %file_name)
    n = len(x)
    a = x.reshape(n,1,70)
    E_ref = [[-4987.12739129]]
    node_list = [70,10,10,1]          #contains the layer sizes
    activations = [sigmoid,sigmoid,linear]    
    nn_Ti_1 = NeuralNetwork(node_list,activations)
    nn_O_1  = NeuralNetwork(node_list,activations)

    index = nn_switcher(a)

    for i in range(250):
        train(nn_Ti_1,nn_O_1,index,a,E_ref,learning_rate)
    
    predicted_energy = sum(structure_forward_prop(a,nn_Ti_1,nn_O_1,index))
    print(predicted_energy)
 
    assert np.isclose(predicted_energy,E_ref)


def test_whether_NN_trained_on_a_single_dataset_overfits_2():
        file_name = 'structure0001.txt'

        x = np.loadtxt(os.path.join('./symmetry_functions','%s') %file_name)
        n = len(x)
        a = x.reshape(n,1,70)
        E_ref = [[-19960.66173260]]
        node_list = [70,10,10,1]          #contains the layer sizes
        activations = [sigmoid,sigmoid,linear]    
        nn_Ti_test2 = NeuralNetwork(node_list,activations)
        nn_O_test2  = NeuralNetwork(node_list,activations)
        index = nn_switcher(a)
        for i in range(250):
            train(nn_Ti_test2,nn_O_test2,index,a,E_ref,learning_rate)
    
        predicted_energy = sum(structure_forward_prop(a,nn_Ti_test2,nn_O_test2,index))
        print(predicted_energy)

        assert np.isclose(predicted_energy,E_ref)


def test_whether_NN_trained_on_a_single_dataset_overfits_3():
        file_name = 'structure7501.txt'

        x = np.loadtxt(os.path.join('./symmetry_functions','%s') %file_name)
        n = len(x)
        a = x.reshape(n,1,70)
        E_ref = [[-78964.89340133]]
        node_list = [70,10,10,1]          #contains the layer sizes
        activations = [sigmoid,sigmoid,linear]    
        nn_Ti_test3 = NeuralNetwork(node_list,activations)
        nn_O_test3  = NeuralNetwork(node_list,activations)
        index = nn_switcher(a)
        for i in range(250):
            train(nn_Ti_test3,nn_O_test3,index,a,E_ref,learning_rate)
    
        predicted_energy = sum(structure_forward_prop(a,nn_Ti_test3,nn_O_test3,index))
        print(predicted_energy)

        assert np.isclose(predicted_energy,E_ref)