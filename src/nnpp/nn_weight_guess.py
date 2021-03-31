"""
====================================================================================================
Module to guess the initial weights in order to save training time
----------------------------------------------------------------------------------------------------
While training the neural network,it was getting stuck at a local minima and fluctuating there for a 
long period of time.So I took two structures with different number of atoms and solved for the 
individual atomic energy of each atomtype to get an approximate energy range for the individual 
atomic energy of the the 2 types of atoms.Then these individual atomic energy of 2 atomtypes were 
used to train the 2  individual atomic NNs,for large no: of epochs.After training were done,the 
weights of individual NNs were saved,to be used as a starting point for the actual training process.
====================================================================================================
"""
import numpy as np
import os

from neural_network import(
    NeuralNetwork,
    sigmoid,
    linear,
    train,
)

l_r = 1e-3
file_name = 'structure1694.txt' #just a random dataset
x = np.loadtxt(os.path.join('./symmetry_txt','%s') %file_name)

n = len(x)
a = x.reshape(n,1,70)
Ti_input = a[0]       #extracting symmetry vector of a Ti atom
O_input = a[-2]       #extracting symmetry vector of an O atom
Ti_output = [-1638]   #approximate energy to be given by Ti-NN
O_output = [-427.5]   #approximate energy to be given by O-NN
node_list = [70,11,11,1]          #contains the layer sizes
activations = ['sigmoid','sigmoid','linear']    
nn_Ti_1a = NeuralNetwork(node_list,activations)
nn_O_1a  = NeuralNetwork(node_list,activations)
print("\n--------------------------------  Weight Guess module----------------------------------------\n")
print('## Training the individual atomic NNs to get a suitable starting point for actual training  ##\n\n')

def train(nn,a,e_ref,learning_rate):
    """A simple training function to train the atomic NN"""
    output = nn.forward_prop(a)
    w1,w2,w3,_,_,_ = nn.backward_prop(a,output,e_ref)
    nn.NN_optimize(w1,w2,w3,0,0,0,learning_rate)

for i in range(10000):
    train(nn_Ti_1a,Ti_input,Ti_output,learning_rate=l_r)

predicted_energy = nn_Ti_1a.forward_prop(Ti_input)
print('Reference =',Ti_output ,'-------------','Predicted = ',predicted_energy)

for i in range(10000):
    train(nn_O_1a,O_input,O_output,learning_rate=l_r)

predicted_energy2 = nn_O_1a.forward_prop(O_input)
print('Reference =',O_output,'-------------','Predicted = ',predicted_energy2)



Ti_weights = {
                'w1' : nn_Ti_1a.weights1,
                'w2' : nn_Ti_1a.weights2,
                'w3' : nn_Ti_1a.weights3,
                'b1' : nn_Ti_1a.bias1,
                'b2' : nn_Ti_1a.bias2,
                'b3' : nn_Ti_1a.bias3,        
                }

O_weights = {
                'w1' : nn_O_1a.weights1,
                'w2' : nn_O_1a.weights2,
                'w3' : nn_O_1a.weights3,
                'b1' : nn_O_1a.bias1,
                'b2' : nn_O_1a.bias2,
                'b3' : nn_O_1a.bias3,                 
}

np.savez('params/dict_ti_11_weights.npz',**Ti_weights)   #saving the weights of Ti-NN
#npzfile1 = np.load('dict_ti_11_weights.npz')
#print(npzfile1['w1'].shape)


np.savez('params/dict_O_11_weights.npz',**O_weights)
#npzfile2 = np.load('dict_O_11_weights.npz')       #saving the weights of O-NN
#print(npzfile2['w1'].shape)

print('\n-----------  The weight parameters for starting the training has been saved    ------------\n')