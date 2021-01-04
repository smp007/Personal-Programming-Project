import numpy as np
import os

from neural_network2 import(
    NeuralNetwork,
    sigmoid,
    linear,
    train,
)

l_r = 0.001
file_name = 'structure1694.txt'
x = np.loadtxt(os.path.join('./sym_fun_all_2','%s') %file_name)

n = len(x)
print(n)
a = x.reshape(n,1,70)
Ti_input = a[0] 
O_input = a[-2] 
Ti_output = [-1638]
O_output = [-427.5]
node_list = [70,9,9,1]          #contains the layer sizes
activations = ['sigmoid','sigmoid','linear']    
nn_Ti_1a = NeuralNetwork(node_list,activations)
nn_O_1a  = NeuralNetwork(node_list,activations)

def train(nn,a,e_ref,learning_rate):
    output = nn.forward_prop(a)
    w1,w2,w3,_,_,_ = nn.backward_prop(a,output,e_ref)
    nn.NN_optimize(w1,w2,w3,0,0,0,learning_rate)

for i in range(10000):
    train(nn_Ti_1a,Ti_input,Ti_output,learning_rate=l_r)

predicted_energy = nn_Ti_1a.forward_prop(Ti_input)
print(predicted_energy)

for i in range(10000):
    train(nn_O_1a,O_input,O_output,learning_rate=l_r)

predicted_energy2 = nn_O_1a.forward_prop(O_input)
print(predicted_energy2)


print(nn_Ti_1a.weights1,'\n',nn_Ti_1a.weights2,'\n',nn_Ti_1a.weights3)

Ti_weights = {
                'w1' : nn_Ti_1a.weights1,
                'w2' : nn_Ti_1a.weights2,
                'w3' : nn_Ti_1a.weights3,
                }

O_weights = {
                'w1' : nn_O_1a.weights1,
                'w2' : nn_O_1a.weights2,
                'w3' : nn_O_1a.weights3,
}


np.savez('dict_ti_weights.npz',**Ti_weights)
npzfile1 = np.load('dict_ti_weights.npz')
print(npzfile1['w1'].shape)


np.savez('dict_O_weights.npz',**O_weights)
npzfile2 = np.load('dict_O_weights.npz')
print(npzfile2['w1'].shape)
