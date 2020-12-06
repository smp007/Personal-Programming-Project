import numpy as np 
import matplotlib.pyplot as plt 

# z = np.linspace(-100,100,10000)
# print(leaky_ReLU(z))

# #plt.plot(z,leaky_ReLU(z))
# plt.plot(z,sigmoid(z))
# plt.plot(z,tanh(z))
# #plt.plot(z,ReLU(z))
# plt.grid('True')
# plt.show()

x1 = np.array([1,2,3,4])
x2 = np.array([6,8,3,1])
y = np.array([92])
print(x1,x2)


class NeuralNetwork:
    def __init__(self):
        '''Initialisation of neural network parameters'''
        self.input_nodes = 4
        self.hidden_layer1_nodes = 3
        self.output_nodes = 1

        self.weights1 = np.random.randn(self.input_nodes,self.hidden_layer1_nodes)
        print(self.weights1)
        self.weights2 = np.random.randn(self.hidden_layer1_nodes,self.output_nodes)
        print(self.weights2)

    #---------------------------------------------------------------------------
    #Activation functions
    #---------------------------------------------------------------------------

    def sigmoid(self,z):
        s = 1/(1+np.exp(-1*z))
        return s 

    def ReLU(self,z):
        val = [x if x>0 else 0 for x in z]
        return val

    def leaky_ReLU(self,z):
        val = [x if x>0 else (0.01*x) for x in z]
        return val

    def tanh(self,z):
        val = (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
        return val

    def linear(self,z):
        return z
        

    def forward_prop(self,x):
        self.layer1 = self.sigmoid(np.dot(x,self.weights1))
        output      = self.sigmoid(np.dot(self.layer1,self.weights2))
        return output
        



nn_Ti = NeuralNetwork()
nn_O  = NeuralNetwork()
o1 = nn_Ti.forward_prop(x1)
o2 = nn_O.forward_prop(x2)

print(o1,o2)
