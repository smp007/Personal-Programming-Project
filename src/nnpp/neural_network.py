import os
import numpy as np 
import time
import matplotlib.pyplot as plt 

path = './data_set_TiO2_small'
file_list = sorted(os.listdir(path))
#print((file_list))

number_of_atomss = []
energy_list = []

for file in file_list:

    with open(path+'/'+'%s'%(file)) as f:
        for i,line in enumerate(f):

            if i==0:  #energy line strips at =
                energy_list.append(float((line.strip().split('='))[1][:-3]))            

            if i==8:  #to get no: of atoms...goes to that line strips and splits first element
                no_of_atoms = line.strip().split(' ')
                n=int(no_of_atoms[0])
                number_of_atomss.append(n)

#print(energy_list)


tic = time.time()
def test_train_split(filelist,energylist,split):
    '''Creates an empty array for test split and pops each element from total
dataset and append it to the test set simultaneously'''
    n_total_set = len(filelist)
    n_train_set = split/100 * n_total_set
    train_set = []   
    test_set = filelist
    train_energy = [] 
    test_energy = energylist
    while len(train_set) < n_train_set :
        index = np.random.randint(0,len(test_set))     #randrange(len(train_set))
        #print(index)
        train_set.append(test_set.pop(index))
        train_energy.append(test_energy.pop(index))
    return train_set,test_set,(train_energy),(test_energy)
np.random.seed(5)
#print(test_train_split(file_list,energy_list,80))

a,b,c,d = test_train_split(file_list,energy_list,80)
#print(len(c))
toc = time.time()

test_xy = ([(np.loadtxt(os.path.join('./symmetry_functions','%s') %(x[:-3]+'txt')),y)for x,y in zip(b,d)]) #np.loadtxt(os.path.join('./symmetry_functions','%s') %(x[:-3]+'txt'))
train_xy = ([(np.loadtxt(os.path.join('./symmetry_functions','%s') %(x[:-3]+'txt')),y)for x,y in zip(a,c)])

#print(a[0],b[0])
#print((test_xy),train_xy[0][0])


# z = np.linspace(-100,100,10000)
# print(leaky_ReLU(z))

# #plt.plot(z,leaky_ReLU(z))
# plt.plot(z,sigmoid(z))
# plt.plot(z,tanh(z))
# #plt.plot(z,ReLU(z))
# plt.grid('True')
# plt.show()

x6 = np.array([1,2,3,4])
x7 = np.array([6,8,3,1])
x8 = [[1,2,3,4],[6,8,3,1],[2,3,4,5],[5,4,3,2],[6,7,5,4],[0,3,4,3]]
y8 = np.array([92])
#print(x1,x2)

#-------------------------------------------------------------------------------
#Activation functions
#-------------------------------------------------------------------------------

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
#Activation function derivatives
#-------------------------------------------------------------------------------
def d_sigmoid(s):
    #derivative of sigmoid
    return s * (1 - s)

def d_ReLU(s):
    #derivative of ReLU
    return np.where(s>=0,1,0)

def d_tanh(s):
    #derivative of tanh
    return 1-(s**2)

def d_linear(s):
    return 1

#E_nn = np.array([1,2,3])
#E_ref = np.array([1,1,1])
#m = len(E_ref)

def RMSE(E_nn,E_ref,m):     
    return np.sqrt(np.sum(np.square(E_nn-E_ref))/m)

def MAE(E_nn,E_ref,m):
    return np.sum(np.abs(E_nn-E_ref))/m

def MSE(E_nn,E_ref,m):
    return np.square(np.subtract(E_nn,E_ref)).mean()


    

#print('rmse',RMSE(E_nn,E_ref,m))
#print('mae',MAE(E_nn,E_ref,m))
#print('mse',MSE(E_nn,E_ref,m))



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

    def backward_prop(self,x,e_nn,e_ref):
        #output layer
        self.error_output_layer = e_nn-e_ref
        print('error_output_layer',self.error_output_layer)
        self.delta_output_layer = self.error_output_layer * d_linear(e_nn)
        print('delta_output_layer',self.delta_output_layer)
        #layer 2
        self.error_layer2 = self.delta_output_layer.dot(self.weights3.T)
        print('error_layer2',self.error_layer2)
        self.delta_layer2 = self.error_layer2 * d_sigmoid(self.layer2)
        print('delta_layer2',self.delta_layer2)
        #layer 1
        self.error_layer1 = self.delta_layer2.dot(self.weights2.T)
        self.delta_layer1 = self.error_layer1 * d_sigmoid(self.layer1)

        #weight update term
        dJdw1 =  x.T.dot(self.delta_layer1)
        dJdw2 =  (self.layer1.T).dot(self.delta_layer2)
        dJdw3 =  self.layer2.T.dot(self.delta_output_layer)

        return dJdw1,dJdw2,dJdw3

    def NN_optimize(self,dw1,dw2,dw3):
        self.weights1 -= 0.01*dw1
        self.weights2 -= 0.01*dw2
        self.weights3 -= 0.01*dw3


##node_list1 = [4,3,3,1]          #contains the layer sizes
#activations = [sigmoid,sigmoid,sigmoid]    
#nn_Ti = NeuralNetwork(node_list1,activations)
#nn_O  = NeuralNetwork(node_list,activations) 

#print(nn_Ti)

# nn_Ti.weights1 = np.ones(nn_Ti.weights1.shape)
# nn_O.weights1 = np.ones(nn_O.weights1.shape)
# nn_Ti.weights2 = np.ones(nn_Ti.weights2.shape)
# nn_O.weights2 = np.ones(nn_O.weights2.shape)

#print(nn_Ti.layer1,'\n',nn_Ti.layer2)
#print(nn_O.weights1,'\n',nn_O.weights2)

# no_of_atoms = len(test_xy[0][0])
# print(no_of_atoms)

#70-10-10-1 nn -----------------------------------------------------------------
file_name = 'structure1249.txt'
x = np.loadtxt(os.path.join('./symmetry_functions','%s') %file_name)
a = (x.reshape(6,1,70))
print(a[0])
E_ref = [[-4987.12739129]]
print(len(E_ref))
#print(len(x))
#print(x[0])

#o1 = nn_Ti.forward_prop(x1)
#o2 = nn_O.forward_prop(x2)
print(a[0].shape)
#print(o1)
node_list = [70,10,10,1]          #contains the layer sizes
activations = [sigmoid,sigmoid,linear]    
nn_Ti = NeuralNetwork(node_list,activations)
nn_O  = NeuralNetwork(node_list,activations)
print('Ti --',nn_Ti,'\n','O --',nn_O)


#print(o1,o2,o3)

#print('Weights 1 \n',nn_Ti.weights1,'\n','Weights 2 \n',nn_Ti.weights2,'\n','Weights 3 \n',nn_Ti.weights3)

def nn_switcher(x):
    val = len(x)
    no_of_ti_atoms = {
      # No of atoms : No of Ti atoms
                 6  :   2,                
                22  :   8,
                23  :   8,
                24  :   8,
                46  :   16,
                47  :   16,
                94  :   32,
                95  :   32
    }
    return no_of_ti_atoms[val]

index = nn_switcher(a)

#print('no of ti :',index)


def structure_forward_prop(a):
    output=[]
    for i in range(len(a)):
        if i<= (index-1):        
            output.append(nn_Ti.forward_prop(a[i]))
        else:
            output.append(nn_O.forward_prop(a[i]))
    return (output)

print(structure_forward_prop(a))
        
def train():
    output = sum(structure_forward_prop(a))  
    print('output',output,'e_ref',E_ref) 
    w1_t,w2_t,w3_t = nn_Ti.backward_prop(a[0],output,E_ref)
    #print('w1',w1_t,'w2',w2_t,'w3',w3_t)
    nn_Ti.NN_optimize(w1_t,w2_t,w3_t)
    w1_o,w2_o,w3_o = nn_O.backward_prop(a[5],output,E_ref)
    #print('w1',w1_o,'w2',w2_o,'w3',w3_o)
    nn_O.NN_optimize(w1_o,w2_o,w3_o)

for i in range(120):
    train()

print(structure_forward_prop(a))
print('output',sum(structure_forward_prop(a)),'e_ref',E_ref)
print('cost',sum(structure_forward_prop(a))-E_ref)
# for i in range(2):
#     train(a,E_ref)
# # print((output))
# # print(sum(output))

'''##Test for a small network------------------------------------------------------
node_list = [2,4,3,1]
activations = [sigmoid,sigmoid,linear]  
x = np.array([6.4,2.9]).reshape(1,2)
y = np.array([-88.9051]).reshape(1,1)
print(x.shape)
print(y.shape)
nn_test = NeuralNetwork(node_list,activations)

def train(a,b):
    out = nn_test.forward_prop(a)
    w1,w2,w3 = nn_test.backward_prop(a,out,b)
    print(w1,'\n',w2,'\n',w3)
    nn_test.NN_optimize(w1,w2,w3)

print('before',nn_test.forward_prop(x))
for i in range(5):
    train(x,y)

print('after',nn_test.forward_prop(x))
print('cost',nn_test.forward_prop(x)-y)
#------------------------------------------------------------------------------'''
#print(structure_forward_prop(a))


print('Time taken =',str((toc-tic)) + 'sec')