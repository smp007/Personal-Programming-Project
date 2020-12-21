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

E_nn = np.array([1,2,3])
E_ref = np.array([1,1,1])


def RMSE(predictions,references):
    m = len(references)     
    return np.sqrt(np.sum(np.square(predictions-references))/m)

def MAE(predictions,references):
    m = len(references)
    return np.sum(np.abs(predictions-references))/m

def MSE(predictions,references):
    m = len(references)
    return 0.5*np.sum(np.square((predictions-references)))/m

def d_MSE(predictions,references):
    return(predictions-references)

    

print('rmse',RMSE(E_nn,E_ref))
print('mae',MAE(E_nn,E_ref))
print('mse',MSE(E_nn,E_ref))



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
        self.error_output_layer = d_MSE(e_nn,e_ref)
        ##print('error_output_layer',self.error_output_layer)
        self.delta_output_layer = self.error_output_layer * d_linear(e_nn)
        ##print('delta_output_layer',self.delta_output_layer)
        #layer 2
        self.error_layer2 = self.delta_output_layer.dot(self.weights3.T)
        ##print('error_layer2',self.error_layer2)
        self.delta_layer2 = self.error_layer2 * d_sigmoid(self.layer2)
        ##print('delta_layer2',self.delta_layer2)
        #layer 1
        self.error_layer1 = self.delta_layer2.dot(self.weights2.T)
        self.delta_layer1 = self.error_layer1 * d_sigmoid(self.layer1)

        #weight update term(derivatives)
        dJdw1 =  x.T.dot(self.delta_layer1)
        dJdw2 =  (self.layer1.T).dot(self.delta_layer2)
        dJdw3 =  self.layer2.T.dot(self.delta_output_layer)

        return dJdw1,dJdw2,dJdw3

    def NN_optimize(self,dw1,dw2,dw3,learning_rate):
        self.weights1 -= learning_rate * dw1
        self.weights2 -= learning_rate * dw2
        self.weights3 -= learning_rate * dw3

#gradient checking
    def collect_parameters(self):
        #to convert weight matrix into a flattened n x 1 array
        parameters = np.concatenate((self.weights1.reshape(-1),self.weights2.reshape(-1),self.weights3.reshape(-1)))
        return parameters
    
    def set_parameters(self,parameters):
        #w1 
        w1_first = 0
        w1_last = self.input_nodes * self.hidden_layer1_nodes
        self.weights1 = parameters[w1_first:w1_last].reshape(self.input_nodes,self.hidden_layer1_nodes)
        #w2 
        w2_first = w1_last
        w2_last = w2_first + (self.hidden_layer1_nodes*self.hidden_layer2_nodes)
        self.weights2 = parameters[w2_first:w2_last].reshape(self.hidden_layer1_nodes,self.hidden_layer2_nodes)
        #w3 
        w3_first = w2_last
        w3_last = w3_first +(self.hidden_layer2_nodes*self.output_nodes)
        self.weights3 = parameters[w3_first:w3_last].reshape(self.hidden_layer2_nodes,self.output_nodes)
        return 0

    def analytical_gradients(self,x,e_nn,e_ref):
        '''returns the gradients found by the backprop algorithm'''
        djdw1,djdw2,djdw3 = self.backward_prop(x,e_nn,e_ref)
        return np.concatenate((djdw1.reshape(-1),djdw2.reshape(-1),djdw3.reshape(-1))) #flattens the gradient values into a 1D array


epsilon = 1e-5    
def numerical_gradients(nn,x,e_nn,e_ref):
    parameter_values          = nn.collect_parameters()      #collects the weights of the NN
    numerical_gradient_values = np.zeros(parameter_values.shape) #to store the numerical gradients of corresponding perturbing weight
    small_change_vector       = np.zeros(parameter_values.shape) #vector for disturbing the weights(one at a time)   

    for i in range(len(parameter_values)):
        #we need to change the i th element of small_change_vector to epsilon
        small_change_vector[i] = epsilon
        #now we add this change to the parameter values and pass it to set_parameters function and calculate loss
        new_parameter_values = parameter_values+small_change_vector
        nn.set_parameters(new_parameter_values)
        e_nn = nn.forward_prop(x)
        loss_plus = MSE(e_nn,e_ref)
        #now we subtract the change and find loss
        new_parameter_values = parameter_values-small_change_vector
        nn.set_parameters(new_parameter_values)
        e_nn = nn.forward_prop(x)
        loss_minus = MSE(e_nn,e_ref)
        #derivative using central difference method
        numerical_gradient_values[i] = (loss_plus-loss_minus)/(2*epsilon)
        small_change_vector[i] = 0
    
    nn.set_parameters(parameter_values)

    return numerical_gradient_values



#------------------------------------------------------------------------------- dec 19 night
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

def structure_forward_prop(a,nn1,nn2,index):
    output=[]
    for i in range(len(a)):
        if i<= (index-1):        
            output.append(nn1.forward_prop(a[i]))
        else:
            output.append(nn2.forward_prop(a[i]))
    return (output)

def train(nn1,nn2,index,a,e_ref,learning_rate):
    output = sum(structure_forward_prop(a,nn1,nn2,index))  
    print('output',output,'e_ref',e_ref) 
    w1_t,w2_t,w3_t = nn1.backward_prop(a[0],output,e_ref)
    #print('w1',w1_t,'w2',w2_t,'w3',w3_t)
    nn1.NN_optimize(w1_t,w2_t,w3_t,learning_rate)
    w1_o,w2_o,w3_o = nn2.backward_prop(a[index],output,e_ref)
    #print('w1',w1_o,'w2',w2_o,'w3',w3_o)
    nn2.NN_optimize(w1_o,w2_o,w3_o,learning_rate)


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

'''#70-10-10-1 nn -----------------------------------------------------------------
file_name = 'structure1249.txt'
x = np.loadtxt(os.path.join('./symmetry_functions','%s') %file_name)
n = len(x)
A = x.reshape(n,1,70)
print(A)
E_ref = [[0]]  #-19960.66173260
#print(len(E_ref))
#print(len(x))
#print(x[0])

#o1 = nn_Ti.forward_prop(x1)
#o2 = nn_O.forward_prop(x2)
print(A[0].shape)
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

index = nn_switcher(A)

#print('no of ti :',index)


def structure_forward_prop(a,nn1,nn2):
    output=[]
    for i in range(len(a)):
        if i<= (index-1):        
            output.append(nn1.forward_prop(a[i]))
        else:
            output.append(nn2.forward_prop(a[i]))
    return (output)

print(structure_forward_prop(A,nn_Ti,nn_O))
        
def train(nn1,nn2,index,a,e_ref):
    output = sum(structure_forward_prop(a,nn1,nn2))  
    print('output',output,'e_ref',e_ref) 
    w1_t,w2_t,w3_t = nn1.backward_prop(a[0],output,e_ref)
    #print('w1',w1_t,'w2',w2_t,'w3',w3_t)
    nn1.NN_optimize(w1_t,w2_t,w3_t,learning_rate)
    w1_o,w2_o,w3_o = nn2.backward_prop(a[0],output,e_ref)
    #print('w1',w1_o,'w2',w2_o,'w3',w3_o)
    nn2.NN_optimize(w1_o,w2_o,w3_o,learning_rate)

for i in range(130):
    train(nn_Ti,nn_O,index,A,E_ref)

print(structure_forward_prop(A,nn_Ti,nn_O))
print('output',sum(structure_forward_prop(A,nn_Ti,nn_O)),'e_ref',E_ref)
print('cost',sum(structure_forward_prop(A,nn_Ti,nn_O))-E_ref)
# for i in range(2):
#     train(a,E_ref)
# # print((output))
# # print(sum(output))
#70-10-10-1 nn -----------------------------------------------------------------'''

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
    nn_test.NN_optimize(w1,w2,w3,learning_rate)

print('before',nn_test.forward_prop(x))
for i in range(5):
    train(x,y)

print('after',nn_test.forward_prop(x))
print('cost',nn_test.forward_prop(x)-y)
#------------------------------------------------------------------------------'''
#print(structure_forward_prop(a))


print('Time taken =',str((toc-tic)) + 'sec')

'''#numerical gradient check starts------------------------------------------------

node_list = [2,3,5,1]
activations = [sigmoid,sigmoid,linear]  
x = np.array([2,20]).reshape(1,2)
y = np.array([1000]).reshape(1,1)
print(x.shape)
print(y.shape)
nn_test = NeuralNetwork(node_list,activations)

print('Weights 1 \n',nn_test.weights1,'\n','Weights 2 \n',nn_test.weights2,'\n','Weights 3 \n',nn_test.weights3)

def train(a,b):
    out = nn_test.forward_prop(a)
    w1,w2,w3 = nn_test.backward_prop(a,out,b)
    print(w1,'\n',w2,'\n',w3)
    #print(nn_test.backward_prop(a,out,b))
    ################nn_test.NN_optimize(w1,w2,w3,learning_rate)
    #print(nn_test.backward_prop(a,out,b))

print('before',nn_test.forward_prop(x))
for i in range(1):
    train(x,y)

print('after',nn_test.forward_prop(x))
print('error',nn_test.forward_prop(x)-y)


print('Weights 1 \n',nn_test.weights1,'\n','Weights 2 \n',nn_test.weights2,'\n','Weights 3 \n',nn_test.weights3)
print(nn_test.set_parameters(nn_test.collect_parameters()))
e_nn = nn_test.forward_prop(x)
print(nn_test.analytical_gradients(x,e_nn,y))
print(numerical_gradients(nn_test,x,e_nn,y))
print(nn_test.analytical_gradients(x,e_nn,y)-numerical_gradients(nn_test,x,e_nn,y))
#print(np.linalg.norm(nn_test.analytical_gradients(x,e_nn,y)-numerical_gradients(nn_test,x,e_nn,y)))

#numerical gradient check ends--------------------------------------------------'''