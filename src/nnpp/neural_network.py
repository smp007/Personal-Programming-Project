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

#-------------------------------------------------------------------------------
#Activation functions
#-------------------------------------------------------------------------------

def sigmoid(z):
    s = 1.0/(1+np.exp(-z))
    return s 

def ReLU(z):
    return np.maximum(0,z)

def tanh(z):
    #val = (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
    val = np.tanh(z)
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
    return 1.0 *(s>0)

def d_tanh(s):
    #derivative of tanh
    return 1-(s**2)

def d_linear(s):
    return 1

ActivationFunction = {
    'sigmoid' : (sigmoid,d_sigmoid),
    'ReLU'    : (ReLU,d_ReLU),
    'tanh'    : (tanh,d_tanh),
    'linear'  : (linear,d_linear),
    }

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


#-------------------------------------------------------------------------------

class NeuralNetwork:
    def __init__(self,nodelist,activations):
        '''Initialisation of neural network parameters'''
        self.input_nodes = nodelist[0]
        self.hidden_layer1_nodes = nodelist[1]        
        self.hidden_layer2_nodes = nodelist[2]        
        self.output_nodes = nodelist[-1]
        self.layer1_activation = ActivationFunction[activations[0]][0]
        self.layer2_activation = ActivationFunction[activations[1]][0]
        self.output_activation = ActivationFunction[activations[-1]][0]
        self.layer1_der_activation = ActivationFunction[activations[0]][1]
        self.layer2_der_activation = ActivationFunction[activations[1]][1]
        self.output_der_activation = ActivationFunction[activations[-1]][1]

        self.weights1 = np.random.randn(self.input_nodes,self.hidden_layer1_nodes)
        #print(self.weights1)
        self.bias1 = np.zeros((1,self.hidden_layer1_nodes),dtype=float)
        self.weights2 = np.random.randn(self.hidden_layer1_nodes,self.hidden_layer2_nodes)
        #print(self.weights2)
        self.bias2 = np.zeros((1,self.hidden_layer2_nodes),dtype=float)
        self.weights3 = np.random.randn(self.hidden_layer2_nodes,self.output_nodes)
        self.bias3 = np.zeros((1,self.output_nodes),dtype=float)

    def __repr__(self):
        return f"This is a {self.input_nodes}-{self.hidden_layer1_nodes}-{self.hidden_layer2_nodes}-{self.output_nodes} neural network"

        
    def forward_prop(self,x):
        self.layer1 = self.layer1_activation(np.dot(x,self.weights1))#+self.bias1)
        self.layer2 = self.layer2_activation(np.dot(self.layer1,self.weights2))#+self.bias2)
        output      = self.output_activation(np.dot(self.layer2,self.weights3))#+self.bias3)
        return output

    
    def backward_prop(self,x,e_nn,e_ref):
        #output layer
        self.error_output_layer = d_MSE(e_nn,e_ref)
        self.delta_output_layer = self.error_output_layer * self.output_der_activation(e_nn)
        #layer 2
        self.error_layer2 = self.delta_output_layer.dot(self.weights3.T)
        self.delta_layer2 = self.error_layer2 * self.layer2_der_activation(self.layer2)
        #layer 1
        self.error_layer1 = self.delta_layer2.dot(self.weights2.T)
        self.delta_layer1 = self.error_layer1 * self.layer1_der_activation(self.layer1)

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
    """
    Returns the index which is used the switch the atomic NNs
    ie. if there are 6 atoms out of which 2 are Ti and 4 are O,we need to switch
    from Ti_NN to O_NN after 2 elements in the 6 element array.

    Arguments : 
    x -- 2d array of symmetry functions[G] with shape [n x 70]

    Returns : 
    no_of_ti_atoms[val] -- no of Ti atoms in the structure,so as to switch NN 
    after that value

    """
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
    """
    Predicts the total structural energy with the help of atomic NNs
    
    Arguments : 
    a -- 3d array of symmetry functions[G] with shape [n x 1 x 70]
    nn1 -- object of class Neuralnetwork
    nn2 -- another object of class Neuralnetwork
    index -- return value of nn_switcher function

    Returns : 
    output -- array of predicted individual atomic energies
    ie if there are 6 atoms out of which 2 are Ti and 4 are O,first 2 [G]s of Ti 
    are passed through Ti_NN and remaining 4 [G]s of O are passed through O_NN 
    and all the 6 values are stored in an array 

    """
    output=[]
    for i in range(len(a)):
        if i<= (index-1):        
            output.append(nn1.forward_prop(a[i]))
        else:
            output.append(nn2.forward_prop(a[i]))
    return (output)

def weight_array_init(nn1):
    r_,d_ = np.zeros(nn1.weights1.shape),np.zeros(nn1.weights1.shape)
    s_,e_ = np.zeros(nn1.weights2.shape),np.zeros(nn1.weights2.shape)
    t_,f_ = np.zeros(nn1.weights3.shape),np.zeros(nn1.weights3.shape)
    return r_,d_,s_,e_,t_,f_ 



def structure_backward_prop(a,nn1,nn2,index,output,e_ref,learning_rate):#try only for bkwd prop sum
    r,d,s,e,t,f = weight_array_init(nn1)
    for i in range(len(a)):
        if i<= (index-1):
            r += nn1.backward_prop(a[i],output,e_ref)[0]
            #print('r',r)
            s += nn1.backward_prop(a[i],output,e_ref)[1]
            #print('s',s)
            t += nn1.backward_prop(a[i],output,e_ref)[2]
        else:
            d += nn2.backward_prop(a[i],output,e_ref)[0]
            e += nn2.backward_prop(a[i],output,e_ref)[1]
            #print('e',e)
            f += nn2.backward_prop(a[i],output,e_ref)[2]
    nn1.NN_optimize(r,s,t,learning_rate)
    nn2.NN_optimize(d,e,f,learning_rate)

def train(nn1,nn2,index,a,e_ref,learning_rate):#working
    output = sum(structure_forward_prop(a,nn1,nn2,index))  
    print('output',output,'e_ref',e_ref) 
    w1_t,w2_t,w3_t = nn1.backward_prop(a[0],output,e_ref)
    print('w1',w1_t,'w2',w2_t,'w3',w3_t)
    nn1.NN_optimize(w1_t,w2_t,w3_t,learning_rate)
    w1_o,w2_o,w3_o = nn2.backward_prop(a[index],output,e_ref)
    print('w1',w1_o,'w2',w2_o,'w3',w3_o)
    nn2.NN_optimize(w1_o,w2_o,w3_o,learning_rate)


def train2(nn1,nn2,index,a,e_ref,learning_rate):#try only
    output = sum(structure_forward_prop(a,nn1,nn2,index))
    structure_backward_prop(a,nn1,nn2,index,output,e_ref,learning_rate)


#70-10-10-1 nn -----------------------------------------------------------------
file_name = 'structure1249.txt'
x = np.loadtxt(os.path.join('./symmetry_functions','%s') %file_name)
n = len(x)
A = x.reshape(n,1,70)
print(A)
E_ref = [[-4987.12739129]]  #-19960.66173260
print(A[0].shape)
#print(o1)
node_list = [70,10,10,1]          #contains the layer sizes
activations = ['sigmoid','sigmoid','linear']    
nn_Ti = NeuralNetwork(node_list,activations)
nn_O  = NeuralNetwork(node_list,activations)
print('Ti --',nn_Ti,'\n','O --',nn_O)


#print(o1,o2,o3)

#print('Weights 1 \n',nn_Ti.weights1,'\n','Weights 2 \n',nn_Ti.weights2,'\n','Weights 3 \n',nn_Ti.weights3)

index = nn_switcher(A)

#print('no of ti :',index)






for i in range(1250):
    #train(nn_Ti,nn_O,index,A,E_ref)
    train2(nn_Ti,nn_O,index,A,E_ref,learning_rate=4e-5)

print(structure_forward_prop(A,nn_Ti,nn_O,index))
print('output',sum(structure_forward_prop(A,nn_Ti,nn_O,index)),'e_ref',E_ref)
print('cost',sum(structure_forward_prop(A,nn_Ti,nn_O,index))-E_ref)
# for i in range(2):
#     train(a,E_ref)
# # print((output))

# # print(sum(output))
#70-10-10-1 nn -----------------------------------------------------------------'''





print('Time taken =',str((toc-tic)) + 'sec')

def stochastic_gradient_descent(a_list,e_ref_list,learning_rate,epochs):

    m = len(e_ref_list)
    cost_extract = np.zeros(epochs)
    for i in range(epochs):
        cost = 0
        e_nn_list = []
        for j in range(m):
            train2(nn_Ti,nn_O,index,a_list[j],e_ref_list[j],learning_rate)
            e_nn_list.append(sum(structure_forward_prop(a_list[j],nn_Ti,nn_O,index)))
        print('predicted',e_nn_list,'\n','reference',e_ref_list) 
        cost = MSE(np.asarray(e_nn_list),np.asarray(e_ref_list))        
        cost_extract[i] = cost

    return cost_extract


'''

file_name_list = ['structure0005.txt','structure0004.txt','structure0003.txt','structure0002.txt','structure0001.txt']#,'structure1249.txt']
X_list = [np.loadtxt(os.path.join('./symmetry_functions','%s') %file_name) for file_name in file_name_list]
A_list = [x.reshape(len(x),1,70) for x in X_list]
print(A_list[1].shape)
E_ref_list = [[-19960.74194513],[-19960.78597929],[-19960.75811714],[-19960.69526834],[-19960.66173260]]#,[-4987.12739129]]
#print(A_list[2].shape)
#print(len(E_ref_list))

#E_ref_list = np.asarray([[-4987.12739129],[-78964.89340133],[-19960.66173260]])
#E_nn_list = np.asarray([[-4987],[-78964],[-19960]])

#print(MSE(E_nn_list,E_ref_list))
#print(stochastic_gradient_descent(A_list,E_ref_list,learning_rate=4e-5,epochs=150))

cost_variation = stochastic_gradient_descent(A_list,E_ref_list,learning_rate=4e-5,epochs=40)

fig = plt.figure(figsize = (6,4),dpi =150)
plt.plot(cost_variation,'o')
plt.xlabel('epochs')
plt.ylabel('cost')
plt.title('Cost variation for a very small dataset')
plt.show()
fig.tight_layout()
fig.savefig('cost_variation_5_dataset.png')'''