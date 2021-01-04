import os
import numpy as np 
import time
import matplotlib.pyplot as plt 
import random

path = './data_set_TiO2'
file_list = (os.listdir(path))
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
    test_set = filelist[:]
    train_energy = [] 
    test_energy = energylist
    while len(train_set) < n_train_set :
        indox = np.random.randint(0,len(test_set))     #randrange(len(train_set))
        train_set.append(test_set.pop(indox))
        train_energy.append(test_energy.pop(indox))
    return train_set,test_set,(train_energy),(test_energy)


#splitting the data and storing it in 4 arrays 
    #a)train input    b)train output    c)test input    d)test output
a,b,c,d = test_train_split(file_list,energy_list,split=99)

#loading the symmetry function vectors from the corresponding files from which energy value is taken
test_xy = ([(np.loadtxt(os.path.join('./sym_fun_all_2','%s') %(x[:-3]+'txt')),y)for x,y in zip(b,d)]) 
train_xy = ([(np.loadtxt(os.path.join('./sym_fun_all_2','%s') %(x[:-3]+'txt')),y)for x,y in zip(a,c)])

#train set arrays-----reshaping input data in the form of (nx1x70) array and shuffling input and output with the same seed
inputs,outputs = zip(*train_xy)
inputs_ = np.asarray([x.reshape(len(x),1,70) for x in inputs]) #to np arrays
outputs_ = np.asarray([*outputs])
np.random.seed(60)
r = random.random()
random.shuffle(inputs_,lambda:r)
random.shuffle(outputs_,lambda:r)

#test set arrays-----reshaping input data in the form of (nx1x70) array and shuffling input and output with the same seed
inputs2,outputs2 = zip(*test_xy)
inputs2_= np.asarray([x.reshape(len(x),1,70) for x in inputs2])
outputs2_= np.asarray([*outputs2])
r = random.random()
random.shuffle(inputs2_,lambda:r)
random.shuffle(outputs2_,lambda:r)

#min max normalization----------------------------------------------------------
def min_max_norm(G):
    g_max_array = []
    for i in range(len(inputs_)):
        g_max_array.append(np.max(inputs_[i],axis=0))                 #finding max vector of each dataset in training set
    g_max_array_np = np.array(g_max_array)
    G_max = np.max(g_max_array_np,axis=0)                             #finding max vector of the entire training set

    g_min_array = []
    for i in range(len(inputs_)):
        g_min_array.append(np.min(inputs_[i],axis=0))                 #finding min vector of each dataset in training set
    g_min_array_np = np.array(g_min_array)
    G_min = np.min(g_min_array_np,axis=0)                             #finding min vector of the entire training set

    for i in range(len(G)):
        for j in range(len(G[i])):
            G[i][j] = (2*(G[i][j]-G_min)/(G_max-G_min))-1             #scales the G vector between [-1,1]

min_max_norm(inputs_)
min_max_norm(inputs2_)

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
    #return sigmoid(s) * (1-sigmoid(s))
    return s * (1-s)

def d_ReLU(s):
    #derivative of ReLU
    return 1.0 *(s>0)

def d_tanh(s):
    #derivative of tanh
    #return 1-((np.tanh(s))**2)
    return 1 - (s**2)

def d_linear(s):
    return 1

ActivationFunction = {
    'sigmoid' : (sigmoid,d_sigmoid),
    'ReLU'    : (ReLU,d_ReLU),
    'tanh'    : (tanh,d_tanh),
    'linear'  : (linear,d_linear),
    }

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
    return np.asarray(predictions-references)

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
        self.bias1 = np.zeros((1,self.hidden_layer1_nodes))
        self.weights2 = np.random.randn(self.hidden_layer1_nodes,self.hidden_layer2_nodes)
        self.bias2 = np.zeros((1,self.hidden_layer2_nodes))
        self.weights3 = np.random.randn(self.hidden_layer2_nodes,self.output_nodes)
        self.bias3 = np.zeros((1,self.output_nodes))
        

    def __repr__(self):
        return f"This is a {self.input_nodes}-{self.hidden_layer1_nodes}-{self.hidden_layer2_nodes}-{self.output_nodes} neural network"

        
    def forward_prop(self,x):
        self.layer1 = self.layer1_activation(np.dot(x,self.weights1)+self.bias1)
        self.layer2 = self.layer2_activation(np.dot(self.layer1,self.weights2)+self.bias2)
        output      = self.output_activation(np.dot(self.layer2,self.weights3)+self.bias3)
        return output

    
    def backward_prop(self,x,e_nn,e_ref):

        self.error_output_layer = d_MSE(e_nn,e_ref)
        self.delta_output_layer = self.error_output_layer * self.output_der_activation(e_nn)
        #layer 2
        self.error_layer2 = self.delta_output_layer.dot(self.weights3.T)
        self.delta_layer2 = self.error_layer2 * self.layer2_der_activation(self.layer2)
        #layer 1
        self.error_layer1 = self.delta_layer2.dot(self.weights2.T)
        self.delta_layer1 = self.error_layer1 * self.layer1_der_activation(self.layer1)

        #weight update term(derivatives)
        dJdw1 =  (x.T).dot(self.delta_layer1)
        dJdw2 =  (self.layer1.T).dot(self.delta_layer2)
        dJdw3 =  (self.layer2.T).dot(self.delta_output_layer)

        dJdb1 = self.delta_layer1
        dJdb2 = self.delta_layer2
        dJdb3 = self.delta_output_layer

        return dJdw1,dJdw2,dJdw3,dJdb1,dJdb2,dJdb3

    def NN_optimize(self,dw1,dw2,dw3,db1,db2,db3,learning_rate):
        self.weights1 -= learning_rate * dw1
        self.weights2 -= learning_rate * dw2
        self.weights3 -= learning_rate * dw3
        self.bias1 -= learning_rate * db1
        self.bias2 -= learning_rate * db2
        self.bias3 -= learning_rate * db3

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

def structure_forward_prop(a,nn1,nn2):
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
    index = nn_switcher(a)
    for i in range(len(a)):
        
        if i<= (index-1):
                      
            output.append(nn1.forward_prop(a[i]))
        else:
            output.append(nn2.forward_prop(a[i]))
    return (output)

def weight_array_init(nn1):
    w1_t_,w1_o_ = np.zeros(nn1.weights1.shape),np.zeros(nn1.weights1.shape)
    w2_t_,w2_o_ = np.zeros(nn1.weights2.shape),np.zeros(nn1.weights2.shape)
    w3_t_,w3_o_ = np.zeros(nn1.weights3.shape),np.zeros(nn1.weights3.shape)
    return w1_t_,w1_o_,w2_t_,w2_o_,w3_t_,w3_o_ 

def bias_array_init(nn1):
    b1_t_,b1_o_ = np.zeros(nn1.bias1.shape),np.zeros(nn1.bias1.shape)
    b2_t_,b2_o_ = np.zeros(nn1.bias2.shape),np.zeros(nn1.bias2.shape)
    b3_t_,b3_o_ = np.zeros(nn1.bias3.shape),np.zeros(nn1.bias3.shape)
    return b1_t_,b1_o_,b2_t_,b2_o_,b3_t_,b3_o_ 


#momentum

def velocity_init(nn1):
    v_w1_t,v_w1_o,v_b1_t,v_b1_o = np.zeros(nn1.weights1.shape),np.zeros(nn1.weights1.shape),np.zeros(nn1.bias1.shape),np.zeros(nn1.bias1.shape)
    v_w2_t,v_w2_o,v_b2_t,v_b2_o = np.zeros(nn1.weights2.shape),np.zeros(nn1.weights2.shape),np.zeros(nn1.bias2.shape),np.zeros(nn1.bias2.shape)
    v_w3_t,v_w3_o,v_b3_t,v_b3_o = np.zeros(nn1.weights3.shape),np.zeros(nn1.weights3.shape),np.zeros(nn1.bias3.shape),np.zeros(nn1.bias3.shape)
    return v_w1_t,v_w2_t,v_w3_t,v_b1_t,v_b2_t,v_b3_t,v_w1_o,v_w2_o,v_w3_o,v_b1_o,v_b2_o,v_b3_o

def momentum(beta,nn1,nn2,a,e_ref,learning_rate):
    output = sum(structure_forward_prop(a,nn1,nn2))    
    w_b_params = np.asarray(structure_backward_prop(a,nn1,nn2,output,e_ref,learning_rate))  
    v_params = np.asarray(velocity_init(nn1))
    v_params = beta*v_params +(1-beta)*w_b_params
    nn1.NN_optimize(*v_params[:6],learning_rate)
    nn2.NN_optimize(*v_params[6:],learning_rate)


def structure_backward_prop(a,nn1,nn2,output,e_ref,learning_rate):#try only for bkwd prop sum
    w1_t,w1_o,w2_t,w2_o,w3_t,w3_o  = weight_array_init(nn1)
    b1_t,b1_o,b2_t,b2_o,b3_t,b3_o  =  bias_array_init(nn1)
    index = nn_switcher(a)
    for i in range(len(a)):
        if i<= (index-1):
            grad_params_nn1 = nn1.backward_prop(a[i],output,e_ref)
            w1_t += grad_params_nn1[0]    
            w2_t += grad_params_nn1[1]          
            w3_t += grad_params_nn1[2]
            b1_t += grad_params_nn1[3]    
            b2_t += grad_params_nn1[4]          
            b3_t += grad_params_nn1[5]
        else:
            grad_params_nn2 = nn2.backward_prop(a[i],output,e_ref)
            w1_o += grad_params_nn2[0]
            w2_o += grad_params_nn2[1]            
            w3_o += grad_params_nn2[2]
            b1_o += grad_params_nn2[3]
            b2_o += grad_params_nn2[4]            
            b3_o += grad_params_nn2[5]  
    val = len(a)-index
    w_b_parameters = [w1_t,w2_t,w3_t,b1_t,b2_t,b3_t,w1_o,w2_o,w3_o,b1_o,b2_o,b3_o]  
    return w_b_parameters
    

'''def train(nn1,nn2,a,e_ref,learning_rate):#working
    index = nn_switcher(a)
    output = sum(structure_forward_prop(a,nn1,nn2))  
    #print('output',output,'e_ref',e_ref) 
    w1_t,w2_t,w3_t,_,_,_ = nn1.backward_prop(a[0],output,e_ref)
    #print('w1',w1_t,'w2',w2_t,'w3',w3_t)
    nn1.NN_optimize(w1_t,w2_t,w3_t,0,0,0,learning_rate)
    w1_o,w2_o,w3_o,_,_,_ = nn2.backward_prop(a[index],output,e_ref)
    #print('w1',w1_o,'w2',w2_o,'w3',w3_o)
    nn2.NN_optimize(w1_o,w2_o,w3_o,0,0,0,learning_rate)'''


def train(nn1,nn2,a,e_ref,learning_rate):#try only
    #print(index)
    output = sum(structure_forward_prop(a,nn1,nn2))
    
    w1_T,w2_T,w3_T,b1_T,b2_T,b3_T,w1_O,w2_O,w3_O,b1_O,b2_O,b3_O = \
    structure_backward_prop(a,nn1,nn2,output,e_ref,learning_rate)       
    nn1.NN_optimize(w1_T,w2_T,w3_T,b1_T,b2_T,b3_T,learning_rate)
    nn2.NN_optimize(w1_O,w2_O,w3_O,b1_O,b2_O,b3_O,learning_rate)

#exit(02

def stochastic_gradient_descent(nn1,nn2,g_list,e_ref_list,learning_rate,epochs):
    print('############ SGD ################')
    m = len(e_ref_list)
    cost_extract = np.zeros(epochs)
    for i in range(epochs):
        cost = 0
        e_nn_list = []
        for j in range(m):
            e_nn_list.append(sum(structure_forward_prop(g_list[j],nn1,nn2)))
            #print(e_nn_list)
            train(nn1,nn2,g_list[j],e_ref_list[j],learning_rate)
            
        
        #print('predicted',e_nn_list,'\n','reference',e_ref_list) 
        #cost = RMSE(np.asarray(e_nn_list),np.asarray(e_ref_list)) 
        cost = MSE(np.asarray(np.concatenate(e_nn_list).reshape(-1).tolist()),np.asarray(e_ref_list)) 
          
        print('{0: <6}'.format('epoch ='),i,'----','{0: <4}'.format('lr ='),learning_rate,'----','{0: <6}'.format('cost ='),cost)     
        cost_extract[i] = cost
        #np.random.shuffle(g_list)
        r = random.random()
        random.shuffle(g_list,lambda :r)
        random.shuffle(e_ref_list,lambda:r)
    return cost_extract,learning_rate

def SGD_momentum(nn1,nn2,g_list,e_ref_list,learning_rate,epochs,beta):
    print('############ momentum ################')
    m = len(e_ref_list)
    cost_extract = np.zeros(epochs)
    for i in range(epochs):
        cost = 0
        e_nn_list = []
        v_params = np.asarray(velocity_init(nn1))
        for j in range(m):
            e_nn_list.append(sum(structure_forward_prop(g_list[j],nn1,nn2)))
            #print(e_nn_list)
            output = sum(structure_forward_prop(g_list[j],nn1,nn2))    
            w_b_params = np.asarray(structure_backward_prop(g_list[j],nn1,nn2,output,e_ref_list[j],learning_rate))  
            
            v_params = beta*v_params +(1-beta)*w_b_params
            # if (j==0 or j==1) and (i==0 or i==1):
            #     print(v_params[1])
            #print(v_params[5].shape)
            nn1.NN_optimize(*v_params[:6],learning_rate)
            nn2.NN_optimize(*v_params[6:],learning_rate)
            
        
        #print('predicted',e_nn_list,'\n','reference',e_ref_list) 
        cost = MSE(np.asarray(np.concatenate(e_nn_list).reshape(-1).tolist()),np.asarray(e_ref_list)) 
        print('{0: <6}'.format('epoch ='),i,'----','{0: <4}'.format('lr ='),learning_rate,'----','{0: <6}'.format('cost ='),cost)     
        cost_extract[i] = cost
        #np.random.shuffle(g_list)
        r = random.random()
        random.shuffle(g_list,lambda :r)
        random.shuffle(e_ref_list,lambda:r)
    return cost_extract,learning_rate

def minibatch_gradient_descent(nn1,nn2,g_list,e_ref_list,learning_rate,batchSize,epochs):
    print('############ Mini batch GD ################')
    m = len(e_ref_list)
    batchNumber = int(m/batchSize)
    cost_extract = np.zeros(epochs)
    for i in range(epochs):
        cost = 0
        # indexes = np.random.permutation(m)
        # g_list = g_list[indexes]
        # e_ref_list = e_ref_list[indexes]
        e_nn_list = []
        for j in range(0,m,batchSize):
            w1_t_mb,w1_o_mb,w2_t_mb,w2_o_mb,w3_t_mb,w3_o_mb  = weight_array_init(nn1)
            b1_t_mb,b1_o_mb,b2_t_mb,b2_o_mb,b3_t_mb,b3_o_mb  =  bias_array_init(nn1)
            
            
            g_list_j = g_list[j:j+batchSize]
            e_ref_list_j = e_ref_list[j:j+batchSize]
            for k in range(len(g_list_j)):
                output = sum(structure_forward_prop(g_list_j[k],nn1,nn2))
                w1_T,w2_T,w3_T,b1_T,b2_T,b3_T,w1_O,w2_O,w3_O,b1_O,b2_O,b3_O = \
                    structure_backward_prop(g_list_j[k],nn1,nn2,output,e_ref_list_j[k],learning_rate)
                w1_t_mb += w1_T #summing over the dw and db  over the batch to take the mean
                w2_t_mb += w2_T
                w3_t_mb += w3_T
                w1_o_mb += w1_O
                w2_o_mb += w2_O
                w3_o_mb += w3_O
                b1_t_mb += b1_T
                b2_t_mb += b2_T
                b3_t_mb += b3_T
                b1_o_mb += b1_O
                b2_o_mb += b2_O
                b3_o_mb += b3_O
                e_nn_list.append(sum(structure_forward_prop(g_list_j[k],nn1,nn2)))
            parameters_nn1 = np.asarray([w1_t_mb,w2_t_mb,w3_t_mb,b1_t_mb,b2_t_mb,b3_t_mb])/batchSize #mean dw,mean db
            parameters_nn2 = np.asarray([w1_o_mb,w2_o_mb,w3_o_mb,b1_o_mb,b2_o_mb,b3_o_mb])/batchSize
            nn1.NN_optimize(*parameters_nn1,learning_rate)    
            nn2.NN_optimize(*parameters_nn2,learning_rate) 
            
        #print('predicted',len(e_nn_list),'\n','reference',len(e_ref_list)) 
        cost = MSE(np.asarray(np.concatenate(e_nn_list).reshape(-1).tolist()),np.asarray(e_ref_list)) 
        print('{0: <6}'.format('epoch ='),i,'----','{0: <4}'.format('lr ='),learning_rate,'----','{0: <6}'.format('cost ='),cost)     
        cost_extract[i] = cost
        r = random.random()
        random.shuffle(g_list,lambda :r)
        random.shuffle(e_ref_list,lambda:r)
        
    return cost_extract,learning_rate

def weight_initialize(nn,w_1,w_2,w_3):
    nn.weights1 = w_1
    nn.weights2 = w_2
    nn.weights3 = w_3

Ti_weights = np.load('dict_ti_weights.npz')
O_weights = np.load('dict_O_weights.npz')
w1_ti = Ti_weights['w1']
w2_ti = Ti_weights['w2']
w3_ti = Ti_weights['w3']
w1_ox = O_weights['w1']
w2_ox = O_weights['w2']
w3_ox = O_weights['w3']

node_list_1 = [70,9,9,1]          #contains the layer sizes
activations_1 = ['tanh','tanh','linear']  
np.random.seed(113)  
nn_Ti_1 = NeuralNetwork(node_list_1,activations_1)
weight_initialize(nn_Ti_1,w1_ti,w2_ti,w3_ti)
nn_O_1  = NeuralNetwork(node_list_1,activations_1)
weight_initialize(nn_O_1,w1_ox,w2_ox,w3_ox)
#print(nn_Ti_1.weights3)

node_list_2 = [70,9,9,1]          #contains the layer sizes
activations_2 = ['tanh','tanh','linear']    
np.random.seed(1)
nn_Ti_2 = NeuralNetwork(node_list_2,activations_2)
weight_initialize(nn_Ti_2,w1_ti,w2_ti,w3_ti)
nn_O_2 = NeuralNetwork(node_list_2,activations_2)
weight_initialize(nn_O_2,w1_ox,w2_ox,w3_ox)

#node_list_2 = [70,9,9,1]          #contains the layer sizes
#activations_2 = ['tanh','tanh','linear']    
#np.random.seed(115)
nn_Ti_3 = NeuralNetwork(node_list_2,activations_2)
weight_initialize(nn_Ti_3,w1_ti,w2_ti,w3_ti)
#np.random.seed(213)
nn_O_3 = NeuralNetwork(node_list_2,activations_2)
weight_initialize(nn_O_3,w1_ox,w2_ox,w3_ox)

def predict_energy(g_list,e_ref_list):
    #r = random.random()
    #random.shuffle([*g_list],lambda :r)
    #random.shuffle([*e_ref_list],lambda:r)
    m = len(e_ref_list)

    e_predicted_list = []
    e_predicted_list_2 = []
    for j in range(m):
        e_predicted_list.append(sum(structure_forward_prop(g_list[j],nn_Ti_3,nn_O_3)))
        ##e_predicted_list_2.append([(*structure_forward_prop(g_list[j],nn_Ti_3,nn_O_3))])
        ##e_predicted_list_2.append('#####################')
        #print(e_predicted_list)
    print('#######################################################','\n')
    #print('predicted',np.concatenate(e_predicted_list).reshape(-1).tolist(),'\n\n','reference',e_ref_list)
    ##print([('p',a,'ref',b,'err',c) for a,b,c in zip(np.concatenate(e_predicted_list).reshape(-1).tolist(),e_ref_list,np.concatenate(e_predicted_list).reshape(-1).tolist()-e_ref_list)])
    ##print((np.concatenate(e_predicted_list).reshape(-1).tolist()-e_ref_list).tolist())
    ##print(e_predicted_list_2)
    print('#######################################################','\n')
    test_error = (np.concatenate(e_predicted_list).reshape(-1).tolist()-e_ref_list).tolist()
    print('mse',(np.mean(np.square(test_error))))
    #cost = RMSE(np.asarray(e_predicted_list),np.asarray(e_ref_list))
    #print(cost)
    return e_predicted_list,e_ref_list,test_error

predict_energy(inputs2_,outputs2_)
epochs_val = 60
##predict_energy(inputs_,outputs_)
#cost_variation_mbgd,lr_mbgd = minibatch_gradient_descent(nn_Ti_1,nn_O_1,inputs2_,outputs2_,learning_rate=3e-5,batchSize=50,epochs=epochs_val)
#cost_variation_mom,lr_mom = SGD_momentum(nn_Ti_2,nn_O_2,inputs2_,outputs2_,learning_rate=3e-5,epochs=epochs_val,beta=0.75)                      
cost_variation_sgd,lr_sgd = stochastic_gradient_descent(nn_Ti_3,nn_O_3,inputs2_,outputs2_,learning_rate=1e-5,epochs=epochs_val)




x,y,z = predict_energy(inputs2_,outputs2_)
print(x)

x_ = (np.concatenate(x).reshape(-1))
print(x_.tolist())
print(outputs2_)
print(x_.tolist()-outputs2_)
val = np.mean(np.square(x_.tolist()-outputs2_))
print(val)
#exit(0)
toc = time.time()
#exit(0)
x_axis = np.linspace(0,epochs_val,epochs_val)

if __name__ == '__main__':
    fig = plt.figure(figsize = (6,4),dpi =150)
    plt.plot(x_axis,cost_variation_sgd,'.-b',label='SGD')
    #plt.plot(x_axis,cost_variation_mbgd,'.-r',label='minibatch GD')
    #plt.plot(x_axis,cost_variation_mom,'.:y',label='SGD momentum')
    plt.xlabel('epochs')
    plt.ylabel('cost')
    plt.legend()
    plt.title('Cost variation for a small dataset')

    plt.show()
    fig.tight_layout()
    fig.savefig('SGD v mini_batch.png')

#print(data[1]['output'])



fig = plt.figure(figsize = (7,4),dpi =150)
if __name__== '__main__':
    plt.plot(y,'o:y',label='reference')
    plt.plot(x_,'.--r',label='predicted')
    plt.xlabel('m')
    plt.ylabel('Energy')
    plt.legend()
    plt.title('Predicted v Reference')
    #plt.ylim(-19945,-19975)
    plt.show()
    fig.tight_layout()
    fig.savefig('predict_v_reference_test_set.png')

fig = plt.figure(figsize = (7,4),dpi =150)
if __name__== '__main__':
    plt.plot(z,'.:b',label='reference')
    plt.xlabel('m')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Test set error')
    #plt.ylim(-19945,-19975)
    plt.show()
    fig.tight_layout()
    fig.savefig('testset error.png')


if __name__== '__main__':
    fig, ax = plt.subplots()
    plt.scatter(y,x_)
    ax.set_xlabel('ANN energy')
    ax.set_ylabel('DFT energy')
    ax.legend()
    ax.set_title('ANN v DFT')
    ax.set_aspect('equal','datalim')
    #plt.ylim(-19945,-19975)
    plt.show()
    fig.tight_layout()
    fig.savefig('ANN_v_DFT.png')



print('Time taken =',str((toc-tic)) + 'sec')