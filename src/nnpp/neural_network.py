"""
====================================================================================================
Neural network module
----------------------------------------------------------------------------------------------------
Builds the neural network for prediction of potential from scratch using numpy.This module does the 
following tasks,
    -Reads required attributes of data (output of symmetry module)
    -Test train split
    -Min-max normalization
    -Shuffles the data frequently
    -Neural network with different activation functions 
        -Activation functions - sigmoid,tanh,ReLU
    -Simultaneous training of the NN assembly as we train for the total energy value but update parameters of individual NNs
        -Optimizers used - SGD,SGD with momentum,RmsProp,Adam,Minibatch GD
    -Saves the weights of the trained neural networks for further predictions.
====================================================================================================
"""
# ==================================================================================================
# imports
# --------------------------------------------------------------------------------------------------
import os
import numpy as np 
import time
import matplotlib.pyplot as plt 
import random
# ==================================================================================================

#Data reading---------------------------------------------------------------------------------------
def data_read(path):
    """
    Reads the data from the files in a folder described by the path.
    Arguments :
    path -- path of the folder found using 'os' library
    Returns:
    file_list_ -- list of names of all files in the folder
    energy_list_ -- list of energy values of each file in the folder
    number_of_atomss -- list of number of atoms in each file
    """
    file_list_ = (os.listdir(path))
    number_of_atomss = []
    energy_list_ = []

    for file in file_list_:

        with open(path+'/'+'%s'%(file)) as f:
            for i,line in enumerate(f):

                if i==0:  #energy line strips at =
                    energy_list_.append(float((line.strip().split('='))[1][:-3]))            

                if i==8:  #to get no: of atoms...goes to that line strips and splits first element
                    no_of_atoms = line.strip().split(' ')
                    n=int(no_of_atoms[0])
                    number_of_atomss.append(n)
    return file_list_,energy_list_,number_of_atomss

#Test-train split-----------------------------------------------------------------------------------

def test_train_split(filelist,energylist,split):
    '''Creates an empty array for test split and pops each element from total
dataset and append it to the test set simultaneously,thereby creating test set as well as train set
    Arguments:
    filelist -- list of names of all files in the folder
    energylist -- list of energy values of each file in the folder
    split -- percentage by which data should be splitted

    Returns:
    train_set -- training dataset used for training the NN
    test_set -- testing data set to validate the tested NN
    train_energy -- reference output for training dataset
    test_energy -- reference output for testing dataset
    '''
    n_total_set = len(filelist)
    n_train_set = split/100 * n_total_set
    train_set = []                                       #train set is empty
    test_set = filelist[:]                               #takes a copy of the entire dataset(testset is full)
    train_energy = [] 
    test_energy = energylist[:]
    np.random.seed(6)
    while len(train_set) < n_train_set :
        #np.random.seed(61)
        indox = np.random.randint(0,len(test_set))       #chooses a randdom index
        train_set.append(test_set.pop(indox))            #removes from testset,appends to train set
        train_energy.append(test_energy.pop(indox))      #does the same for output
    return train_set,test_set,(train_energy),(test_energy)

#min max normalization---------------------------------------------------------
def min_max_param(train_input):
    """
    Finds the minimum and maximum values required for normalization.
    Arguments :
    train_input -- training data
    Returns:
    G_min -- min vector of the entire training set
    G_max -- max vector of the entire training set
    """
    g_max_array = []
    for i in range(len(train_input)):
        g_max_array.append(np.max(train_input[i],axis=0))         #finding max vector of each dataset in training set
    g_max_array_np = np.array(g_max_array)
    G_max = np.max(g_max_array_np,axis=0)                         #finding max vector of the entire training set
                                       

    g_min_array = []
    for i in range(len(train_input)):
        g_min_array.append(np.min(train_input[i],axis=0))         #finding min vector of each dataset in training set
    g_min_array_np = np.array(g_min_array)
    G_min = np.min(g_min_array_np,axis=0)                         #finding min vector of the entire training set

    return G_min,G_max

def min_max_norm(G,Gmin,Gmax):
    """
    Performs the normalization process.
    Arguments :
    G --  input data (test or train)
    Gmin -- min vector of the entire training set
    Gmax -- max vector of the entire training set
    Return:
    None
    """
    for i in range(len(G)):                                       #nested for loop to transform each element
        for j in range(len(G[i])):
            G[i][j] = (2*(G[i][j]-Gmin)/(Gmax-Gmin))-1            #scales the G vector between [-1,1]
        

def data_shuffle(A,B):
    """
    Shuffles 2 arrays(NN input & NN output) with same seed
    Arguments :
    A -- array 1 (input data in this case)
    B -- array 2 (output data in this case)
    Returns :
    None
    """
    r = random.random()
    random.shuffle(A,lambda:r)   #shuffles A,B with same random parameter (r value)
    random.shuffle(B,lambda:r)

#---------------------------------------------------------------------------------------------------
#Activation functions
#---------------------------------------------------------------------------------------------------

def sigmoid(z):
    """Sigmoid activation function"""
    s = 1.0/(1+np.exp(-z))
    return s 

def ReLU(z):
    """ReLU activation function"""
    return np.maximum(0,z)

def tanh(z):
    """tanh activation function"""
    val = np.tanh(z)
    return val

def linear(z):
    """Linear function for output node"""
    return z

#---------------------------------------------------------------------------------------------------
#Activation function derivatives
#---------------------------------------------------------------------------------------------------
def d_sigmoid(s):
    #derivative of sigmoid
    return s * (1-s)

def d_ReLU(s):
    #derivative of ReLU
    return 1.0 *(s>0)

def d_tanh(s):
    #derivative of tanh
    return 1 - (s**2)

def d_linear(s):
    #derivative of linear function
    return 1

#Putting all activation functions and their derivatives into a dictionary
ActivationFunction = {
    'sigmoid' : (sigmoid,d_sigmoid),
    'ReLU'    : (ReLU,d_ReLU),
    'tanh'    : (tanh,d_tanh),
    'linear'  : (linear,d_linear),
    }

def RMSE(predictions,references,n,errortype):
    """
    Gives the root mean squared error of predictions and references.
    Arguments:
    predictions -- predicted value by the neural network
    references -- reference value from the labelled data
    n -- number of atoms in the structure
    errortype -- type of error required
    Returns:
    -returns the RMSE
    """
    m = len(references)     
    if errortype == 'eV_per_atom':
        return np.sqrt(np.sum(np.square(np.divide((predictions-references),n)))/m)
    elif errortype == 'eV_per_structure':
        return np.sqrt(np.sum(np.square(predictions-references))/m)


def MAE(predictions,references):
    """
    Gives the mean absolute error of predictions and references.
    Arguments:
    predictions -- predicted value by the neural network
    references -- reference value from the labelled data
    Returns:
    -returns the MAE
    """    
    m = len(references)
    return np.sum(np.abs(predictions-references))/m

def MSE(predictions,references,n,errortype):
    """
    Gives the mean squared error of predictions and references.
    Arguments:
    predictions -- predicted value by the neural network
    references -- reference value from the labelled data
    n -- number of atoms in the structure
    errortype -- type of error required
    Returns:
    -returns the MSE
    """    
    m = len(references)
    if errortype == 'eV_per_atom':
        return 0.5*np.sum(np.square(np.divide((predictions-references),n)))/m
    elif errortype == 'eV_per_structure':
        return 0.5*np.sum(np.square(predictions-references))/m

def MSE_basic(predictions,references):
    """
    basic MSE function for calculating numerical gradient for gradient checking
    Arguments:
    predictions -- predicted value by the neural network
    references -- reference value from the labelled data
    """
    m = len(references)
    return 0.5*np.sum(np.square(predictions-references))/m

def d_MSE(predictions,references):
    """Gives the derivative of MSE."""
    return np.asarray(predictions-references)

#---------------------------------------------------------------------------------------------------

class NeuralNetwork:
    """
    The neural network class to be used as individual atomic NN to predict the atomic energy contributions
    of different atoms
    """
    def __init__(self,nodelist,activations):
        '''Initialization of neural network parameters(weights,bias,activation functions,architecture etc.)
        Arguments:
        nodelist -- list of number of nodes in each layer (in the order)
        activations -- list of activation functions in each layer (in the order)
        '''
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
        """Function to represent the neural network if needed."""
        return f"This is a {self.input_nodes}-{self.hidden_layer1_nodes}-{self.hidden_layer2_nodes}-{self.output_nodes} neural network"

        
    def forward_prop(self,x):
        """
        The input data fed is propogated forward to the output node.Each layers accepts the input data,
        processes it by multiplying the weight parameters and adding the bias parameters followed by
        passing it through the activation function and passes to the next layer.
        Arguments :
        x -- input to the input layer of NN
        Returns:
        output -- output from the neural network
        """
        self.layer1 = self.layer1_activation(np.dot(x,self.weights1)+self.bias1)
        self.layer2 = self.layer2_activation(np.dot(self.layer1,self.weights2)+self.bias2)
        output      = self.output_activation(np.dot(self.layer2,self.weights3)+self.bias3)
        return output

    
    def backward_prop(self,x,e_nn,e_ref):
        """
        This function backpropogates the error in order to adjust the weight parameters in order to 
        get a better prediction.
        Arguments :
        x -- input data
        e_nn -- energy value predicted by NN  
        e_ref -- reference energy value from labelled data
        Returns:
        dJdw1,dJdw2,dJdw3,dJdb1,dJdb2,dJdb3 -- weight update terms(derivatives)
        """
        self.d_output_layer = d_MSE(e_nn,e_ref)
        self.delta_output_layer = self.d_output_layer * self.output_der_activation(e_nn)
        #layer 2
        self.d_layer2 = self.delta_output_layer.dot(self.weights3.T)
        self.delta_layer2 = self.d_layer2 * self.layer2_der_activation(self.layer2)
        #layer 1
        self.d_layer1 = self.delta_layer2.dot(self.weights2.T)
        self.delta_layer1 = self.d_layer1 * self.layer1_der_activation(self.layer1)

        #weight update term(derivatives)
        dJdw1 =  (x.T).dot(self.delta_layer1)
        dJdw2 =  (self.layer1.T).dot(self.delta_layer2)
        dJdw3 =  (self.layer2.T).dot(self.delta_output_layer)

        dJdb1 = self.delta_layer1
        dJdb2 = self.delta_layer2
        dJdb3 = self.delta_output_layer

        return dJdw1,dJdw2,dJdw3,dJdb1,dJdb2,dJdb3

    def dEi_dGi(self,output):
        """
        This is the first term in the chain rule to calculate the force,which is obtained from the 
        architecture of the NN
        Arguments:
        output -- output of individual atomic NN
        Return :
        self.d_input_layer -- the derivative of output of neural network with respect to the input
        """
        #print(output)
        self.delta_output_layer = np.asarray(self.output_der_activation(output))
        #print(self.delta_output_layer)

        self.d_layer2 = self.delta_output_layer.dot(self.weights3.T)
        self.delta_layer2 = self.d_layer2 * self.layer2_der_activation(self.layer2)

        self.d_layer1 = self.delta_layer2.dot(self.weights2.T)
        self.delta_layer1 = self.d_layer1 * self.layer1_der_activation(self.layer1)

        self.d_input_layer = self.delta_layer1.dot(self.weights1.T)

        return self.d_input_layer


    def NN_optimize(self,dw1,dw2,dw3,db1,db2,db3,learning_rate):
        """
        Functions which update the weight and bias parameters with the output from backpropogation 
        function
        Arguments :
        dw1,dw2,dw3,db1,db2,db3 -- weight update terms(derivatives)
        learning_rate -- tuning parameter that determines the stepsize of optimization
        Return: none
        """
        self.weights1 -= learning_rate * dw1
        self.weights2 -= learning_rate * dw2
        self.weights3 -= learning_rate * dw3
        self.bias1 -= learning_rate * db1
        self.bias2 -= learning_rate * db2
        self.bias3 -= learning_rate * db3

    #gradient checking------------------------------------------------------------------------------
    def collect_parameters(self):
        """ This function collects all the weights and converts weight matrix into a flattened n x 1 array
        which is needed to perform gradient checking"""        
        parameters = np.concatenate((self.weights1.reshape(-1),self.weights2.reshape(-1),self.weights3.reshape(-1)))  #makes 1d array and joins all
        return parameters
    
    def set_parameters(self,parameters):
        """
        This function does the opposite of the above functions.It transforms the flattened array back
        to the 2d array as it intially was.
        Arguments:
        parameters -- output of the above function
        Returns : 0
        """
        #w1 matrix
        w1_first = 0
        w1_last = self.input_nodes * self.hidden_layer1_nodes
        self.weights1 = parameters[w1_first:w1_last].reshape(self.input_nodes,self.hidden_layer1_nodes)
        #w2 matrix
        w2_first = w1_last
        w2_last = w2_first + (self.hidden_layer1_nodes*self.hidden_layer2_nodes)
        self.weights2 = parameters[w2_first:w2_last].reshape(self.hidden_layer1_nodes,self.hidden_layer2_nodes)
        #w3 matrix
        w3_first = w2_last
        w3_last = w3_first +(self.hidden_layer2_nodes*self.output_nodes)
        self.weights3 = parameters[w3_first:w3_last].reshape(self.hidden_layer2_nodes,self.output_nodes)
        return 0

    def analytical_gradients(self,x,e_nn,e_ref):
        '''returns the gradients found by the backprop algorithm
        Arguments :
        x -- input data
        e_nn -- energy value predicted by NN  
        e_ref --reference energy value from labelled data
        '''
        djdw1,djdw2,djdw3,_,_,_ = self.backward_prop(x,e_nn,e_ref)
        return np.concatenate((djdw1.reshape(-1),djdw2.reshape(-1),djdw3.reshape(-1))) #flattens the gradient values into a 1D array


epsilon = 1e-5    
def numerical_gradients(nn,x,e_nn,e_ref):
    """
    The numerical derivative is calculated using the definition of derivatives(central difference method).
    Arguments :
    nn -- object of the Neural network class
    x -- input data
    e_nn -- energy value predicted by NN  
    e_ref --reference energy value from labelled data 
    Return:
    numerical_gradient_values -- the numerical derivative is obtained   

    """
    parameter_values          = nn.collect_parameters()          #collects the weights of the NN
    numerical_gradient_values = np.zeros(parameter_values.shape) #to store the numerical gradients of corresponding perturbing weight
    small_change_vector       = np.zeros(parameter_values.shape) #vector for disturbing the weights(one at a time)   

    for i in range(len(parameter_values)):
        #we need to change the i th element of small_change_vector to epsilon
        small_change_vector[i] = epsilon
        #now we add this change to the parameter values and pass it to set_parameters function and calculate loss
        new_parameter_values = parameter_values+small_change_vector
        nn.set_parameters(new_parameter_values)
        e_nn = nn.forward_prop(x)
        loss_plus = MSE_basic(e_nn,e_ref)
        #now we subtract the change and find loss
        new_parameter_values = parameter_values-small_change_vector
        nn.set_parameters(new_parameter_values)
        e_nn = nn.forward_prop(x)
        loss_minus = MSE_basic(e_nn,e_ref)
        #derivative  calculated using central difference method
        numerical_gradient_values[i] = (loss_plus-loss_minus)/(2*epsilon)
        small_change_vector[i] = 0
    
    nn.set_parameters(parameter_values) #set the shape back(from flattened array to 2d array)

    return numerical_gradient_values
#---------------------------------------------------------------------------------------------------
#Combining multiple atomic NNs (2 distinct atomic NNs)to train for the total structural energy
#---------------------------------------------------------------------------------------------------
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
    """
    Assigns the weight matrices of corresponding sizes with zero values

    Arguments :
    nn1 -- object of class Neuralnetwork

    Return :
    returns 6 weight matrices assigned with zero values
    (3 matrices for atom1(as the NN has 2 hidden layers,there are 3 weight matrices),similarly for atom2)
    """
    w1_t_,w1_o_ = np.zeros(nn1.weights1.shape),np.zeros(nn1.weights1.shape)
    w2_t_,w2_o_ = np.zeros(nn1.weights2.shape),np.zeros(nn1.weights2.shape)
    w3_t_,w3_o_ = np.zeros(nn1.weights3.shape),np.zeros(nn1.weights3.shape)
    return w1_t_,w1_o_,w2_t_,w2_o_,w3_t_,w3_o_ 

def bias_array_init(nn1):
    """
    Assigns the bias matrices of corresponding sizes with zero values

    Arguments :
    nn1 -- object of class Neuralnetwork
    
    Return :
    returns 6 bias matrices assigned with zero values
    (3 matrices for atom1(as the NN has 2 hidden layers,there are 3 bias matrices),similarly for atom2)
    """
    b1_t_,b1_o_ = np.zeros(nn1.bias1.shape),np.zeros(nn1.bias1.shape)
    b2_t_,b2_o_ = np.zeros(nn1.bias2.shape),np.zeros(nn1.bias2.shape)
    b3_t_,b3_o_ = np.zeros(nn1.bias3.shape),np.zeros(nn1.bias3.shape)
    return b1_t_,b1_o_,b2_t_,b2_o_,b3_t_,b3_o_ 

def structure_nn_gradient(a,nn1,nn2):
    """
    Finds the gradient of output neuron wrt the input vector (.ie dEi/dG)

    Arguments :
    a -- 3d array of symmetry functions[G] with shape [n x 1 x 70]
    nn1 -- object of class Neuralnetwork
    nn2 -- another object of class Neuralnetwork

    
    Return :
    grad_array -- gradient array(dEi/dG)  with shape [n x 1 x 70] (ie (1x70) for each atom)
    """    
    grad_array = []
    index = nn_switcher(a)
    
    for i in range(len(a)):
        if i<= (index-1):
            grad_array.append(nn1.dEi_dGi(a[i]))
        else:
            grad_array.append(nn2.dEi_dGi(a[i]))
    return grad_array



def param_init(nn1):
    """
    Assigns parameters to be used in optimizer methods

    Arguments :
    nn1 -- object of class Neuralnetwork
    
    Return :
    returns parameters assigned with zero values    
    """
    v_w1_t,v_w1_o,v_b1_t,v_b1_o = np.zeros(nn1.weights1.shape),np.zeros(nn1.weights1.shape),np.zeros(nn1.bias1.shape),np.zeros(nn1.bias1.shape)
    v_w2_t,v_w2_o,v_b2_t,v_b2_o = np.zeros(nn1.weights2.shape),np.zeros(nn1.weights2.shape),np.zeros(nn1.bias2.shape),np.zeros(nn1.bias2.shape)
    v_w3_t,v_w3_o,v_b3_t,v_b3_o = np.zeros(nn1.weights3.shape),np.zeros(nn1.weights3.shape),np.zeros(nn1.bias3.shape),np.zeros(nn1.bias3.shape)
    return v_w1_t,v_w2_t,v_w3_t,v_b1_t,v_b2_t,v_b3_t,v_w1_o,v_w2_o,v_w3_o,v_b1_o,v_b2_o,v_b3_o


def structure_backward_prop(a,nn1,nn2,output,e_ref,learning_rate):
    """
    Gives the gradients of loss function wrt the weights and biases

    Arguments : 
    a -- 3d array of symmetry functions[G] with shape [n x 1 x 70]
    nn1 -- object of class Neuralnetwork
    nn2 -- another object of class Neuralnetwork
    output -- returned by structure_forward_prop function
    e_ref -- array of reference output energies
    learning rate -- tuning parameter that determines the stepsize of optimization

    Returns : 
    w_b_parameters -- array of gradients of loss function wrt weights(summed over for a particular atomic NN)
    ie if there are 6 atoms out of which 2 are Ti and 4 are O,first 2 gradient params of Ti are found 
    by backpropogating through Ti_NN and then summed over.Now the remaining 4 gradient params of O are found by backpropogating
    through O_NN and summed over.Later all the 6 arrays are stored in an another array 

    """
    w1_t,w1_o,w2_t,w2_o,w3_t,w3_o  = weight_array_init(nn1)
    b1_t,b1_o,b2_t,b2_o,b3_t,b3_o  =  bias_array_init(nn1)
    index = nn_switcher(a)
    for i in range(len(a)):
        if i<= (index-1):
            #Ti--NN 
            grad_params_nn1 = nn1.backward_prop(a[i],output,e_ref) #gets all the weight update terms as an array and takes the sum over atoms (of same type)
            w1_t += grad_params_nn1[0]    
            w2_t += grad_params_nn1[1]          
            w3_t += grad_params_nn1[2]
            b1_t += grad_params_nn1[3]    
            b2_t += grad_params_nn1[4]          
            b3_t += grad_params_nn1[5]
        else:
            #O--NN 
            grad_params_nn2 = nn2.backward_prop(a[i],output,e_ref)
            w1_o += grad_params_nn2[0]
            w2_o += grad_params_nn2[1]            
            w3_o += grad_params_nn2[2]
            b1_o += grad_params_nn2[3]
            b2_o += grad_params_nn2[4]            
            b3_o += grad_params_nn2[5]  
    val = len(a)-index
    w_b_parameters = [w1_t,w2_t,w3_t,b1_t,b2_t,b3_t,w1_o,w2_o,w3_o,b1_o,b2_o,b3_o]  #zips all the update parameters to an array
    return w_b_parameters
    

def train(nn1,nn2,a,e_ref,learning_rate):#try only
    """
    Changes the weigths and biases so as to optimize the loss function

    Arguments : 
    nn1 -- object of class Neuralnetwork
    nn2 -- another object of class Neuralnetwork
    a -- 3d array of symmetry functions[G] with shape [n x 1 x 70]
    e_ref -- array of reference output energies
    learning rate -- tuning parameter that determines the stepsize of optimization
    """
    output = sum(structure_forward_prop(a,nn1,nn2))    
    w1_T,w2_T,w3_T,b1_T,b2_T,b3_T,w1_O,w2_O,w3_O,b1_O,b2_O,b3_O = \
    structure_backward_prop(a,nn1,nn2,output,e_ref,learning_rate)        #getting the update parameters
    nn1.NN_optimize(w1_T,w2_T,w3_T,b1_T,b2_T,b3_T,learning_rate)         #updates the Ti--NN
    nn2.NN_optimize(w1_O,w2_O,w3_O,b1_O,b2_O,b3_O,learning_rate)         #updates the O--NN

#---------------------------------------------------------------------------------------------------
#Optimizers
#---------------------------------------------------------------------------------------------------

def stochastic_gradient_descent(nn1,nn2,g_list,e_ref_list,learning_rate,epochs):
    """
    Weight update takes place after each dataset passes.
    Arguments:
    nn1 -- NN for atomtype1 (Ti)
    nn2 -- NN for atomtype2 (O)
    g_list -- input data to the NN assembly
    e_ref_list -- reference energy list
    learning_rate -- value with which weight update term is scaled
    epochs -- Number of times entire data should pass through the NN
    Returns :
    cost_extract -- cost variation of training process
    learning_rate -- learning rate for plotting/visualization
    """
    print('############ SGD ################')
    m = len(e_ref_list)
    cost_extract = np.zeros(epochs)
    for i in range(1,epochs+1):          #loop for epochs
        cost = 0
        e_nn_list = []
        num_atoms = []
        for j in range(m):               #loop for datasets(weight update takes place as each dataset passes)
            e_nn_list.append(sum(structure_forward_prop(g_list[j],nn1,nn2)))
            num_atoms.append(len(structure_forward_prop(g_list[j],nn1,nn2)))
            train(nn1,nn2,g_list[j],e_ref_list[j],learning_rate)
        e_nn = np.asarray(np.concatenate(e_nn_list).reshape(-1).tolist()) #converting to 1d array and then to list
        e_ref = np.asarray(e_ref_list)
        n = np.asarray(num_atoms)
        cost = RMSE(e_nn,e_ref,n,errortype='eV_per_atom')          #calculating cost 
        print('{0: <6}'.format('epoch ='),i,'----','{0: <4}'.format('lr ='),learning_rate,'----','{0: <6}'.format('cost ='),cost)     
        cost_extract[i-1] = cost                                        #extracting cost
        data_shuffle(g_list,e_ref_list)                                 #shuffling data before next epoch
    return cost_extract,learning_rate

def SGD_momentum(nn1,nn2,g_list,e_ref_list,learning_rate,epochs,beta):
    """
    Weight update takes place after each dataset passes but now with momentum term which considers the history as well.
    Arguments:
    nn1 -- NN for atomtype1 (Ti)
    nn2 -- NN for atomtype2 (O)
    g_list -- input data to the NN assembly
    e_ref_list -- reference energy list
    learning_rate -- value with which weight update term is scaled
    epochs -- Number of times entire data should pass through the NN
    beta -- momentum term
    Returns :
    cost_extract -- cost variation of training process
    learning_rate -- learning rate for plotting/visualization
    """
    print('############ SGD with momentum ################')
    m = len(e_ref_list)
    cost_extract = np.zeros(epochs)
    for i in range(1,epochs+1):             #loop for epochs
        cost = 0
        e_nn_list = []
        num_atoms = []
        v_params = np.asarray(param_init(nn1))
        for j in range(m):                  #loop for datasets
            e_nn_list.append(sum(structure_forward_prop(g_list[j],nn1,nn2)))
            num_atoms.append(len(structure_forward_prop(g_list[j],nn1,nn2))) 
            output = sum(structure_forward_prop(g_list[j],nn1,nn2))    
            w_b_params = np.asarray(structure_backward_prop(g_list[j],nn1,nn2,output,e_ref_list[j],learning_rate))   #weight update according to momentum           
            v_params = beta*v_params +(1-beta)*w_b_params
            nn1.NN_optimize(*v_params[:6],learning_rate)
            nn2.NN_optimize(*v_params[6:],learning_rate)

        n = np.asarray(num_atoms)            
        e_nn = np.asarray(np.concatenate(e_nn_list).reshape(-1).tolist()) #converting to 1d array and then to list
        e_ref = np.asarray(e_ref_list)
        cost = MSE(e_nn,e_ref,n,errortype='eV_per_atom')                   #calculating cost 
        print('{0: <6}'.format('epoch ='),i,'----','{0: <4}'.format('lr ='),learning_rate,'----','{0: <6}'.format('cost ='),cost)     
        cost_extract[i-1] = cost                                           #extracting cost
        data_shuffle(g_list,e_ref_list)                                    #shuffling data before next epoch
    return cost_extract,learning_rate

def RMSprop(nn1,nn2,g_list,e_ref_list,learning_rate,epochs,beta):
    """
    Weight update using RMS Prop optimizer
    Arguments:
    nn1 -- NN for atomtype1 (Ti)
    nn2 -- NN for atomtype2 (O)
    g_list -- input data to the NN assembly
    e_ref_list -- reference energy list
    learning_rate -- value with which weight update term is scaled
    epochs -- Number of times entire data should pass through the NN
    beta -- moving average parameter
    Returns :
    cost_extract -- cost variation of training process
    learning_rate -- learning rate for plotting/visualization
    """
    print('############ RMSprop ################')
    m = len(e_ref_list)
    cost_extract = np.zeros(epochs)
    for i in range(1,epochs+1):                           #loop for epochs
        cost = 0
        e_nn_list = []
        num_atoms = []
        s_params = np.asarray(param_init(nn1))
        update_param = np.asarray(param_init(nn1))
        for j in range(m):                               #loop for datasets
            e_nn_list.append(sum(structure_forward_prop(g_list[j],nn1,nn2)))
            num_atoms.append(len(structure_forward_prop(g_list[j],nn1,nn2)))
            output = sum(structure_forward_prop(g_list[j],nn1,nn2))    
            #weight update terms for RMSProp
            w_b_params = np.asarray(structure_backward_prop(g_list[j],nn1,nn2,output,e_ref_list[j],learning_rate))             
            s_params = beta*s_params +(1-beta)*(w_b_params)**2
            update_param = np.divide(w_b_params,(s_params+1e-8)**0.5)

            nn1.NN_optimize(*update_param[:6],learning_rate)  #param update
            nn2.NN_optimize(*update_param[6:],learning_rate)

        n = np.asarray(num_atoms)
        e_nn = np.asarray(np.concatenate(e_nn_list).reshape(-1).tolist())
        e_ref = np.asarray(e_ref_list)                 
        cost = RMSE(e_nn,e_ref,n,errortype='eV_per_atom') 
        print('{0: <6}'.format('epoch ='),i,'----','{0: <4}'.format('lr ='),learning_rate,'----','{0: <6}'.format('cost ='),cost)     
        cost_extract[i-1] = cost                             #cost extraction for visualization
        data_shuffle(g_list,e_ref_list)
    return cost_extract,learning_rate

def Adam(nn1,nn2,g_list,e_ref_list,learning_rate,epochs,beta1,beta2):
    """
    Weight update according to Adaptive moment estimation
    Arguments:
    nn1 -- NN for atomtype1 (Ti)
    nn2 -- NN for atomtype2 (O)
    g_list -- input data to the NN assembly
    e_ref_list -- reference energy list
    learning_rate -- value with which weight update term is scaled
    epochs -- Number of times entire data should pass through the NN
    beta1,beta2 -- hyperparameters

    Returns :
    cost_extract -- cost variation of training process
    learning_rate -- learning rate for plotting/visualization
    """    
    print('############ Adam ################')
    m = len(e_ref_list)
    cost_extract = np.zeros(epochs)
    for i in range(1,epochs+1):          #loop for epochs
        cost = 0
        e_nn_list = []
        num_atoms = []        
        v_params,v_params_corrected = np.asarray(param_init(nn1)),np.asarray(param_init(nn1))
        s_params,s_params_corrected = np.asarray(param_init(nn1)),np.asarray(param_init(nn1))
        update_param = np.asarray(param_init(nn1))
        for j in range(m):               #loop for datasets
            e_nn_list.append(sum(structure_forward_prop(g_list[j],nn1,nn2)))
            num_atoms.append(len(structure_forward_prop(g_list[j],nn1,nn2)))            
            output = sum(structure_forward_prop(g_list[j],nn1,nn2)) 
            #terms need for Adam   
            w_b_params = np.asarray(structure_backward_prop(g_list[j],nn1,nn2,output,e_ref_list[j],learning_rate))  
            v_params = beta1*v_params +(1-beta1)*w_b_params
            s_params = beta2*s_params +(1-beta2)*(w_b_params)**2

            v_params_corrected = v_params/(1-(beta1**(j+1)))
            s_params_corrected = s_params/(1-(beta2**(j+1)))

            update_param = np.divide(v_params_corrected,(s_params_corrected+1e-8)**0.5)
            #Update steps
            nn1.NN_optimize(*update_param[:6],learning_rate)
            nn2.NN_optimize(*update_param[6:],learning_rate)
            
        n = np.asarray(num_atoms)
        e_nn = np.asarray(np.concatenate(e_nn_list).reshape(-1).tolist())   #converting to 1d array and then to list
        e_ref = np.asarray(e_ref_list)          
        cost = RMSE(e_nn,e_ref,n,errortype='eV_per_atom') 
        print('{0: <6}'.format('epoch ='),i,'----','{0: <4}'.format('lr ='),learning_rate,'----','{0: <6}'.format('cost ='),cost)     
        cost_extract[i-1] = cost
        data_shuffle(g_list,e_ref_list)
    return cost_extract,learning_rate

def minibatch_gradient_descent(nn1,nn2,g_list,e_ref_list,learning_rate,batchSize,epochs):
    """
    Weight update takes place after each minibatch passes through the NN
    Arguments:
    nn1 -- NN for atomtype1 (Ti)
    nn2 -- NN for atomtype2 (O)
    g_list -- input data to the NN assembly
    e_ref_list -- reference energy list
    learning_rate -- value with which weight update term is scaled
    batchSize -- number of datasets in the batch
    epochs -- Number of times entire data should pass through the NN

    Returns :
    cost_extract -- cost variation of training process
    learning_rate -- learning rate for plotting/visualization
    """       
    print('############ Mini batch GD ################')
    m = len(e_ref_list)
    batchNumber = int(m/batchSize)
    cost_extract = np.zeros(epochs)
    for i in range(1,epochs+1):
        cost = 0
        e_nn_list = []
        num_atoms = [] 
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
                num_atoms.append(len(structure_forward_prop(g_list_j[k],nn1,nn2))) 

            parameters_nn1 = np.asarray([w1_t_mb,w2_t_mb,w3_t_mb,b1_t_mb,b2_t_mb,b3_t_mb])/batchSize #mean dw,mean db
            parameters_nn2 = np.asarray([w1_o_mb,w2_o_mb,w3_o_mb,b1_o_mb,b2_o_mb,b3_o_mb])/batchSize
            nn1.NN_optimize(*parameters_nn1,learning_rate)    
            nn2.NN_optimize(*parameters_nn2,learning_rate) 
            
        n = np.asarray(num_atoms)
        e_nn = np.asarray(np.concatenate(e_nn_list).reshape(-1).tolist())
        e_ref = np.asarray(e_ref_list)          
        cost = RMSE(e_nn,e_ref,n,errortype='eV_per_atom')          
        print('{0: <6}'.format('epoch ='),i,'----','{0: <4}'.format('lr ='),learning_rate,'----','{0: <6}'.format('cost ='),cost)     
        cost_extract[i-1] = cost
        data_shuffle(g_list,e_ref_list)
    return cost_extract,learning_rate

def load_params(paramtype):
    """
    Loads the trained(saved) weights to the Neural network.Thereby helps to continue the training process 
    from where it was stopped.After training the saved weights are used for prediction.
    Arguments:
    paramtype -- type of param(trained or guess)
    Returns:
    params -- weight/bias params of Ti--NN and O--NN
    """
    if paramtype=='trained':
        Ti_weights = np.load('params/trained_ti_weights.npz')
        O_weights = np.load('params/trained_O_weights.npz')

    if paramtype=='guess':
        Ti_weights = np.load('params/dict_ti_11_weights.npz')
        O_weights = np.load('params/dict_O_11_weights.npz')        

    w1_ti = Ti_weights['w1']
    w2_ti = Ti_weights['w2']
    w3_ti = Ti_weights['w3']
    b1_ti = Ti_weights['b1']
    b2_ti = Ti_weights['b2']
    b3_ti = Ti_weights['b3'] 

    w1_ox = O_weights['w1']
    w2_ox = O_weights['w2']
    w3_ox = O_weights['w3']
    b1_ox = O_weights['b1']
    b2_ox = O_weights['b2']
    b3_ox = O_weights['b3'] 
    params = [w1_ti,w2_ti,w3_ti,b1_ti,b2_ti,b3_ti,w1_ox,w2_ox,w3_ox,b1_ox,b2_ox,b3_ox] #zipping into an array
    return params

def initialize_params(nn,w_1,w_2,w_3,b_1,b_2,b_3):
    """
    Assigns the parameters with the values in the Argument.
    Arguments:
    nn -- object of Neural network class
    w_1,w_2,w_3,b_1,b_2,b_3 -- weight/bias params
    """
    nn.weights1 = w_1
    nn.weights2 = w_2
    nn.weights3 = w_3
    nn.bias1    = b_1
    nn.bias2    = b_2    
    nn.bias3    = b_3


def predict_energy(g_list,e_ref_list,nn1,nn2):
    """
    Predicts the structural energy using the given NNs as argument.
    Arguments:
    g_list -- input data to the NN assembly
    e_ref_list -- reference energy list
    nn1 -- NN for atomtype1 (Ti)
    nn2 -- NN for atomtype2 (O)    

    Returns:
    e_predicted_list_1d -- Predicted energy values of all datasets as a list
    e_ref_list -- Reference enegy values of all datasets as a list
    test_error -- (Error in energy)/atom for all structures
    """
    m = len(e_ref_list)
    e_predicted_list = []
    e_predicted_list_2 = []
    num_atoms = []
    for j in range(m):  #loop over datasets
        e_predicted_list.append(sum(structure_forward_prop(g_list[j],nn1,nn2)))  #energy list
        num_atoms.append(len(structure_forward_prop(g_list[j],nn1,nn2)))         # list of number of atoms 

    n = np.asarray(num_atoms)
    e_predicted_list_1d = np.concatenate(e_predicted_list).reshape(-1).tolist()
    test_mse = RMSE(e_predicted_list_1d,e_ref_list,n,errortype='eV_per_atom')
    test_error = np.divide((e_predicted_list_1d-e_ref_list),n)    #error per atom
    return e_predicted_list_1d,e_ref_list,test_error

def predict_energy_2(g_list,nn1,nn2):
    """
    Predicts the structural energy
    Arguments:
    g_list -- input data to the NN assembly
    nn1 -- NN for atomtype1 (Ti)
    nn2 -- NN for atomtype2 (O)     

    Returns:
    e_predicted_list_1d -- Predicted total energy values of all datasets as a list
    E_i_array -- Predicted  individual atomic energy values of all datasets as a list of lists

    """
    e_predicted_list = []
    for g in g_list:
        e_predicted_list.append(sum(structure_forward_prop(g,nn1,nn2)))
        E_i_array = structure_forward_prop(g,nn1,nn2)
    e_predicted_list_1d = np.concatenate(e_predicted_list).reshape(-1).tolist()
    return e_predicted_list_1d,E_i_array


def correlation_coefficient(e_predicted,e_reference):
    """
    Calculates the correlation coefficient as a measure of NN performance
    Arguments:
    e_predicted -- predicted energy
    e_reference -- reference energy
    Returns:
    val -- value of correlation coefficient

    """
    numerator = np.sum(np.square(e_reference-e_predicted))   #squared_error_sum

    e_reference_mean = np.mean(e_reference)
    denominator = np.sum(np.square(e_reference-e_reference_mean))

    val = 1 - numerator/denominator

    return val

def trained_NN_save_param(nn1,nn2):
    """
    Function which saves the trained weights in npz file format for future use(predictions).
    Arguments:
    nn1 -- NN for atomtype1 (Ti)
    nn2 -- NN for atomtype2 (O)    
    """
    trained_Ti_weights = {
                    'w1' : nn1.weights1,
                    'w2' : nn1.weights2,
                    'w3' : nn1.weights3,
                    'b1' : nn1.bias1,
                    'b2' : nn1.bias2,
                    'b3' : nn1.bias3,
                    }

    trained_O_weights = {
                    'w1' : nn2.weights1,
                    'w2' : nn2.weights2,
                    'w3' : nn2.weights3,
                    'b1' : nn2.bias1,
                    'b2' : nn2.bias2,
                    'b3' : nn2.bias3,
    }

    np.savez('params/trained_ti_weights.npz',**trained_Ti_weights)
    np.savez('params/trained_O_weights.npz',**trained_O_weights)



if __name__ == '__main__':
    print("\n----------------------------- NEURAL NETWORK MODULE--------------------------------------\n")
    print('##########  Implementation of NN as well as training process takes place here.  #########\n\n')
    print('Data loading...this might take few minutes...\n\n')
    
    tic = time.time()
    path = './dataset_TiO2'
    file_list,energy_list,n = data_read(path)
    energy_list2 = ([(a,b) for a,b in zip(energy_list,n)])
    ##print(data_read(path)[0][0],data_read(path)[1][0])
    #splitting the data and storing it in 4 arrays 
    #a)train input    b)train output    c)test input    d)test output
    a,b,c,d = test_train_split(file_list,energy_list,split=99)

    #loading the symmetry function vectors from the corresponding files from which energy value is taken
    test_xy = ([(np.loadtxt(os.path.join('./symmetry_txt','%s') %(x[:-3]+'txt')),y)for x,y in zip(b,d)]) 
    train_xy = ([(np.loadtxt(os.path.join('./symmetry_txt','%s') %(x[:-3]+'txt')),y)for x,y in zip(a,c)])

    #train set arrays-----reshaping input data in the form of (nx1x70) array and shuffling input and output with the same seed
    inputs,outputs = zip(*train_xy)
    inputs_ = np.asarray([x.reshape(len(x),1,70) for x in inputs]) #to np arrays

    outputs_ = np.asarray([*outputs])
    #np.random.seed(60)
    data_shuffle(inputs_,outputs_)

    #test set arrays-----reshaping input data in the form of (nx1x70) array and shuffling input and output with the same seed
    inputs2,outputs2 = zip(*test_xy)
    inputs2_= np.asarray([x.reshape(len(x),1,70) for x in inputs2])
    outputs2_= np.asarray([*outputs2])
    data_shuffle(inputs2_,outputs2_)
    
    #calculating minimum and maximum,followed by min-mx normalization
    g_min,g_max = min_max_param(inputs_)
    min_max_norm(inputs2_,g_min,g_max)
    min_max_norm(inputs_,g_min,g_max)
    
    min_max_parameters = {'min' : g_min , 'max' : g_max }
    np.savez('params/min_max_params.npz',**min_max_parameters)   #saving the min-max params to npz file

    trained_params = np.asarray(load_params('trained'))   #collecting the saved weight/bias parameters

    #Declaring NN architecture
    node_list = [70,11,11,1]          
    activations = ['sigmoid','sigmoid','linear']    

    #loading the saved weights to NN
    nn_Ti_4 = NeuralNetwork(node_list,activations)
    initialize_params(nn_Ti_4,*trained_params[0:6])
    nn_O_4 = NeuralNetwork(node_list,activations)
    initialize_params(nn_O_4,*trained_params[6:])

    
    print('The saved weights from the already trained neural network is loaded.\n\n')
    
    epochs_val = 50
    print('----------------------------Training process starts for ',epochs_val,'epochs-----------------------\n')
    #cost_variation_mbgd,lr_mbgd = minibatch_gradient_descent(nn_Ti_4,nn_O_4,inputs_,outputs_,learning_rate=3e-5,batchSize=50,epochs=epochs_val)
    cost_variation_mom,lr_mom = SGD_momentum(nn_Ti_4,nn_O_4,inputs_,outputs_,learning_rate=7e-6,epochs=epochs_val,beta=0.99999) 
    #cost_variation_rms,lr_rms = RMSprop(nn_Ti_3,nn_O_3,inputs_,outputs_,learning_rate=1,epochs=epochs_val,beta=0.9)                       
    #cost_variation_sgd,lr_sgd = stochastic_gradient_descent(nn_Ti_4,nn_O_4,inputs_,outputs_,learning_rate=1e-6,epochs=epochs_val)
    #cost_variation_adam,lr_adam = Adam(nn_Ti_4,nn_O_4,inputs_,outputs_,learning_rate=1e-4,epochs=epochs_val,beta1=0.9,beta2=0.999)

    trained_NN_save_param(nn_Ti_4,nn_O_4)  #saves the weight/bias params of the trained NN
    print('\nWeights and bias parameters of the trained neural networks have been saved.\n')
    toc = time.time()
    print('-----------------------------Training process ended after',epochs_val,'epochs---------------------')
    print('Time taken =',str((toc-tic)) + 'sec')
    print('-----------------------------------------------------------------------------------------\n')



    x_axis = np.linspace(0,epochs_val,epochs_val)   
    fig = plt.figure(figsize = (6,4),dpi =150)    
    plt.plot(x_axis,cost_variation_mom,label='momentum;  lr=7e-6; '+' $\\beta1$=0.99999')
    plt.xlabel('epochs')
    plt.ylabel('cost (eV per atom)')
    plt.legend()
    plt.title('Current cost variation with trained weights')
    fig.tight_layout()
    plt.grid('True')
    plt.show()
    fig.savefig('plots/current state.png')
    exit(0)




