import os
import numpy as np 
import time
import matplotlib.pyplot as plt 

path = '../../dataset_symm'
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
print('Time taken =',str((toc-tic)) + 'sec')

test_xy = ([(np.loadtxt(os.path.join('../../symmetry_functions','%s') %(x[:-3]+'txt')),y)for x,y in zip(b,d)]) #np.loadtxt(os.path.join('./symmetry_functions','%s') %(x[:-3]+'txt'))
train_xy = ([(np.loadtxt(os.path.join('../../symmetry_functions','%s') %(x[:-3]+'txt')),y)for x,y in zip(a,c)])

print(a[0],b[0])
print(test_xy[0][0],train_xy[0][0])




'''
neural network backup
import os
import numpy as np 
import time
import matplotlib.pyplot as plt 

path = '../../data_set_TiO2_small'
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
print('Time taken =',str((toc-tic)) + 'sec')

test_xy = ([(np.loadtxt(os.path.join('../../symmetry_functions','%s') %(x[:-3]+'txt')),y)for x,y in zip(b,d)]) #np.loadtxt(os.path.join('./symmetry_functions','%s') %(x[:-3]+'txt'))
train_xy = ([(np.loadtxt(os.path.join('../../symmetry_functions','%s') %(x[:-3]+'txt')),y)for x,y in zip(a,c)])

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

x1 = np.array([1,2,3,4])
x2 = np.array([6,8,3,1])
x = [[1,2,3,4],[6,8,3,1],[2,3,4,5],[5,4,3,2],[6,7,5,4],[0,3,4,3]]
y = np.array([92])
print(x1,x2)

#---------------------------------------------------------------------------
#Activation functions
#---------------------------------------------------------------------------

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
class NeuralNetwork:
    def __init__(self,nodelist):
        '''Initialisation of neural network parameters'''
        self.input_nodes = nodelist[0]
        self.hidden_layer1_nodes = nodelist[1]
        self.output_nodes = nodelist[-1]

        self.weights1 = np.random.randn(self.input_nodes,self.hidden_layer1_nodes)
        #print(self.weights1)
        self.bias1 = 0
        self.weights2 = np.random.randn(self.hidden_layer1_nodes,self.output_nodes)
        #print(self.weights2)
        self.bias2 = 0

    def __repr__():
        pass
        
    def forward_prop(self,x):
        self.layer1 = linear(np.dot(x,self.weights1)+self.bias1)
        output      = linear(np.dot(self.layer1,self.weights2)+self.bias2)
        return output

node_list = [4,3,1]
nn_Ti = NeuralNetwork(node_list)
nn_O  = NeuralNetwork(node_list)  

# nn_Ti.weights1 = np.ones(nn_Ti.weights1.shape)
# nn_O.weights1 = np.ones(nn_O.weights1.shape)
# nn_Ti.weights2 = np.ones(nn_Ti.weights2.shape)
# nn_O.weights2 = np.ones(nn_O.weights2.shape)
print(nn_Ti.weights1,'\n',nn_O.weights1)
print(nn_Ti.weights2,'\n',nn_O.weights2)

no_of_atoms = len(test_xy[0][0])
print(no_of_atoms)
'''
o = []
for i in range(6):
    if i<2:
        o.append(nn_Ti.forward_prop(x[i]))
    if i>=2:
        o.append(nn_O.forward_prop(x[i]))

print(sum(o))
'''
o1 = nn_Ti.forward_prop(x1)
o2 = nn_O.forward_prop(x2)

print(o1,o2)
'''