"""
====================================================================================================
'Equation of states' module
----------------------------------------------------------------------------------------------------
Module to compute the equation of states of rutile TiO2 and thereby try and validate the energy 
prediction of NN.The atomic positions  and cell volume were extracted from the trajectory file from
https://colab.research.google.com/drive/1PPki67R8tiImOkpTVZ-AHO99WeOg8EGE?usp=sharing#scrollTo=Y-NqFMVWqlYO .
The trained NN was used to predict the energy of the set of configurations in order to fit the 
equation ot states using the ASE package.The results of the fit are plotted as well as stored to a 
text file.
====================================================================================================
"""
# ==================================================================================================
# imports
# --------------------------------------------------------------------------------------------------
import numpy as np 
import ase
from ase.eos import EquationOfState
from ase.units import kJ,kB,fs
from neural_network import*
from symmetry import *
import matplotlib.pyplot as plt 
# --------------------------------------------------------------------------------------------------


#Calling the neural network and assigning the weights of trained NN---------------------------------
trained_params = np.asarray(load_params('trained'))
node_list = [70,11,11,1]          #contains the layer sizes
activations = ['sigmoid','sigmoid','linear']
nn_Ti = NeuralNetwork(node_list,activations)
initialize_params(nn_Ti,*trained_params[:6])  #loading the weights
nn_O = NeuralNetwork(node_list,activations)
initialize_params(nn_O,*trained_params[6:])

def descriptors(atom_data):
    """
    Provides the symmetry vector and its gradient for each atoms in a structure.
    Arguments:
    atom_data -- list of lists containing atomtype and the atomic positions(x,y,z)
    Returns:
    G -- The symmetry vector of the local environment of each atoms.The shape is (n x 70) 
    where n is the number of atoms
    dG_dr -- The derivative of symmetry vector of the local environment of each atoms.The shape 
    is (n x 70 x 30) where n is the number of atoms
    """
    G,dG_dr = symmetry_function(atom_data)
    return G,dG_dr

  
config1 = [['Ti',0.0, 0.0, 0.0],['Ti',2.254, 2.254, 1.4455],['O',1.3524, 1.3524, 0.0],['O',3.1556, 3.1556, 0.0],['O',0.9016000000000001, 3.6064000000000003, 1.4455],['O',3.6064000000000003, 0.9016000000000001, 1.4455]]

config2 = [['Ti', 0.0, 0.0, 0.0],['Ti', 2.2737142857142856, 2.2737142857142856, 1.4581428571428572],['O', 1.3642285714285713, 1.3642285714285713, 0.0],['O', 3.1832, 3.1832, 0.0],['O',0.9094857142857142, 3.637942857142857, 1.4581428571428572],['O', 3.637942857142857, 0.9094857142857142, 1.4581428571428572]]

config3 = [['Ti',0.0, 0.0, 0.0],['Ti', 2.293428571428571, 2.293428571428571, 1.4707857142857146],['O', 1.3760571428571426, 1.3760571428571426, 0.0],['O', 3.2108, 3.2108, 0.0],['O', 0.9173714285714285, 3.669485714285714, 1.4707857142857146],['O', 3.669485714285714, 0.9173714285714285, 1.4707857142857146]]

config4 = [['Ti', 0.0, 0.0, 0.0],['Ti', 2.313142857142857, 2.313142857142857, 1.4834285714285718],['O', 1.3878857142857144, 1.3878857142857144, 0.0],['O', 3.2384000000000004, 3.2384000000000004, 0.0],['O', 0.925257142857143, 3.701028571428572, 1.4834285714285718],['O', 3.701028571428572, 0.925257142857143, 1.4834285714285718]]

config5 = [['Ti', 0.0, 0.0, 0.0],['Ti', 2.3328571428571427, 2.3328571428571427, 1.496071428571429],['O', 1.399714285714286, 1.399714285714286, 0.0],['O', 3.2660000000000005, 3.2660000000000005, 0.0],['O', 0.9331428571428573, 3.732571428571429, 1.496071428571429],['O', 3.732571428571429, 0.9331428571428573, 1.496071428571429]]

config6 = [['Ti', 0.0, 0.0, 0.0],['Ti', 2.3525714285714283, 2.3525714285714283, 1.5087142857142863],['O', 1.4115428571428574, 1.4115428571428574, 0.0],['O', 3.2936000000000005, 3.2936000000000005, 0.0],['O', 0.9410285714285715, 3.764114285714286, 1.5087142857142863],['O', 3.764114285714286, 0.9410285714285715, 1.5087142857142863]]

config7 = [['Ti', 0.0, 0.0, 0.0],['Ti', 2.372285714285714, 2.372285714285714, 1.5213571428571433],['O', 1.4233714285714287, 1.4233714285714287, 0.0],['O', 3.3212, 3.3212, 0.0],['O', 0.9489142857142857, 3.795657142857143, 1.5213571428571433],['O', 3.795657142857143, 0.9489142857142857, 1.5213571428571433]]

config8 = [['Ti', 0.0, 0.0, 0.0],['Ti', 2.392, 2.392, 1.5340000000000007], ['O', 1.4352000000000005, 1.4352000000000005, 0.0],['O', 3.3488000000000007, 3.3488000000000007, 0.0],['O', 0.9568000000000001, 3.8272000000000004, 1.5340000000000007],['O', 3.8272000000000004, 0.9568000000000001, 1.5340000000000007]]

configs = [config1,config2,config3,config4,config5,config6,config7,config8]

enn_array = []

for config in configs:
    G,_ = descriptors(config) 
    G_ = np.asarray([x.reshape(len(x),1,70) for x in [G]])
    #Normalizing the data
    min_max = np.load('params/min_max_params.npz')
    g_min,g_max = min_max['min'],min_max['max']
    min_max_norm(G_,g_min,g_max)

    e_nn,_ = predict_energy_2(np.asarray(G_),nn_Ti,nn_O)      #prediciton of energy using the neural network
    #print(e_nn)
    enn_array.append(e_nn[0])

E_val = enn_array
V_val = [58.75108702399997, 60.30618319748107, 61.88848153184836, 63.49821788398833, 65.13562811078714, 66.8009480691312, 68.49441361590668, 70.21626060799997]

eos = EquationOfState(V_val,E_val)
v0,e0,B = eos.fit()

f = open('results/result_eos.txt','w') #writes results to txt file


print('\n\n-----------------Birch-Murnaghan Equation of states of rutile TiO2-------------------------',file=f)
print('Equilibrium volume = ','{0: <4.4f}'.format(v0),'eV',file=f)
print('Equilibrium energy = ','{0: <4.4f}'.format(e0),'Ä³',file=f)
print('Bulk modulus predicted by NN model  = ','{0: <4.4f}'.format(B/kJ*1e24),'GPa',file=f)
print('Bulk modulus found  using DFT       =  211 GPa',file=f)
print('--------------------------------------------------------------------------------------------',file=f)

print('\n-----------------Birch-Murnaghan Equation of states of rutile TiO2-------------------------')
print('Equilibrium volume = ','{0: <4.4f}'.format(v0),'eV')
print('Equilibrium energy = ','{0: <4.4f}'.format(e0),'Ä³')
print('Bulk modulus predicted by NN model  = ','{0: <4.4f}'.format(B/kJ*1e24),'GPa')
print('Bulk modulus found using DFT        =  211 GPa')
print('--------------------------------------------------------------------------------------------')

#print(v0,e0,B/kJ * 1e24)

fig = plt.figure(figsize = (7,4),dpi =150)
plt.xlabel('volume[Ä³]')
plt.ylabel('energy [eV]')
plt.title('Equation of states of rutile TiO2')
fig.tight_layout()
plt.grid('True')  
eos.plot('eos_try.png')  
fig.savefig('plots/eos')
plt.show()