from neural_network2 import*
from symmetry import *
from reader import*


#Extracting the weights of trained NN

'''Ti_weights = np.load('trained_ti_weights.npz')
O_weights = np.load('trained_O_weights.npz')
w1_ti = Ti_weights['w1']
w2_ti = Ti_weights['w2']
w3_ti = Ti_weights['w3']
w1_ox = O_weights['w1']
w2_ox = O_weights['w2']
w3_ox = O_weights['w3']'''

trained_params = np.asarray(load_params())

node_list = [70,11,11,1]          #contains the layer sizes
activations = ['sigmoid','sigmoid','linear']

nn_Ti = NeuralNetwork(node_list,activations)
initialize_params(nn_Ti,*trained_params[0:6])
#weight_initialize(nn_Ti,w1_ti,w2_ti,w3_ti)
nn_O = NeuralNetwork(node_list,activations)
initialize_params(nn_O,*trained_params[6:])
#weight_initialize(nn_O,w1_ox,w2_ox,w3_ox)

path = './md_structure_ref'
file_list = sorted(os.listdir(path))
_,_,atom_data = xsf_reader(file_list[0])
##print(atom_data[:,1:])
#exit(0)
print(len(atom_data))
A = []
A.append([i[1:] for i in atom_data])
points = np.asarray(A[0])
print(points)
#exit(0)
G,dG_dr = symmetry_function(atom_data)

print(G.shape,dG_dr.shape)
print(dG_dr[0])




G_file,E_ref,_ = data_read(path)
print(G_file,E_ref)
G = np.asarray(np.loadtxt(os.path.join('./symmetry_functions_demo','%s') %(G_file[0][:-3]+'txt')))
##print(G)

print(np.asarray(G).shape)

G_ = np.asarray([x.reshape(len(x),1,70) for x in [G]])
#print(G_[0])
#exit(0)
min_max = np.load('min_max_params.npz')
g_min,g_max = min_max['min'],min_max['max']

###print(g_min,g_max)
min_max_norm(G_,g_min,g_max)

#print(G_)

a,b,c = predict_energy(np.asarray(G_),np.asarray(E_ref),nn_Ti,nn_O)

print(a,b,c)

e_nn,e_i_array = predict_energy_2(np.asarray(G_),nn_Ti,nn_O)
print(e_nn,e_i_array)

dEi_dG = np.asarray(structure_nn_gradient(e_i_array,nn_Ti,nn_O))
print(dEi_dG)

forces = []
for i in range(len(dEi_dG)):
    
    force = np.asarray(dEi_dG[i][0].dot(dG_dr[i]))
    #print(force)
    forces.append(force)

forces = np.asarray(forces)
print(forces)

mass_Ti = 47.867
mass_O = 15.999

def acceleration():
    A = []
    n = nn_switcher(forces)
    for i in range(len(forces)):
        if i<=(n-1):
            A.append(forces[i]/mass_Ti)
        else:
            A.append(forces[i]/mass_O)
    
    print(np.asarray(A))
    return np.asarray(A)

Acc = acceleration()

delta_t = 1e-12
def velocity():
    initial_V = np.zeros(Acc.shape)
    previous_V = initial_V
    next_V = previous_V + (Acc*delta_t)
    print(initial_V)
    print(next_V)
    return next_V

Vel = velocity()

def displacement():
    initial_position = points
    next_position = initial_position + (Vel * delta_t)
    print(points)
    print(next_position)
    return next_position

displacement()