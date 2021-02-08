from neural_network2 import*
from symmetry import *
from reader import*


#Extracting the weights of trained NN
trained_params = np.asarray(load_params())
node_list = [70,11,11,1]          #contains the layer sizes
activations = ['sigmoid','sigmoid','linear']
nn_Ti = NeuralNetwork(node_list,activations)
initialize_params(nn_Ti,*trained_params[0:6])
nn_O = NeuralNetwork(node_list,activations)
initialize_params(nn_O,*trained_params[6:])

path = './md_structure_ref'
file_list = sorted(os.listdir(path))
_,_,atom_data = xsf_reader(file_list[0])


mass_Ti = 47.867
mass_O = 15.999
delta_t = 0.01


class MolecularDynamics:
    
    def __init__(self):
        self.num_of_atoms = 6
        self.dimension = 3
        self.positions = np.zeros((self.num_of_atoms,self.dimension))
        self.velocities = np.zeros((self.num_of_atoms,self.dimension))
        self.Kb = 1
        self.T = 1

    def initial_positions(self):

        #print(len(atom_data))
        A = []
        A.append([i[1:] for i in atom_data])
        points = np.asarray(A[0])
        self.positions = points


    def initial_velocities(self):
        val1 = np.sqrt((2*self.Kb*self.T)/(3*mass_Ti))
        val2 = np.sqrt((2*self.Kb*self.T)/(3*mass_O))
        size = self.num_of_atoms
        index = nn_switcher(self.positions)
        print(index)
        self.velocities[:index] = np.random.normal(loc=0,scale = val1,size =(index,self.dimension))
        self.velocities[index:] = np.random.normal(loc=0,scale = val2,size =(size-index,self.dimension))
        #self.velocities = np.random.normal(loc=0,scale = val,size =(self.num_of_atoms,self.dimension))
        #print(self.velocities)
        return 0

    def descriptors(self,atom_data):
        G,dG_dr = symmetry_function(atom_data)
        return G,dG_dr

    def force_predict(self,G,dG_dr):
        G_ = np.asarray([x.reshape(len(x),1,70) for x in [G]])

        #normalizing the input data b4 prediciton
        min_max = np.load('min_max_params.npz')
        g_min,g_max = min_max['min'],min_max['max']
        min_max_norm(G_,g_min,g_max)

        e_nn,e_i_array = predict_energy_2(np.asarray(G_),nn_Ti,nn_O)

        dEi_dG = np.asarray(structure_nn_gradient(e_i_array,nn_Ti,nn_O))

        forces = []
        for i in range(len(dEi_dG)):  
            force = -np.asarray(dEi_dG[i][0].dot(dG_dr[i]))
            forces.append(force)
        forces = np.asarray(forces)
        #print('force',forces)
        return forces
    
    def position_update(self,forces):
        A = []
        n = nn_switcher(forces)
        for i in range(len(forces)):
            if i<=(n-1):
                A.append(forces[i]/mass_Ti)
            else:
                A.append(forces[i]/mass_O)
        Acc = np.asarray(A)

        previous_V = self.velocities
        next_V = previous_V + (Acc*delta_t)
        print('current_v',previous_V)
        print('next_v',next_V)
        self.velocities = next_V
        Vel = next_V

        previous_position = self.positions
        next_position = previous_position + (Vel * delta_t)
        self.positions = next_position
        print('current_pos',previous_position)
        print('next_pos',next_position)

        return next_position

    def final_position(self):
        return self.positions

md = MolecularDynamics()
md.initial_positions()
md.initial_velocities()
t=0
while t<1:
    print('iteration=',t,'\n\n\n')

    G_array,dG_dr_array = md.descriptors(atom_data)
    force_array = md.force_predict(G_array,dG_dr_array)
    new_position = md.position_update(force_array)

    Atoms = np.asarray(['Ti','Ti','O','O','O','O']).reshape(6,1)
    #print(Atoms)
    A = np.hstack((Atoms,new_position)).tolist()
    #print('A',A)
    for i in range(len(A)):
        for j in range(len(A[i])):
            if j>0:
                A[i][j]= float(A[i][j])
    atom_data = A
    t+=delta_t


final_state = md.final_position()
print('##################################')
print('final position \n',final_state)
print('##################################')
#print(A)
#print(atom_data)
exit(0)





















#---------------------------------------------------------------------------------------------------
exit(0)
from neural_network2 import*
from symmetry import *
from reader import*


#Extracting the weights of trained NN
trained_params = np.asarray(load_params())
node_list = [70,11,11,1]          #contains the layer sizes
activations = ['sigmoid','sigmoid','linear']
nn_Ti = NeuralNetwork(node_list,activations)
initialize_params(nn_Ti,*trained_params[0:6])
nn_O = NeuralNetwork(node_list,activations)
initialize_params(nn_O,*trained_params[6:])


path = './md_structure_ref'
file_list = sorted(os.listdir(path))
_,_,atom_data = xsf_reader(file_list[0])
#print(len(atom_data))
A = []
A.append([i[1:] for i in atom_data])
points = np.asarray(A[0])
G,dG_dr = symmetry_function(atom_data)
print(G.shape,dG_dr.shape)
print(dG_dr[0])




G_file,E_ref,_ = data_read(path)
#print(G_file,E_ref)
#G = np.asarray(np.loadtxt(os.path.join('./symmetry_functions_demo','%s') %(G_file[0][:-3]+'txt')))
#exit(0)
##print(G)

print(np.asarray(G).shape)

G_ = np.asarray([x.reshape(len(x),1,70) for x in [G]])
#normalizing the input data b4 prediciton
min_max = np.load('min_max_params.npz')
g_min,g_max = min_max['min'],min_max['max']
min_max_norm(G_,g_min,g_max)

#a,b,c = predict_energy(np.asarray(G_),np.asarray(E_ref),nn_Ti,nn_O)

#print(a,b,c)

e_nn,e_i_array = predict_energy_2(np.asarray(G_),nn_Ti,nn_O)
print(e_nn,e_i_array)

dEi_dG = np.asarray(structure_nn_gradient(e_i_array,nn_Ti,nn_O))
print(dEi_dG)

forces = []
for i in range(len(dEi_dG)):
    
    force = -np.asarray(dEi_dG[i][0].dot(dG_dr[i]))
    #print(force)
    forces.append(force)

forces = np.asarray(forces)
print('force',forces)





mass_Ti = 47.867
mass_O = 15.999


#initial conditions

def acceleration():
    A = []
    n = nn_switcher(forces)
    for i in range(len(forces)):
        if i<=(n-1):
            A.append(forces[i]/mass_Ti)
        else:
            A.append(forces[i]/mass_O)
    
    print('acceleration',np.asarray(A))
    return np.asarray(A)

Acc = acceleration()

delta_t = 1
def velocity():
    #initial conditions
    initial_V = np.zeros(Acc.shape)
    previous_V = initial_V
    next_V = previous_V + (Acc*delta_t)
    print('initial_v',initial_V)
    print('next_v',next_V)
    return next_V

Vel = velocity()

def displacement():
    #initial conditions
    initial_position = points
    next_position = initial_position + (Vel * delta_t)
    print('initial_pos',points)
    print('next_pos',next_position)
    return next_position

displacement()



class MolecularDynamics:
    
    def __init__(self):
        self.num_of_atoms = 6
        self.dimension = 3
        self.positions = np.zeros((self.num_of_atoms,self.dimension))
        self.velocities = np.zeros((self.num_of_atoms,self.dimension))
        self.Kb = 1
        self.T = 1

    def initial_positions(self):
        self.positions = points

    def initial_velocities(self):
        val1 = np.sqrt((2*self.Kb*self.T)/(3*mass_Ti))
        val2 = np.sqrt((2*self.Kb*self.T)/(3*mass_O))
        size = self.num_of_atoms
        index = nn_switcher(points)
        print(index)
        self.velocities[:index] = np.random.normal(loc=0,scale = val1,size =(index,self.dimension))
        self.velocities[index:] = np.random.normal(loc=0,scale = val2,size =(size-index,self.dimension))
        #self.velocities = np.random.normal(loc=0,scale = val,size =(self.num_of_atoms,self.dimension))
        print(self.velocities)
        return 0

    def force(self):
        pass

md = MolecularDynamics()
md.initial_velocities()


Atoms = np.asarray(['Ti','Ti','O','O','O','O']).reshape(6,1)
print(Atoms)
print(points)
points2 = [np.hstack((Atoms[i],[int(points[i][j]) for j in range(3)])) for i in range(6)]
#print([np.hstack((Atoms[i],[float(points[i][j]) for j in range(3)])) for i in range(6)])

A = np.hstack((Atoms,displacement())).tolist()
print('A',A)
for i in range(len(A)):
    for j in range(len(A[i])):
        if j>0:
            A[i][j]= float(A[i][j])

print(A)
print(atom_data)
exit(0)

#print(A[:,1:])
#A[:,1:] = A[:,1:].astype(float)
#print(A)
#exit(0)
print(np.hstack((Atoms,points.astype(float))))
print(type(atom_data))
#print(symmetry_function(points))