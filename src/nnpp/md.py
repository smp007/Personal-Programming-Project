# ==================================================================================================
# imports
# --------------------------------------------------------------------------------------------------
from neural_network2 import*
from symmetry import *
from reader import*
from visualizer import*
# ==================================================================================================


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
_,_,atom_data = xsf_reader(file_list[1])
print(atom_data)
n = len(atom_data)
#print(n)
Atoms = [atom_data[i][0] for i in range(len(atom_data)) ]
print(Atoms)
#exit(0)

mass_Ti = 47.867 #* 1.66e-27
mass_O = 15.999 #* 1.66e-27

enn_array= []

class MolecularDynamics:
    
    def __init__(self):
        """
        Initializes the memory location for velocity vector,position and other constants
        """
        self.num_of_atoms = n
        self.dimension = 3
        self.positions = np.zeros((self.num_of_atoms,self.dimension))
        self.velocities = np.zeros((self.num_of_atoms,self.dimension))
        self.Kb = 8.617330337217213e-05 #0.00086173303#1#1.38e-23
        self.T = 500 #298.15

    def initial_positions(self):
        """
        Initializes the positions of atoms at time t = 0
        """
        A = []
        A.append([i[1:] for i in atom_data])
        points = np.asarray(A[0])
        self.positions = points


    def initial_velocities(self):
        """
        Each velocity component of every particle is assigned with a value that is drawn from a 
        uniform distribution
        """
        val1 = np.sqrt((self.Kb*self.T)/(mass_Ti))
        val2 = np.sqrt((self.Kb*self.T)/(mass_O))
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
        min_max = np.load('params/min_max_params.npz')
        g_min,g_max = min_max['min'],min_max['max']
        min_max_norm(G_,g_min,g_max)

        e_nn,e_i_array = predict_energy_2(np.asarray(G_),nn_Ti,nn_O)
        print(e_nn)
        enn_array.append(e_nn[0])


        dEi_dG = np.asarray(structure_nn_gradient(e_i_array,nn_Ti,nn_O))

        forces = []
        for i in range(len(dEi_dG)):  
            force = -np.asarray(dEi_dG[i][0].dot(dG_dr[i]))
            forces.append(force)
        forces = np.asarray(forces)
        return forces

    def acceleration(self,forces):
        A = []
        n = nn_switcher(forces)
        for i in range(len(forces)):
            if i<=(n-1):
                A.append(forces[i]/mass_Ti)
            else:
                A.append(forces[i]/mass_O)
        Acc = np.asarray(A)
        return Acc


    
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
        #print('current_v',previous_V)
        #print('next_v',next_V)
        self.velocities = next_V
        Vel = next_V

        previous_position = self.positions
        next_position = previous_position + (Vel * delta_t)
        self.positions = next_position
        #print('current_pos',previous_position)
        #print('next_pos',next_position)

        return next_position

    def final_position(self):
        return self.positions

    def velocity_verlet_a(self,forces,acc):

        previous_position = self.positions
        previous_V = self.velocities
        previous_A = acc


        next_position = previous_position + (previous_V * delta_t) + (0.5 * previous_A * delta_t**2)
        self.positions = next_position
        return next_position,previous_V
    
    def velocity_verlet_b(self,acc1,acc2):
        previous_A = acc1
        next_A = acc2
        previous_V = self.velocities
        next_V = previous_V + 0.5*(previous_A + next_A)* delta_t
        self.velocities = next_V
        return next_V


f = open('TiO2_try2.xyz',"w+")
frame=0
def xyz_writer(A,frame):
    
    f.write(str(n)+'\n')
    f.write(str(frame)+'\n')
    for atom in A:
        f.write(str(atom[0])+'\t'+str(atom[1])+'\t'+str(atom[2])+'\t'+str(atom[3])+'\n')
    return 0




#xyz_writer(A)
fs = 0.09822694788464063
start_t = 0
stop_t  = 20 * fs
delta_t = 1 * fs
visualize(atom_data)
md = MolecularDynamics()
md.initial_positions()
md.initial_velocities()

n_steps = (stop_t-start_t)/delta_t
count=0
integration = 'Velocity verlet'

if integration == 'Euler method':
    while count<n_steps:
        print('iteration=',count,'\n')

        G_array,dG_dr_array = md.descriptors(atom_data)
        force_array = md.force_predict(G_array,dG_dr_array)
        new_position = md.position_update(force_array)

        Atoms = np.asarray(Atoms).reshape(n,1)
        #print(Atoms)
        A = np.hstack((Atoms,new_position)).tolist()
        #print('A',A)
        for i in range(len(A)):
            for j in range(len(A[i])):
                if j>0:
                    A[i][j]= round(float(A[i][j]),6)
        #print('A',A)
        xyz_writer(A,frame)

        atom_data = A
        #t+=delta_t
        frame +=1
        count+=1

elif integration == 'Velocity verlet':
    while count<n_steps:
        print('iteration=',count,'\n')

        G_array,dG_dr_array = md.descriptors(atom_data)
        force_array = md.force_predict(G_array,dG_dr_array)
        accelerations1 = md.acceleration(force_array)
        #new_position = md.position_update(force_array)
        new_position,old_v = md.velocity_verlet_a(force_array,accelerations1)
        
        Atoms = np.asarray(Atoms).reshape(n,1)
        A = np.hstack((Atoms,new_position)).tolist()

        for i in range(len(A)):
            for j in range(len(A[i])):
                if j>0:
                    A[i][j]= round(float(A[i][j]),6)

        G_array2,dG_dr_array2 = md.descriptors(A)
        new_force_array = md.force_predict(G_array2,dG_dr_array2)
        accelerations2 = md.acceleration(new_force_array)    
        new_V = md.velocity_verlet_b(accelerations1,accelerations2)

        #print('A',A)
        xyz_writer(A,frame)

        atom_data = A
        #t+=delta_t
        frame +=1
        count+=1




final_state = md.final_position()
visualize(atom_data)
print('##################################')
print('final position \n',final_state)
print('##################################')


print(enn_array)

filtered_lst= [v for i, v in enumerate(enn_array) if i % 2 == 0]
print(filtered_lst)
print(len(filtered_lst))

fig = plt.figure(figsize = (7,4),dpi =150)
plt.plot(filtered_lst,'.:b')
plt.xlabel('m')
plt.ylabel('energy')
plt.title('E-v')
fig.tight_layout()
plt.grid('True')    
plt.show()

#print(A)
#print(atom_data)
exit(0)
