"""
====================================================================================================
Molecular dynamics module
----------------------------------------------------------------------------------------------------
A simple molecular dynamics simulation is implemented using the generated neural network potential.
-Initially the neural network is called and the trained weights are loaded and made it ready to predict.
-A structure from the dataset is chosen and its position update is done for 20 timesteps.
-NN predicts the Energy,whose gradient provides the force.This gives the acceleration and it is 
numerically integrated to calculate velocity and displacements and the all the atoms are displaced to 
get positions at the next time step.
====================================================================================================
"""
# ==================================================================================================
# imports
# --------------------------------------------------------------------------------------------------
from neural_network import*
from symmetry import *
from reader import*
from visualizer import*
# ==================================================================================================


#Calling the neural network and assigning the weights of trained NN
trained_params = np.asarray(load_params('trained'))
node_list = [70,11,11,1]          #contains the layer sizes
activations = ['sigmoid','sigmoid','linear']
nn_Ti = NeuralNetwork(node_list,activations)
initialize_params(nn_Ti,*trained_params[0:6])  #loading the weights
nn_O = NeuralNetwork(node_list,activations)
initialize_params(nn_O,*trained_params[6:])

path = './md_structure_ref'
file_list = sorted(os.listdir(path))
_,_,atom_data = xsf_reader(file_list[2])

n = len(atom_data)
Atoms = [atom_data[i][0] for i in range(len(atom_data)) ]

mass_Ti = 47.867 
mass_O = 15.999  
m_array = np.asarray([47.867,47.867,15.999,15.999,15.999,15.999]).reshape(1,6)

enn_array= []
KE_array = []
PE_array = []

class MolecularDynamics:
    """
    Class for implementind the MD simulations.
    """
    
    def __init__(self):
        """
        Initializes the memory location for velocity vector,position and other constants
        """
        self.num_of_atoms = n
        self.dimension = 3   #as we are doing simulation in 3d
        self.positions = np.zeros((self.num_of_atoms,self.dimension))
        self.velocities = np.zeros((self.num_of_atoms,self.dimension))
        self.Kb = 8.617330337217213e-05 
        self.T = 1000 

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
        uniform distribution.
        """
        val1 = np.sqrt((self.Kb*self.T)/(mass_Ti))
        val2 = np.sqrt((self.Kb*self.T)/(mass_O))
        size = self.num_of_atoms
        index = nn_switcher(self.positions)   #finding index to switch atomtype from 1 to 2
        self.velocities[:index] = np.random.normal(loc=0,scale = val1,size =(index,self.dimension))      #initial velocities for atomtype1 -- Ti
        self.velocities[index:] = np.random.normal(loc=0,scale = val2,size =(size-index,self.dimension)) #initial velocities for atomtype2 -- O

        return 0

    def descriptors(self,atom_data):
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

    

    def force_predict(self,G,dG_dr):
        """
        Calculates the force by taking the gradient of energy from neural network.The gradient is 
        calculated by applying chain rule.
        Arguments:
        G,dG_dr -- outputs of descriptors function

        Return:
        forces -- force vector(3 components) on each atoms
        """
        G_ = np.asarray([x.reshape(len(x),1,70) for x in [G]])

        #normalizing the input data before prediciton
        min_max = np.load('params/min_max_params.npz')
        g_min,g_max = min_max['min'],min_max['max']
        min_max_norm(G_,g_min,g_max)

        e_nn,e_i_array = predict_energy_2(np.asarray(G_),nn_Ti,nn_O)
        enn_array.append(e_nn[0])

        dEi_dG = np.asarray(structure_nn_gradient(e_i_array,nn_Ti,nn_O))

        forces = []
        for i in range(len(dEi_dG)):  
            force = -np.asarray(dEi_dG[i][0].dot(dG_dr[i])) #chain rule
            forces.append(force)
        forces = np.asarray(forces)
        return forces,e_nn

    def acceleration(self,forces):
        """
        Calculates acceleration from the force using Newtons law,F = m*a.
        Arguments:
        forces -- force vector(3 components) on each atoms
        Return:
        Acc -- acceleration vector of each atom
        """
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
        """
        Updates positions using explicit euler method.
        (Euler method is not used in MD simulations as it is numerically unstable.Just to see whether the code works.)
        Arguments:
        forces -- force vector(3 components) on each atoms
        Return:
        next_position -- position after the timestep

        """
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

        return next_position,Vel

    def final_position(self):
        """Functions returns the current atomic positions"""
        return self.positions

    #Velocity-Verlet alogorithm---------------------------------------------------------------------
    #Velocity verlet is implemented using 2 functions. 
    #Initially velocity at half time-step forward is computed.
    #We use this half-velocity to obtain new positions.
    #Then we calculate new acceleration from forces based on the new positons

    def velocity_verlet_a(self,forces,acc):
        """
        Implements the velocity verlet using 2 functions
        Arguments:
        forces -- force vector(3 components) on each atoms
        acc -- accelration.
        Return:
        next_position -- new position
        previous_V -- old velocity
        """
        previous_position = self.positions
        previous_V = self.velocities
        previous_A = acc

        next_position = previous_position + (previous_V * delta_t) + (0.5 * previous_A * delta_t**2)
        self.positions = next_position
        return next_position,previous_V

    
    def velocity_verlet_b(self,acc1,acc2):
        """
        Arguments:
        acc1 -- old accelration.
        acc2 -- new accelration.        
        Return:
        next_V -- new velocity
        """
        previous_A = acc1
        next_A = acc2
        previous_V = self.velocities
        next_V = previous_V + 0.5*(previous_A + next_A)* delta_t
        self.velocities = next_V
        return next_V
    #-----------------------------------------------------------------------------------------------


f = open('results/TiO2_md_run.xyz',"w+") #opening an xyz file to store positions of each timesteps
frame=0
def xyz_writer(A,frame):
    """
    Function towrite the positions of each time step in an xyz forman,in order to visualize using 
    Ovito--(Open Visualization Tool)
    Arguments:
    A -- list of atomic positions of each atom
    frame -- value which counts the timesteps
    Return:
    none
    """    
    f.write(str(n)+'\n')
    f.write(str(frame)+'\n')
    for atom in A:
        f.write(str(atom[0])+'\t'+str(atom[1])+'\t'+str(atom[2])+'\t'+str(atom[3])+'\n')
    return 0


#Initializing values--------------------------------------------------------------------------------
fs = 0.09822694788464063 #femtosecond
start_t = 0
stop_t  = 20 * fs
delta_t = 1 * fs
if __name__ == '__main__':

    visualize(atom_data)     #calling visualize module to plot initial positon
    md = MolecularDynamics() #making object for md simulations


    print("\n--------------------------------MOLECULAR DYNAMICS SIMULATION----------------------------------\n")
    print('##########  Updates position after each timestep using the Neural network potential.  ############\n\n')
    tic = time.time()

    #printing the initial positions of the 6 atoms------------------------------------------------------
    B = []
    B.append([i[1:] for i in atom_data])   #data from a random xsf file
    initial_state = np.asarray(B[0])
    print('#####################################')
    print('initial position \n------------------------------------\n',initial_state)
    print('#####################################\n')
    #---------------------------------------------------------------------------------------------------


    md.initial_positions()   #Initial conditions
    md.initial_velocities()

    n_steps = (stop_t-start_t)/delta_t
    count=0
    integration = 'Velocity verlet'
    print('------------------------------------')
    print('Integrator ----',integration)
    print('------------------------------------\n')

    #Position update with euler explicit ---------------------------------------------------------------
    if integration == 'Euler method':
        while count<n_steps:
            #calculating force and new positions using trained NN

            G_array,dG_dr_array = md.descriptors(atom_data)
            force_array,enn_val = md.force_predict(G_array,dG_dr_array)
            new_position,new_v = md.position_update(force_array)
            v_norm = np.asarray([np.linalg.norm(a) for a in new_v]).reshape(-1) #norm of velocity to find KE
            kin_e = 0.5*m_array*v_norm                                          #calculates KE
            KE_array.append(sum(kin_e[0]))

            print('{0: <6}'.format('time ='),'{:2}'.format(count),'fs','----','{0: <4}'.format('E_pot ='),'{:1.4f}'.format(*enn_val),' eV')     
            #print(count,'-------',enn_val,'\n')

            Atoms = np.asarray(Atoms).reshape(n,1)
            A = np.hstack((Atoms,new_position)).tolist()
            for i in range(len(A)):                                  #nested loop to round the values except the char datatype (atomtype)
                for j in range(len(A[i])):
                    if j>0:
                        A[i][j]= round(float(A[i][j]),6)
            xyz_writer(A,frame)                                      #writes frame details of current step to xyz file

            atom_data = A
            frame +=1
            count+=1

    #Position update with velocity verlet algorithm ----------------------------------------------------
    elif integration == 'Velocity verlet':
        while count<n_steps:
            #calculating force and new positions using trained NN

            G_array,dG_dr_array = md.descriptors(atom_data)
            force_array,enn_val = md.force_predict(G_array,dG_dr_array)
            accelerations1 = md.acceleration(force_array)
            new_position,old_v = md.velocity_verlet_a(force_array,accelerations1)
            
            print('{0: <6}'.format('time ='),'{:2}'.format(count),'fs','----','{0: <4}'.format('E_pot ='),'{:1.4f}'.format(*enn_val),' eV')     


            Atoms = np.asarray(Atoms).reshape(n,1)
            A = np.hstack((Atoms,new_position)).tolist()

            for i in range(len(A)):                                          #nested loop to round the values except the char datatype (atomtype)
                for j in range(len(A[i])):
                    if j>0:
                        A[i][j]= round(float(A[i][j]),6)

            G_array2,dG_dr_array2 = md.descriptors(A)
            new_force_array,_ = md.force_predict(G_array2,dG_dr_array2)
            accelerations2 = md.acceleration(new_force_array)    
            new_V = md.velocity_verlet_b(accelerations1,accelerations2)

            v_norm = np.asarray([np.linalg.norm(a) for a in new_V]).reshape(-1) #norm of velocity to find KE
            kin_e = 0.5*m_array*v_norm                                          #calculates KE
            KE_array.append(sum(kin_e[0]))

            xyz_writer(A,frame)                                                #writes frame details of current step to xyz file
            atom_data = A
            frame +=1
            count+=1


    if integration=='Velocity verlet':
        PE_array = [v for i, v in enumerate(enn_array) if i % 2 == 0]               #taking alternate values  only due to half velocity 
    else:#euler case
        PE_array = enn_array

    TE_array = np.asarray(PE_array)+np.asarray(KE_array)

    #printing the final positions of the 6 atoms--------------------------------------------------------
    final_state = md.final_position()
    visualize(atom_data)
    print('\n\n#####################################')
    print('final position \n------------------------------------\n',final_state)
    print('#####################################')

    print('\nPositions of each atoms after each time step have been saved in xyz file format.')
    toc = time.time()
    print('--------------------------------------------------------------------------------------------')
    print('Time taken =',str((toc-tic)) + 'sec')
    print('--------------------------------------------------------------------------------------------\n')

    #plotting energy

    fig = plt.figure(figsize = (6,4),dpi =150)                                                                          
    plt.plot(PE_array,label='P.E')  
    plt.xlabel('frame')
    plt.ylabel('Potential Energy (eV)')
    plt.legend()
    plt.title('PE variation')
    fig.tight_layout()
    plt.show()
    fig.savefig('plots/PE.png')


    fig = plt.figure(figsize = (6,4),dpi =150)                                                                          
    plt.plot(KE_array,label='K.E')  
    plt.xlabel('frame')
    plt.ylabel('Kinetic Energy (eV)')
    plt.legend()
    plt.title('KE variation')
    fig.tight_layout()
    plt.show()
    fig.savefig('plots/KE.png')



    fig = plt.figure(figsize = (6,4),dpi =150)                                                                          
    plt.plot(KE_array,label='K.E') 
    plt.plot(PE_array,label='P.E',marker='o')  
    plt.plot(TE_array,label='T.E',marker='+')   
    plt.xlabel('frame')
    plt.ylabel('Energy (eV)')
    plt.legend()
    plt.title('Energy variation')
    fig.tight_layout()
    plt.show()
    fig.savefig('plots/TE.png')

    #---------------------------------------------------------------------------------------------------

