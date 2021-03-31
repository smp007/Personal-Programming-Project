"""
====================================================================================================
A simple test for the MD code.
In this test 6 atoms are arranged in a line in X direction and their velocities are set to zero.
Now equal force in Y direction is given to them in each time step to check whether they all move in 
Y direction.The initial state and final state are showed via 3d plots.
====================================================================================================
"""
# ==================================================================================================
# imports
# --------------------------------------------------------------------------------------------------
from md import MolecularDynamics
from visualizer import*
#---------------------------------------------------------------------------------------------------
f = open('results/result_md_test.txt','w') #writes results to txt file

print("\n------------------------------     Test for MD code      --------------------------------------\n",file=f)
print('#############  Checks whether force in Y direction moves the atoms in Y direction. ###########\n\n',file=f)

print('ie.Only the Y coordinates of each atom chould change from 0 to a particular value\n',file=f)


atom_data = [['Ti',0.0, 0.0, 0.0],['Ti',1, 0, 0],['O',2, 0, 0],['O',3,0, 0.0],['O',4, 0, 0],['O',5, 0, 0]]   #Atoms are arranged in a line
visualize(atom_data)

md_test = MolecularDynamics()
md_test.T = 0                       

A = []
A.append([i[1:] for i in atom_data])

md_test.positions = np.array(A[0])
md_test.initial_velocities()

print('Initial Position :\n\n',md_test.positions)
print('Initial Position :\n\n',md_test.positions,'\n',file=f)

#print(md_test.velocities)

#exit(0)

force_val = [['Ti',0.0, 1.0, 0.0],['Ti',0, 1, 0],['O',0, 1, 0],['O',0,1, 0.0],['O',0, 1, 0],['O',0, 1, 0]]


#force_val = [['Ti',0, 0, 1.0],['Ti',0, 0, 1],['O',0, 0, 1],['O',0,0, 1.0],['O',0, 0, 1],['O',0, 0, 1]]

B = []
forces = B.append([i[1:] for i in force_val])
force_array_t = np.array(B[0])
print('Force at each timestep in Y direction:\n\n',force_array_t)
print('Force at each timestep in Y direction:\n\n',force_array_t,file=f)

#print(md_test.acceleration(force_array_t))

#print(md_test.positions)

count = 0

while count<500:

    acc_1 = md_test.acceleration(force_array_t)
    new_position_t,old_v_t = md_test.velocity_verlet_a(force_array_t,acc_1)

    acc_2 = md_test.acceleration(force_array_t)    
    new_V = md_test.velocity_verlet_b(acc_1,acc_2)
    count+=1


print('\nFinal position :\n',new_position_t)
print('\nFinal position :\n',new_position_t,file=f)


Atoms = [atom_data[i][0] for i in range(len(atom_data)) ]
Atoms = np.asarray(Atoms).reshape(6,1)
A = np.hstack((Atoms,new_position_t)).tolist()


for i in range(len(A)):                                #nested loop to round the values except the char datatype (atomtype)
    for j in range(len(A[i])):
        if j>0:
            A[i][j]= round(float(A[i][j]),3)

visualize(A)

print('--------------------------------------------------------------------------------------------\n',file=f)


#Same as above but,
#---------------------------------------------------------------------------------------------------
#Force in Z direction --Movement in Z direction
#---------------------------------------------------------------------------------------------------
atom_data = [['Ti',0.0, 0.0, 0.0],['Ti',1, 0, 0],['O',2, 0, 0],['O',3,0, 0.0],['O',4, 0, 0],['O',5, 0, 0]]   #Atoms are arranged in a line
visualize(atom_data)

md_test = MolecularDynamics()
md_test.T = 0                       

A = []
A.append([i[1:] for i in atom_data])

md_test.positions = np.array(A[0])
md_test.initial_velocities()

print('Initial Position :\n\n',md_test.positions)

force_val = [['Ti',0, 0, 1.0],['Ti',0, 0, 1],['O',0, 0, 1],['O',0,0, 1.0],['O',0, 0, 1],['O',0, 0, 1]]

B = []
forces = B.append([i[1:] for i in force_val])
force_array_t = np.array(B[0])
print('Force at each timestep in Z direction :\n\n',force_array_t)

count = 0

while count<500:

    acc_1 = md_test.acceleration(force_array_t)
    new_position_t,old_v_t = md_test.velocity_verlet_a(force_array_t,acc_1)

    acc_2 = md_test.acceleration(force_array_t)    
    new_V = md_test.velocity_verlet_b(acc_1,acc_2)
    count+=1


print('\nFinal position :\n',new_position_t)


Atoms = [atom_data[i][0] for i in range(len(atom_data)) ]
Atoms = np.asarray(Atoms).reshape(6,1)
A = np.hstack((Atoms,new_position_t)).tolist()


for i in range(len(A)):                                          #nested loop to round the values except the char datatype (atomtype)
    for j in range(len(A[i])):
        if j>0:
            A[i][j]= round(float(A[i][j]),3)

visualize(A)
