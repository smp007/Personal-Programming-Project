'''
====================================================================================================
XSF file reader module
----------------------------------------------------------------------------------------------------
takes in XSF file;
returns 1)structural energy
        2)No: of atoms
        3)Atom type & corresponding atomic co-ordinates
====================================================================================================
'''
#
# ==================================================================================================
# imports
# --------------------------------------------------------------------------------------------------
import numpy as np
import os 
import matplotlib.pyplot as plt

# ==================================================================================================
#

no_of_atoms = 0
path = './data_set_TiO2+outlier'
file_list = sorted(os.listdir(path))


def xsf_reader(file):
    """Takes in an XCrysden Structure File and returns the list of atomic 
    cordinates along with atom types and the total structural energy.
    
    Arguments :
    file -- file in the xsf file format

    Returns : 
    energy -- total energy of the structure
    n -- number of atoms in the structure file
    atom_coordinates_list -- gives the list of lists containing atomtype and the atomic positions(x,y,z)
    """
    #print(file)
    data_list = []
    with open(path+'/'+'%s'%(file)) as f:
        for i,line in enumerate(f):

            if i==0:                                                 #energy line strips at =
                energy_list.append(line.strip().split('='))

            if i==8:                                                 #to get no: of atoms...goes to that line strips and splits first element
                no_of_atoms = line.strip().split(' ')
                n=int(no_of_atoms[0])
        
            if (i>8 and i<(9+n)):                                   #to take  n lines from the line 10 to 10+n
                data_list.append(line.strip().split('\n'))

    energy = float(energy_list[0][1][:-3])                          #extracts requires number from line 1 in list form
 
    data = []
    for i in range(len(data_list)):                                 #splitting the line in data
        data.append((data_list[i][0].strip(' ').split('    ')))
    data = np.array(data,dtype=list)

    AtomType = np.array([data[:,0]],dtype=str).T                    #transposes the atom type strings

    XYZ = np.array(data[:,1:4])     

    new_data = np.hstack((AtomType,XYZ))                            #stacking

    atom_coordinates = []
    atom_coordinates_list = []
    atom_coordinates = ([element,float(x),float(y),float(z)] for element,x,y,z in zip(new_data[:,0],new_data[:,1],new_data[:,2],new_data[:,3]))  #list comprehension is used to zip the data in to a list  
    atom_coordinates_list = [*atom_coordinates]
  
    return energy,n,atom_coordinates_list

energy_value_list=[]
number_of_atoms_list = []
atom_data_list = []
for file in file_list:
    energy_list = []
    data_list = []
    if __name__ == "__main__":
        energy_value,number_of_atoms,atom_data = xsf_reader(file)
        energy_value_list.append(energy_value)
        number_of_atoms_list.append(number_of_atoms)
        atom_data_list.append(atom_data)

if __name__ == "__main__":#just a check for desired output.
    print("\n------------------------------------------- -READER MODULE------------------------------------\n")
    print('\n----------------------------------------------ENERGY VALUE LIST------------------------- \n\n',energy_value_list,'\n')  
    print('------------------------------------------LIST OF NUMBER OF ATOMS------------------------\n\n',number_of_atoms_list)  
    #print((atom_data_list))
    print('-----------------------------------------------------------------------------------------\n')


