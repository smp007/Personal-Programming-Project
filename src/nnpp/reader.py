'''
================================================================================
XSF file reader module
--------------------------------------------------------------------------------
takes in XSF file;
returns 1)structural energy
        2)No: of atoms
        3)Atom type & corresponding atomic co-ordinates
================================================================================
'''
#
# ==============================================================================
# imports
# ------------------------------------------------------------------------------
import numpy as np
#import pandas as pd
import os 
import matplotlib.pyplot as plt

# ==============================================================================
#

no_of_atoms = 0
path = './data_set_TiO2'
file_list = sorted(os.listdir(path))
# print(path)
#print(path+'/'+file_list[0])
# for file in file_list:
#     print(path+'/'+'%s'%(file))
def xsf_reader(file):
    """Takes in an XCrysden Structure File and returns the list of atomic 
    cordinates along with atom types"""
    #print(file)
    data_list = []
    with open(path+'/'+'%s'%(file)) as f:
        for i,line in enumerate(f):

            if i==0:  #energy line strips at =
                energy_list.append(line.strip().split('='))

            if i==8:  #to get no: of atoms...goes to that line strips and splits first element
                no_of_atoms = line.strip().split(' ')
                n=int(no_of_atoms[0])
        
            if (i>8 and i<(9+n)):  ##to take  n lines from the line 10 to 10+n
                data_list.append(line.strip().split('\n'))

    energy = float(energy_list[0][1][:-3])  #extracts requires number from line 1 in list form
 
    data = []
    #print(data_list[1])
    for i in range(len(data_list)):#splitting the line in data
        data.append(data_list[i][0].strip(' ').split('    '))
        #print(len(data[i][0]))
    data = np.array(data,dtype=list)

    #print(data)
    #print(data[:,0])
    #first column print(data[:,0])
    #remaining columns print(data[:,1:])
    AtomType = np.array([data[:,0]],dtype=str).T #transposes the atom type strings
    #print(AtomType)
    XYZ = np.array(data[:,1:4])   #data[:,1:]
    #print(XYZ)
    new_data = np.hstack((AtomType,XYZ))#stacking
    #print(new_data)
    # print(new_data[:,0])

    atom_coordinates = []
    atom_coordinates_list = []

    atom_coordinates = ([element,float(x),float(y),float(z)] for element,x,y,z in zip(new_data[:,0],new_data[:,1],new_data[:,2],new_data[:,3]))
    
    atom_coordinates_list = [*atom_coordinates]
    #print(atom_coordinates_list)
    #print(atom_coordinates_list)
    #print([*atom_coordinates])
    
    return energy,n,atom_coordinates_list

energy_value_list=[]
number_of_atoms_list = []
atom_data_list = []
#energy_value = 0
for file in file_list:
    energy_list = []
    data_list = []
    #print(file)
    if __name__ == "__main__":
        energy_value,number_of_atoms,atom_data = xsf_reader(file)
    #print(energy_value)
        energy_value_list.append(energy_value)
        number_of_atoms_list.append(number_of_atoms)
        atom_data_list.append(atom_data)

if __name__ == "__main__":
    print(energy_value_list)  
    print(number_of_atoms_list)  
    print((atom_data_list))

#print('{:<15}={:>17}'.format('Energy',energy))  #string formatting
#print('{:<15}={:>17}'.format('No: of atoms',n),"\n\n")


