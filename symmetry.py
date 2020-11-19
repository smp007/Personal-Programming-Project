# ==============================================================================
# imports
# ------------------------------------------------------------------------------
import os 
import numpy as np
import reader

# ==============================================================================
#

path = './data_set_TiO2_small'
file_list = sorted(os.listdir(path))
file = file_list[1]
#print(file)
datapoints_list = []
_,_,datapoints_list=reader.xsf_reader(file)

# ==============================================================================
# symmetry function parameters
# ------------------------------------------------------------------------------
#radial function parameters
#-------------------------------------------------------------------------------
atom_list_r_items = ['O','Ti']
atom_list_r = atom_list_r_items[:]*8

eta_list_r = [0.003214,0.003214,0.035711,0.035711,0.071421,0.071421,
            0.124987,0.124987,0.214264,0.214264,0.357106,0.357106,
            0.714213,0.714213,1.428426,1.428426]

radial_parameters = [(a,b) for (a,b) in zip(atom_list_r,eta_list_r)]
#print((radial_parameters))
#-------------------------------------------------------------------------------
#angular_function_parameters
#-------------------------------------------------------------------------------
atom_list_a_items = [('O','O'),('O','Ti'),('Ti','Ti')]
atom_list_a =  atom_list_a_items[:]*18
#print((atom_list_a))

eta_list_a_items = [0.000357,0.000357,0.000357,0.028569,0.028569,0.028569,
            0.089277,0.089277,0.089277]
eta_list_a = eta_list_a_items[:]*6
#print(len(eta_list_a))

lambda_list_a_items = [-1,-1,-1,-1,-1,-1,-1,-1,-1,
                        1,1,1,1,1,1,1,1,1,]
lambda_list_a = lambda_list_a_items[:]*3
#print(len(lambda_list_a))

zeta_list_a_items = [1,2,4]
zeta_list_a = [i for i in zeta_list_a_items for j in range(18)]
#print(len(zeta_list_a))

angular_parameters = [(a,b,c,d,e) for ((a,b),c,d,e) in zip(atom_list_a,eta_list_a,lambda_list_a,zeta_list_a)]

#print(angular_parameters)
#-------------------------------------------------------------------------------
#Symmetry function definitions
#-------------------------------------------------------------------------------
R_c = 6.5
R_s = 0

def cutoff_function(R_ij,R_c):
    if R_ij <= R_c:
        return 0.5 * (np.cos((np.pi * R_ij)/R_c)+1)
    else:
        return 0

#print(cutoff_function(1,5))

def radial_distribution(R_ij,R_s,R_c,eeta):
    G = 0
    g_r = np.exp(-1 * eeta * (R_ij-R_s)**2) * cutoff_function(R_ij,R_c)
    #G = sum of g s
    return g_r

#print(radial_distribution(4.399711486557522,0,6.5,0.003214))
def angular_distribution():
    g_a = ((1 + lamda*np.cos(theeta))**zeta) * np.exp(-1*eeta*(R_ij**2+R_ik**2+R_jk**2)) * cutoff_function(R_ij,R_c)* cutoff_function(R_ik,R_c)*cutoff_function(R_jk,R_c)
    pass

#-------------------------------------------------------------------------------


class Atom:
    """An Atom."""

    def __init__(self,atomtype,x,y,z):
        self.atomtype = atomtype
        self.position = np.array((x,y,z))

    def __repr__(self):
        return f"Atom : {self.atomtype} at {self.position}"

    
atoms = (([Atom(atomtype,x,y,z) for (atomtype,x,y,z) in datapoints_list]))

#print(len(datapoints_list))

#print(np.linalg.norm(atoms[0].position-atoms[1].position))
#distances = [] 
distances_list_Ti = []
distances_list_O = []

def distance_btw_atoms(atoms):
    
    for i in range(len(atoms)):
        distances_Ti = []
        distances_O = []
        for j in range(len(atoms)):
            if i!=j:
                if atoms[j].atomtype == 'Ti':
                    R_ij = np.linalg.norm(atoms[i].position-atoms[j].position) #distance btw an atom and a 'Ti' atom
                    if i==0 and j==1:
                        eeta = radial_parameters[1][1] 
                        print(R_ij)
                        print(radial_distribution(R_ij,R_s,R_c,eeta))
                    if i==6:
                        pass
                        print(cutoff_function(R_ij,R_c))
                    distances_Ti.append(R_ij)
                if atoms[j].atomtype == 'O':
                    #print(R_ij)
                    R_ij = np.linalg.norm(atoms[i].position-atoms[j].position) #distance btw an atom and a 'O' atom
                    distances_O.append(R_ij)

        distances_list_Ti.append(distances_Ti)
        distances_list_O.append(distances_O)        
    return distances_list_Ti,distances_list_O
            

((distance_btw_atoms(atoms)))


#print(atoms[8].atomtype)
