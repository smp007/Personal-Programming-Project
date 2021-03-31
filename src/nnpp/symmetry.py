"""
====================================================================================================
Symmetry module
----------------------------------------------------------------------------------------------------
Converts the local atomic environment of each atoms in to a symmetry vector made of 70 function 
values.
    -Reads the required attributes of data using xsf_reader function from reader module
    -Finds Radial distribution functions(16) followed by angular distribution functions (54) and 
    stacks them together to form the symmetry vector
    -Analytical derivatives of symmetry functions are also implemeted.This is done seperately for
    both radial parts and angular parts
    -The data in the Xsf format is read,converted to symmetry vectors and stored in as txt files 
    with the same name in another folder.Later we only need these txt files for NN training.
    -The for loop at the end automates the whole process and all the files are converted and stored 
    as txt files
====================================================================================================
"""# ==================================================================================================
# imports
# --------------------------------------------------------------------------------------------------
import os 
import numpy as np
from nnpp.reader import *
import time

# ==================================================================================================
# symmetry function parameters

# --------------------------------------------------------------------------------------------------
#radial function parameters
#---------------------------------------------------------------------------------------------------
eta_list_r = np.array([0.003214,0.035711,0.071421,0.124987,0.214264,0.357106,0.714213,1.428426])
#---------------------------------------------------------------------------------------------------
#angular_function_parameters
#---------------------------------------------------------------------------------------------------
# eeta,lamda,zeta values for 18 angular basis functions
angulaar_parameters = np.array([(0.000357, -1, 1),  (0.028569, -1, 1), (0.089277, -1, 1),(0.000357, -1, 2),  (0.028569, -1, 2), (0.089277, -1, 2),(0.000357, -1, 4),
(0.028569, -1, 4),(0.089277, -1, 4),(0.000357, 1, 1),(0.028569, 1, 1),(0.089277, 1, 1),(0.000357, 1, 2),(0.028569, 1, 2),(0.089277, 1, 2),
(0.000357, 1, 4),(0.028569, 1, 4),(0.089277, 1, 4)])
#---------------------------------------------------------------------------------------------------
#Symmetry function definitions
#---------------------------------------------------------------------------------------------------

R_c,R_s = 6.5,0

def cutoff_function(R_ij,R_c):
    """
    Defines the cut-off function as atoms outside the cutoff radius is not a part of central atom's 
    chemical environment.
    Arguments:
    R_ij -- distance between the central atom i and its neighbour j.
    R_c -- Radius of the cutoff sphere

    Return:
    case 1 -- monotonically decreasing part of cosine function
    case 2 -- 0
    """
    if R_ij <= R_c:
        return 0.5 * (np.cos(np.pi * R_ij/R_c) + 1)
    else:
        return 0

def d_cutoff_function(R_ij,R_c):
    """
    Derivative the cutoff function implemented above
    Arguments:
    R_ij -- distance between the central atom i and its neighbour j.
    R_c -- Radius of the cutoff sphere

    Return:
    case 1 -- monotonically decreasing part of cosine function
    case 2 -- 0
    """
    if R_ij <= R_c:
        return -0.5 *(np.pi/R_c)*(np.sin(np.pi * R_ij/R_c))
    else:
        return 0


def radial_distribution(R_ij_array,R_s,R_c,eeta):
    """
    Describes the radial distribution of atoms inside the cutoff sphere
    Arguments :
    R_ij_array -- array of the inter-atomic distance between the central atom and its neighbours
    R_s  -- Shifting parameter
    R_c  -- Cutoff radius
    eeta -- width parameter which determines the radial extension of symmetry functions
    Returns:
    g_r_sum --sum of products of Gaussians and the cutoff function
    """
    g_r_sum = 0
    for i in range(len(R_ij_array)):
        g_r = np.exp((-1 * eeta * (R_ij_array[i]-R_s)**2)) * cutoff_function(R_ij_array[i],R_c)        
        g_r_sum += g_r
    return g_r_sum

def d_radial_distribution(R_ij_scalar,R_ij_vector,R_s,R_c,eeta):
    """
    Analytic derivative of the radial part mentioned above
    Arguments :
    R_ij_scalar -- Norm of the vector R_ij
    R_ij_vector -- Vector pointing from point j to point i(array with 3 components)
    R_s  -- Shifting parameter
    R_c  -- Cutoff radius
    eeta -- width parameter which determines the radial extension of symmetry functions
    Returns:
    d_g_r_sum --sum of a complex analytic expression
    """
    d_g_r_sum_array = []
    for eeta in eta_list_r:
        d_g_r_sum = np.zeros((1,3))
        for i in range(len(R_ij_scalar)):
            ##d_g_r = ((1/R_ij_scalar[i])*np.exp((-1*eeta*(R_ij_scalar[i]-R_s)**2))*(d_cutoff_function(R_ij_scalar[i],R_c)-2*eeta*(R_ij_scalar[i]-R_s)*cutoff_function(R_ij_scalar[i],R_c)))* R_ij_vector[i]   #grad_i_i
            d_g_r = -1*((1/R_ij_scalar[i])*np.exp((-1*eeta*(R_ij_scalar[i]-R_s)**2))*(d_cutoff_function(R_ij_scalar[i],R_c)-2*eeta*(R_ij_scalar[i]-R_s)*cutoff_function(R_ij_scalar[i],R_c)))* R_ij_vector[i]   #grad_j_i
            d_g_r_sum += np.asarray(d_g_r)
        d_g_r_sum_array.append(d_g_r_sum)
    return d_g_r_sum_array


def angular_distribution(theeta,R_ij,R_ik,R_jk,eeta,lamda,zeta):
    """
    Describes the angular distribution of atoms inside the cutoff sphere
    Arguments :
    theeta --  angle formed by the central atom i and the two interatomic distances Rij and Rik
    R_ij -- distance between the central atom i and its neighbour j.
    R_ik -- distance between the central atom i and its neighbour k.
    eeta -- parameter associated to atomic seperation
    lamda -- parameter associated to shape of cosine function
    zeta --  parameter associated to angular arrangement
    Returns:
    g_a -- angular distribution before summing up
    """
    g_a = ((1 + lamda*np.cos(theeta))**zeta) * np.exp(-1*eeta*(R_ij**2+R_ik**2+R_jk**2)) * cutoff_function(R_ij,R_c)* cutoff_function(R_ik,R_c)*cutoff_function(R_jk,R_c)
    return g_a

#angular function derivative
def psi(theeta,R1,R2,eeta,lamda,zeta):
    """
    Term in the analytic derivative of angular part of symmetry function.
    Arguments :
    theeta --  angle formed by the central atom i and the two interatomic distances Rij and Rik
    R1,R2 -- two inter atomic distances 
    eeta -- parameter associated to atomic seperation
    lamda -- parameter associated to shape of cosine function
    zeta --  parameter associated to angular arrangement
    Return :
    val -- functional value of the analytical expression
    """
    val = (-1/(R1*R2))*((lamda*zeta)/(1e-8+1+lamda*np.cos(theeta)))
    return val

def phi(theeta,R1,R2,eeta,lamda,zeta):
    """
    Term in the analytic derivative of angular part of symmetry function.
    Arguments :
    theeta --  angle formed by the central atom i and the two interatomic distances Rij and Rik
    R1,R2 -- two inter atomic distances 
    eeta -- parameter associated to atomic seperation
    lamda -- parameter associated to shape of cosine function
    zeta --  parameter associated to angular arrangement
    Return :
    val -- functional value of the analytical expression
    """    
    val = -1*psi(theeta,R1,R2,eeta,lamda,zeta)-(1/R1**2)*((lamda*zeta)/(1e-8+1+lamda*np.cos(theeta)))
    return val

def xhi(R1,R_c):
    """
    Term in the analytic derivative of angular part of symmetry function.
    Arguments :
    R1 -- inter atomic distance
    R_c -- Cutoff radius
    Return :
    val -- functional value of the analytical expression
    """
    val = (1/R1*cutoff_function(R1,R_c))*d_cutoff_function(R1,R_c)
    return val

def d_angular_distribution(theeta,R_ij,R_ij_vector,R_ik,R_ik_vector,R_jk,R_jk_vector,eeta,lamda,zeta):
    """
    Describes the derivative  angular distribution.
    Arguments :
    theeta --  angle formed by the central atom i and the two interatomic distances Rij and Rik
    R_ij -- distance between the central atom i and its neighbour j.
    R_ij_vector -- Vector pointing from point j to point i(array with 3 components)
    R_ik -- distance between the central atom i and its neighbour k.
    R_ik_vector -- Vector pointing from point k to point i(array with 3 components)
    eeta -- parameter associated to atomic seperation
    lamda -- parameter associated to shape of cosine function
    zeta --  parameter associated to angular arrangement
    Returns:
    d_g_a -- gradient of angular distribution before summing up
    """    
    ##d_g_a = angular_distribution(theeta,R_ij,R_ik,R_jk,eeta,lamda,zeta) * (((phi(theeta,R_ij,R_ik,eeta,lamda,zeta)-(2*eeta)+xhi(R_ij,R_c))*R_ij_vector) + ((phi(-1*theeta,R_ik,R_ij,eeta,lamda,zeta)-(2*eeta)+xhi(R_ik,R_c))*R_ik_vector)) #grad_i_i
    d_g_a = angular_distribution(theeta,R_ij,R_ik,R_jk,eeta,lamda,zeta) * (((phi(theeta,R_ij,R_ik,eeta,lamda,zeta)-(2*eeta)+xhi(R_ij,R_c))*-R_ij_vector) + ((psi(-1*theeta,R_ij,R_ik,eeta,lamda,zeta)-(2*eeta)+xhi(R_jk,R_c))*R_jk_vector)) #grad_j_i
    return d_g_a

class Atom:
    """An Atom class"""
    def __init__(self,atomtype,x,y,z):
        self.atomtype = atomtype
        self.position = np.array((x,y,z))
    def __repr__(self):
        return f"Atom : {self.atomtype} at {self.position}"



def symmetry_fun_r(atoms):
    """
    Function which considers the 2 cases in radial part seperately and calculates radial distribution.
    Case 1 -- Interaction with neibouring Ti atoms
    Case 2 -- Interaction with neighbouring O atoms
    Arguments:
    atoms -- list of lists containing atomtype and atomic positions of each atoms in a structure.
    Return: 
    val_array_r_2 -- returns radial distribution as a 2d array of shape (n x 16) where n is the 
    number of atoms.Out of 16,8 is related to case 1 and remaining 8 to case 2,as specified above.
    """
    gaussian_value_Ti_array = []
    gaussian_value_O_array = []
    for i in range(len(atoms)):
        distances_Ti = []
        distances_O = []        
        for j in range(len(atoms)):
            if i!=j :
                if atoms[j].atomtype == 'Ti':
                    R_ij = np.linalg.norm(atoms[i].position-atoms[j].position) #distance btw an atom and a 'Ti' atom (scalar)        
                    distances_Ti.append(R_ij)
                    gaussian_value_Ti = radial_distribution(distances_Ti,R_s,R_c,eta_list_r)

                if atoms[j].atomtype == 'O':
                    R_ij = np.linalg.norm(atoms[i].position-atoms[j].position) #distance btw an atom and a 'O' atom
                    distances_O.append(R_ij)
                    gaussian_value_O = radial_distribution(distances_O,R_s,R_c,eta_list_r)
                            
        gaussian_value_Ti_array = np.append(gaussian_value_Ti_array,gaussian_value_Ti)       
        gaussian_value_O_array = np.append(gaussian_value_O_array,gaussian_value_O)
         
    val_array_r = np.hstack((gaussian_value_Ti_array.reshape(-1,8),gaussian_value_O_array.reshape(-1,8)))
    val_array_r_2 = np.reshape(val_array_r,(-1,16))
    return (val_array_r_2)
            

#derivative of radial part
def d_symmetry_fun_r(atoms):
    """
    Function which considers the 2 cases in radial part seperately and calculates derivative of radial distribution.
    Case 1 -- Interaction with neibouring Ti atoms
    Case 2 -- Interaction with neighbouring O atoms
    Arguments:
    atoms -- list of lists containing atomtype and atomic positions of each atoms in a structure.
    Return: 
    der_array_r -- returns radial distribution derivative as a 3d array of shape (n x 16 x 3) where n is the 
    number of atoms.The 3 corresponds to the 3 components of vector.
    """    
    d_radial_sym_Ti_array = []
    d_radial_sym_O_array  = []
    for i in range(len(atoms)):
        R_scalar_Ti = []
        R_scalar_O  = []
        R_vector_Ti = []
        R_vector_O  = []
        for j in range(len(atoms)):        
            if i!=j:
                if atoms[j].atomtype == 'Ti':
                    R_ij = np.linalg.norm(atoms[j].position-atoms[i].position) #distance btw an atom and a 'Ti' ato
                    R_scalar_Ti.append(R_ij)
                    R_ij_vector = np.asarray(atoms[j].position-atoms[i].position)
                    R_vector_Ti.append(R_ij_vector)
                    d_radial_sym_Ti = d_radial_distribution(R_scalar_Ti,R_vector_Ti,R_s,R_c,eta_list_r)

                if atoms[j].atomtype == 'O':
                    R_ij = np.linalg.norm(atoms[j].position-atoms[i].position) #distance btw an atom and a 'O' atom
                    R_scalar_O.append(R_ij)
                    R_ij_vector = np.asarray(atoms[j].position-atoms[i].position)  
                    R_vector_O.append(R_ij_vector)                                      
                    d_radial_sym_O = d_radial_distribution(R_scalar_O,R_vector_O,R_s,R_c,eta_list_r)
      
        d_radial_sym_Ti_array = np.append(d_radial_sym_Ti_array,d_radial_sym_Ti)       
        d_radial_sym_O_array = np.append(d_radial_sym_O_array,d_radial_sym_O)   
   
    der_array_r = np.hstack((d_radial_sym_Ti_array.reshape(-1,24),d_radial_sym_O_array.reshape(-1,24))).reshape(-1,16,3)
    der_array_r_2 = np.reshape(der_array_r,(-1,16,3))

    return (der_array_r)


def symmetry_fun_a(atoms,atomtype_1,atomtype_2):
    """
    Function which facilitates the consideration of the 3 cases in angular part seperately and calculates angular distribution.
    Case 1 -- Interaction with O-O pair
    Case 2 -- Interaction with Ti-O pair
    case 3 -- Interaction with Ti-Ti pair
    Here a general function is written for considering 3 combinations.
    For case 1 both atomtypes are assigned as 'O'.
    For case 2 atomtype_1 = 'Ti' and atomtype_2 = 'O'
    for case 3 both atomtypes are assigned as 'Ti'
    Arguments:
    atoms -- list of lists containing atomtype and atomic positions of each atoms in a structure.
    atomtype_1 -- 'Ti' or 'O' 
    atomtype_2 -- 'Ti' or 'O'
    Return: 
    val_array_is_2 -- returns angular distribution  for one single case as a 2d array of shape (n x 18)
    where n is the number of atoms.Later we stack the arrays of other 2 case to form an array with shape
    (n x 54)
    """    
    val_array_is = np.array([])
    for i in range(len(atoms)):
        val_array_js = np.array([])
        for j in range(len(atoms)):
            val_array = ([])
            val_array_ks = []

            if i!=j:
                if atoms[j].atomtype == atomtype_1:

                    if i <len(atoms):
                        R_ij = np.linalg.norm(atoms[i].position-atoms[j].position) #distance btw an atom and a 'Ti' atom
                        for k in range(len(atoms)):
                            if k!=i and k!=j:
                                if atoms[k].atomtype == atomtype_2:
                                    #finding the inputs required for angular distribution function
                                    R_ik = np.linalg.norm(atoms[i].position-atoms[k].position)
                                    R_jk = np.linalg.norm(atoms[j].position-atoms[k].position)
                                    theeta = np.arccos(round(np.dot((atoms[j].position-atoms[i].position),(atoms[k].position-atoms[i].position))/(R_ij*R_ik),12))
                                    for p in range(18):
                                        val = angular_distribution(theeta,R_ij,R_ik,R_jk,*angulaar_parameters[p])       #calling the angular distribution function
                                        val_array.append(val)
                        #addition is done by appending to a list,reshaping it to a 2d array and then taking the sum over the rows
                        #(it was done in such a way to get a visual clarity while coding and later I chose to keep it that way.)               
                        val_array_np = np.array(val_array)
                        val_array_np_2 = np.reshape(val_array_np,(-1,18))
                        val_array_ks.append(val_array_np_2)
                        val_array_ks_np = np.array(val_array_ks)
                        val_array_js = np.append(val_array_js,val_array_ks_np[0].sum(axis=0))   #summing over all the Js

            if i <len(atoms):
                val_array_js_2 = np.reshape(val_array_js,(-1,18))           
        val_array_is = np.append(val_array_is,np.multiply(2**(1-(angulaar_parameters[:,2])),val_array_js_2.sum(axis=0))) #summing over all the Ks
    val_array_is_2 = np.reshape(val_array_is,(-1,18))
    return val_array_is_2


#derivative angular part
def d_symmetry_fun_a(atoms,atomtype_1,atomtype_2):
    """
    Function which facilitates the consideration of the 3 cases in angular part seperately and calculates angular distribution.
    Case 1 -- Interaction with O-O pair
    Case 2 -- Interaction with Ti-O pair
    case 3 -- Interaction with Ti-Ti pair
    Here a general function is written for considering 3 combinations.
    For case 1 both atomtypes are assigned as 'O'.
    For case 2 atomtype_1 = 'Ti' and atomtype_2 = 'O'
    for case 3 both atomtypes are assigned as 'Ti'
    Arguments:
    atoms -- list of lists containing atomtype and atomic positions of each atoms in a structure.
    atomtype_1 -- 'Ti' or 'O' 
    atomtype_2 -- 'Ti' or 'O'
    Return: 
    val_array_is_2 -- returns derivative of angular distribution  for one single case as a 3d array of shape (n x 18 x 3)
    where n is the number of atoms.Later we stack the arrays of other 2 case to form an array with shape
    (n x 54 x3)
    """  
    val_array_is = np.array([])

    for i in range(len(atoms)):
        val_array_js = np.array([])
        for j in range(len(atoms)):
            val_array = ([])
            val_array_ks = []

            if i!=j:
                if atoms[j].atomtype == atomtype_1:
                   
                    if i <len(atoms):
                        #finding the inputs required for angular distribution function's derivative
                        R_ij_vector = np.asarray(atoms[j].position-atoms[i].position)
                        R_ij = np.linalg.norm(R_ij_vector) #distance btw central atom and an atom of atom_type 1
                        for k in range(len(atoms)):
                            if k!=i and k!=j:
                                if atoms[k].atomtype == atomtype_2:
                                    R_ik_vector = np.asarray(atoms[k].position-atoms[i].position)
                                    R_ik = np.linalg.norm(R_ik_vector) #distance btw central atom and an atom of atom_type 2
                                    R_jk_vector = np.asarray(atoms[k].position-atoms[j].position)
                                    R_jk = np.linalg.norm(R_jk_vector)
                                    theeta = np.arccos(round(np.dot((atoms[j].position-atoms[i].position),(atoms[k].position-atoms[i].position))/(R_ij*R_ik),12))
                                    #theeta3 = 
                                    for p in range(18):
                                        d_val = d_angular_distribution(theeta,R_ij,R_ij_vector,R_ik,R_ik_vector,R_jk,R_jk_vector,*angulaar_parameters[p]) #calling the angular distribution function's gradient
                                        val_array.append(d_val)
                        #addition is done by appending to a list,reshaping it to a 2d array and then taking the sum over the rows                
                        val_array_np = np.array(val_array).reshape(-1,18,3)
                        val_array_np_2 = np.reshape(val_array_np,(-1,18,3))
                        val_array_ks.append(val_array_np_2)
                        val_array_ks_np = np.array(val_array_ks)
                        val_array_js = np.append(val_array_js,val_array_np.sum(axis=0))


            if i <len(atoms):
                val_array_js_2 = np.reshape(val_array_js,(-1,18,3))
  
        val_array_is = np.append(val_array_is,np.multiply(2**(1-(angulaar_parameters[:,2].reshape(18,1))),val_array_js_2.sum(axis=0)))
    val_array_is_2 = np.reshape(val_array_is,(-1,18,3))
    return val_array_is_2


def symmetry_function(datapoints_list):
    """
    Function which combines the radial part and angular part to form the whole symmetry vector as 
    well as its derivative.
    Arguments:
    dataponits_list -- list of lists containing atomtype and the atomic positions(x,y,z)
    Returns : 
    sym_fun -- The symmetry vector of the local environment of each atoms.The shape is (n x 70) where n is the number of atoms
    d_sym_fun --The derivative of symmetry vector of the local environment of each atoms.The shape is (n x 70 x 30) where n is the number of atoms
    """       
    atoms = (([Atom(atomtype,x,y,z) for (atomtype,x,y,z) in datapoints_list]))
    AtomType = np.array(datapoints_list)[:,0].reshape(-1,1)
    symmetry_fun_a_values = np.hstack((symmetry_fun_a(atoms,'O','O'),symmetry_fun_a(atoms,'Ti','Ti'),symmetry_fun_a(atoms,'Ti','O')))
    num  = len(symmetry_fun_a_values)
    #symmetry functions
    sym_fun = np.hstack((symmetry_fun_r(atoms),symmetry_fun_a_values))
    #gradient of symmetry function
    d_symmetry_fun_r_values = d_symmetry_fun_r(atoms)
    d_symmetry_fun_a_values =(np.hstack((d_symmetry_fun_a(atoms,'O','O'),d_symmetry_fun_a(atoms,'Ti','Ti'),d_symmetry_fun_a(atoms,'Ti','O'))))
    d_sym_fun = np.hstack((d_symmetry_fun_r_values,d_symmetry_fun_a_values))

    return sym_fun,d_sym_fun


if __name__ == '__main__': 
    tic = time.time()

    '''
    #code used for finding the missing files skipped due to errors while the xsf-->txt process was automated
    path = './data_set_TiO2'
    file_list_required = sorted(os.listdir(path))
    print(len(file_list_required))
    file_list_required2 = [x[:-4] for x in file_list_required]
    path = './symmetry_txt'
    file_list_current = sorted(os.listdir(path))
    print(len(file_list_current))
    file_list_current2 = [x[:-4] for x in file_list_current]
    missing = set(file_list_required2).difference(file_list_current2)
    missing_i = (sorted([int(x[-4:]) for x in missing]))
    final_xsf_list = sorted([x+'.xsf' for x in missing])
    print(final_xsf_list)
    '''

    path = './dataset_TiO2'
    file_list = sorted(os.listdir(path))
    f123 = open(os.path.join('./symmetry_txt','error.txt'),'w+') #opens a txt file to note the files causing error.
    print("\n-----------------------------------SYMMETRY MODULE-----------------------------------\n")
    print('###########  Writing data in the form of Symmetry vector to txt file.  ##############\n\n')
    for i,file in enumerate(file_list):
        '''This loop automates the whole proces implemented above for all the data (atomic positons in xsf file format) and 
        converts it into symmetry vector and writes to a txt file with the same name. '''
        if  i<5: #The whole process was already done.Now only considering the first 5 datasets to demonstrate the working of the module.
            
            print('structure no =',file[-8:-4]) 
            datapoints_list = []
            _,_,datapoints_list=xsf_reader(file)
            #print(datapoints_list)
            symm_fun,_ = symmetry_function(datapoints_list)
            file_name = file[:-3]+'txt'
            f = open(os.path.join('./symmetry_txt','%s') %file_name,"w+") #opens a txt file with the same name to store the data in the form of symmetry vector
            print('Reading from :',file,'\nWriting to :',file_name,'\n')
            s = np.matrix(symm_fun)
            np.savetxt(os.path.join('./symmetry_txt','%s') %file_name, s)

    toc = time.time()
    print('\nAll datasets were read and the corresponding symmetry vectors were written to txt files.')
    print('-------------------------------------------------------------------------------------')
    print('Time taken =',str((toc-tic)) + 'sec')
    print('-------------------------------------------------------------------------------------\n')
    exit(0)





