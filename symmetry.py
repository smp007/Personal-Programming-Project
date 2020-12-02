# ==============================================================================
# imports
# ------------------------------------------------------------------------------
import os 
import numpy as np
import reader
import time

# ==============================================================================
#


#file = file_list[1]
#print(file)

#print(datapoints_list)


# ==============================================================================
# symmetry function parameters

# ------------------------------------------------------------------------------
#radial function parameters
#-------------------------------------------------------------------------------
eta_list_r = np.array([0.003214,0.035711,0.071421,
                    0.124987,0.214264,0.357106,
                     0.714213,1.428426])
#-------------------------------------------------------------------------------
#angular_function_parameters
#-------------------------------------------------------------------------------
# eeta,lamda,zeta values for 18 angular basis functions
angulaar_parameters = np.array([(0.000357, -1, 1),  (0.028569, -1, 1), (0.089277, -1, 1),(0.000357, -1, 2),  (0.028569, -1, 2), (0.089277, -1, 2),(0.000357, -1, 4),
(0.028569, -1, 4),(0.089277, -1, 4),(0.000357, 1, 1),(0.028569, 1, 1),(0.089277, 1, 1),(0.000357, 1, 2),(0.028569, 1, 2),(0.089277, 1, 2),
(0.000357, 1, 4),(0.028569, 1, 4),(0.089277, 1, 4)])
#print((angulaar_parameters))
#-------------------------------------------------------------------------------
#Symmetry function definitions
#-------------------------------------------------------------------------------

R_c,R_s = 6.5,0

def symmetry_function(datapoints_list):

    def cutoff_function(R_ij,R_c):
        if R_ij <= R_c:
            return 0.5 * (np.cos(np.pi * R_ij/R_c) + 1)
        else:
            return 0

    def radial_distribution(R_ij_array,R_s,R_c,eeta):
        g_r_sum = 0
        for i in range(len(R_ij_array)):
            g_r = np.exp((-1 * eeta * (R_ij_array[i]-R_s)**2)) * cutoff_function(R_ij_array[i],R_c)
            g_r_sum += g_r
        return g_r_sum

    #print(radial_distribution(4.399711486557522,0,6.5,0.003214))
    def angular_distribution(theeta,R_ij,R_ik,R_jk,eeta,lamda,zeta):
        g_a = ((1 + lamda*np.cos(theeta))**zeta) * np.exp(-1*eeta*(R_ij**2+R_ik**2+R_jk**2)) * cutoff_function(R_ij,R_c)* cutoff_function(R_ik,R_c)*cutoff_function(R_jk,R_c)
        return g_a

    #-------------------------------------------------------------------------------
    class Atom:
        """An Atom."""

        def __init__(self,atomtype,x,y,z):
            self.atomtype = atomtype
            self.position = np.array((x,y,z))
        def __repr__(self):
            return f"Atom : {self.atomtype} at {self.position}"
        
    atoms = (([Atom(atomtype,x,y,z) for (atomtype,x,y,z) in datapoints_list]))

    AtomType = np.array(datapoints_list)[:,0].reshape(-1,1)
    #print(AtomType)

    #print(len(datapoints_list))

    def symmetry_fun_r(atoms):
        gaussian_value_Ti_array = []
        gaussian_value_O_array = []
        for i in range(len(atoms)):
            distances_Ti = []
            distances_O = []
            
            for j in range(len(atoms)):

                if i!=j :
                    if atoms[j].atomtype == 'Ti':
                        R_ij = np.linalg.norm(atoms[i].position-atoms[j].position) #distance btw an atom and a 'Ti' atom
                        #if i<6 :                     
                        distances_Ti.append(R_ij)
                        #print(distances_Ti)
                        gaussian_value_Ti = radial_distribution(distances_Ti,R_s,R_c,eta_list_r)

                    if atoms[j].atomtype == 'O':
                        #print(R_ij)                    
                        #if i<6 :
                        R_ij = np.linalg.norm(atoms[i].position-atoms[j].position) #distance btw an atom and a 'O' atom
                        #print(R_ij)
                        distances_O.append(R_ij)
                        #print(distances_O)

                        gaussian_value_O = radial_distribution(distances_O,R_s,R_c,eta_list_r)
                                
            gaussian_value_Ti_array = np.append(gaussian_value_Ti_array,gaussian_value_Ti)                
            gaussian_value_O_array = np.append(gaussian_value_O_array,gaussian_value_O)
            
        val_array_r = np.hstack((gaussian_value_Ti_array,gaussian_value_O_array))
        val_array_r_2 = np.reshape(val_array_r,(-1,16))

        return (val_array_r_2)
                
    #here we get 2 arrays of 6 * 8 size..ie 48 each...8 for each six atoms

    #print(len(symmetry_fun_r(atoms)))

    def symmetry_fun_a(atoms,atomtype_1,atomtype_2):

        val_array_is = np.array([])

        for i in range(len(atoms)):
            val_array_js = np.array([])
            #j_val = 0
            for j in range(len(atoms)):
                val_array = ([])
                val_array_ks = []

                if i!=j:
                    if atoms[j].atomtype == atomtype_1:
                        #j_val+=1
                        ##print('j=',j_val)
                        
                        if i <len(atoms):#and (j==3 or j==4 or j==5):
                            R_ij = np.linalg.norm(atoms[i].position-atoms[j].position) #distance btw an atom and a 'Ti' atom
                            #k_val = 0
                            for k in range(len(atoms)):
                                if k!=i and k!=j:
                                    if atoms[k].atomtype == atomtype_2:
                                        R_ik = np.linalg.norm(atoms[i].position-atoms[k].position)
                                        R_jk = np.linalg.norm(atoms[j].position-atoms[k].position)
                                        theeta = np.arccos(round(np.dot((atoms[j].position-atoms[i].position),(atoms[k].position-atoms[i].position))/(R_ij*R_ik),12))
                                        ##print(R_ij,R_ik,R_jk,np.cos(theeta))
                                        for p in range(18):
                                            val = angular_distribution(theeta,R_ij,R_ik,R_jk,*angulaar_parameters[p])
                                            ##print(val)
                                            val_array.append(val)
                                        #k_val+=1
                                        ##print('k=',k_val)
                                            
                            val_array_np = np.array(val_array)
                            val_array_np_2 = np.reshape(val_array_np,(-1,18))
                                        #print(val_array)
                            val_array_ks.append(val_array_np_2)
                            val_array_ks_np = np.array(val_array_ks)
                                        #if k==len(atoms)-1:
                                        #print(val_sum)                                    
                                        #print((val_array_np))
                            
                            ##print((val_array_ks_np))
                            ##print(val_array_ks_np[0].sum(axis=0))
                            val_array_js = np.append(val_array_js,val_array_ks_np[0].sum(axis=0))
                            ##print(val_array_js)

                    #if i == 2 and (j==9 or j==10 or j==11 or j==12 or j==13):
                if i <len(atoms):#and (j==3 or j==4 or j==5):
                    val_array_js_2 = np.reshape(val_array_js,(-1,18))
                    #print(val_array_js_2)
                    #print(val_array_js_2.sum(axis=0))                        
                #val_array_array = np.append(val_array_array,val_array_np)            
            val_array_is = np.append(val_array_is,np.multiply(2**(1-(angulaar_parameters[:,2])),val_array_js_2.sum(axis=0)))
        val_array_is_2 = np.reshape(val_array_is,(-1,18))
        #return R_ij,R_ik,R_jk,np.cos(theeta)
        #return np.multiply(2**(1-(angulaar_parameters[:,2])),val_array_js_2.sum(axis=0))
        return val_array_is_2

        
    #print(symmetry_fun_a(atoms,'O','O'))
    #print(symmetry_fun_a(atoms,'Ti','Ti'))
    #print(symmetry_fun_a(atoms,'Ti','O'))

    symmetry_fun_a_values = np.hstack((symmetry_fun_a(atoms,'O','O'),symmetry_fun_a(atoms,'Ti','Ti'),symmetry_fun_a(atoms,'Ti','O')))

    print(len(symmetry_fun_a_values))
    num  = len(symmetry_fun_a_values)

    print(np.hstack((symmetry_fun_r(atoms),symmetry_fun_a_values)).shape)
    #print(np.hstack((symmetry_fun_r(atoms),symmetry_fun_a_values)))
    sym_fun = np.hstack((symmetry_fun_r(atoms),symmetry_fun_a_values))

    return sym_fun




tic = time.time()

path = './data_set_TiO2_small'
file_list = sorted(os.listdir(path))
for i,file in enumerate(file_list):
    if i>=81 and i<100:
        print('structure no =',i-1)
    
        datapoints_list = []
        _,_,datapoints_list=reader.xsf_reader(file)

        symm_fun = symmetry_function(datapoints_list)
        file_name = file[:-3]+'txt'
        f = open(os.path.join('./symmetry_functions','%s') %file_name,"w+") #os.path.join('./symmetry_functions',
        s = np.matrix(symm_fun)
        np.savetxt(os.path.join('./symmetry_functions','%s') %file_name, s)

        #print(np.loadtxt(os.path.join('./symmetry_functions','%s') %file_name))


toc = time.time()
print('Time taken =',str((toc-tic)) + 'sec')