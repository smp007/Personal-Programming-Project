# ==================================================================================================
# imports
# --------------------------------------------------------------------------------------------------
import os 
import numpy as np
import reader
import time

# ==================================================================================================
#


#file = file_list[1]
#print(file)

#print(datapoints_list)


# ==================================================================================================
# symmetry function parameters

# --------------------------------------------------------------------------------------------------
#radial function parameters
#---------------------------------------------------------------------------------------------------
eta_list_r = np.array([0.003214,0.035711,0.071421,
                    0.124987,0.214264,0.357106,
                     0.714213,1.428426])
#---------------------------------------------------------------------------------------------------
#angular_function_parameters
#---------------------------------------------------------------------------------------------------
# eeta,lamda,zeta values for 18 angular basis functions
angulaar_parameters = np.array([(0.000357, -1, 1),  (0.028569, -1, 1), (0.089277, -1, 1),(0.000357, -1, 2),  (0.028569, -1, 2), (0.089277, -1, 2),(0.000357, -1, 4),
(0.028569, -1, 4),(0.089277, -1, 4),(0.000357, 1, 1),(0.028569, 1, 1),(0.089277, 1, 1),(0.000357, 1, 2),(0.028569, 1, 2),(0.089277, 1, 2),
(0.000357, 1, 4),(0.028569, 1, 4),(0.089277, 1, 4)])
#print((angulaar_parameters))
#---------------------------------------------------------------------------------------------------
#Symmetry function definitions
#---------------------------------------------------------------------------------------------------

R_c,R_s = 6.5,0

def cutoff_function(R_ij,R_c):
    if R_ij <= R_c:
        return 0.5 * (np.cos(np.pi * R_ij/R_c) + 1)
    else:
        return 0

def d_cutoff_function(R_ij,R_c):
    if R_ij <= R_c:
        return -0.5 *(np.pi/R_c)*(np.sin(np.pi * R_ij/R_c))
    else:
        return 0


def radial_distribution(R_ij_array,R_s,R_c,eeta):
    g_r_sum = 0
    for i in range(len(R_ij_array)):
        g_r = np.exp((-1 * eeta * (R_ij_array[i]-R_s)**2)) * cutoff_function(R_ij_array[i],R_c)
        
        g_r_sum += g_r
    return g_r_sum

def d_radial_distribution(R_ij_scalar,R_ij_vector,R_s,R_c,eeta):
    d_g_r_sum_array = []
    for eeta in eta_list_r:
        d_g_r_sum = np.zeros((1,3))
        for i in range(len(R_ij_scalar)):
            d_g_r = ((1/R_ij_scalar[i])*np.exp((-1*eeta*(R_ij_scalar[i]-R_s)**2))*(d_cutoff_function(R_ij_scalar[i],R_c)-2*eeta*(R_ij_scalar[i]-R_s)*cutoff_function(R_ij_scalar[i],R_c)))* R_ij_vector[i]
            #print(d_g_r)
            #d_g_r_sum = [x+y for x,y in zip(d_g_r_sum,d_g_r)]
            d_g_r_sum += np.asarray(d_g_r)
        #print('sum',d_g_r_sum)
        d_g_r_sum_array.append(d_g_r_sum)

    return d_g_r_sum_array


#print(radial_distribution(4.399711486557522,0,6.5,0.003214))
def angular_distribution(theeta,R_ij,R_ik,R_jk,eeta,lamda,zeta):
    g_a = ((1 + lamda*np.cos(theeta))**zeta) * np.exp(-1*eeta*(R_ij**2+R_ik**2+R_jk**2)) * cutoff_function(R_ij,R_c)* cutoff_function(R_ik,R_c)*cutoff_function(R_jk,R_c)
    return g_a



class Atom:
    """An Atom."""

    def __init__(self,atomtype,x,y,z):
        self.atomtype = atomtype
        self.position = np.array((x,y,z))
    def __repr__(self):
        return f"Atom : {self.atomtype} at {self.position}"



def symmetry_fun_r(atoms):
    gaussian_value_Ti_array = []
    gaussian_value_O_array = []
    for i in range(len(atoms)):
        distances_Ti = []
        distances_O = []        
        for j in range(len(atoms)):
            if i!=j :
                if atoms[j].atomtype == 'Ti':
                    R_ij = np.linalg.norm(atoms[i].position-atoms[j].position) #distance btw an atom and a 'Ti' atom (scalar)
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
        
    #print(gaussian_value_O_array)    
    val_array_r = np.hstack((gaussian_value_Ti_array.reshape(-1,8),gaussian_value_O_array.reshape(-1,8)))
    #print(val_array_r)
    val_array_r_2 = np.reshape(val_array_r,(-1,16))
    #print(val_array_r_2)
    return (val_array_r_2)
            
#here we get 2 arrays of 6 * 8 size..ie 48 each...8 for each six atoms

#print(len(symmetry_fun_r(atoms)))

#derivative of radial part
def d_symmetry_fun_r(atoms):
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
                    R_ij = np.linalg.norm(atoms[i].position-atoms[j].position) #distance btw an atom and a 'Ti' ato
                   # print(R_ij)
                    R_scalar_Ti.append(R_ij)
                    
                    R_ij_vector = np.asarray(atoms[i].position-atoms[j].position)
                    #print(R_ij_vector)
                    R_vector_Ti.append(R_ij_vector)
                    #print(R_vector_Ti)
                    d_radial_sym_Ti = d_radial_distribution(R_scalar_Ti,R_vector_Ti,R_s,R_c,eta_list_r)
                    #print(d_radial_sym_Ti)
                if atoms[j].atomtype == 'O':
                    #print(R_ij)                    
                    #if i<6 :
                    R_ij = np.linalg.norm(atoms[i].position-atoms[j].position) #distance btw an atom and a 'O' atom
                    R_scalar_O.append(R_ij)
                    R_ij_vector = np.asarray(atoms[i].position-atoms[j].position)  
                    R_vector_O.append(R_ij_vector)                                      
                    d_radial_sym_O = d_radial_distribution(R_scalar_O,R_vector_O,R_s,R_c,eta_list_r)
                    #print(d_radial_sym_O)
        
        d_radial_sym_Ti_array = np.append(d_radial_sym_Ti_array,d_radial_sym_Ti)       
        d_radial_sym_O_array = np.append(d_radial_sym_O_array,d_radial_sym_O)   

    #print(d_radial_sym_Ti_array)      
    der_array_r = np.hstack((d_radial_sym_Ti_array.reshape(-1,24),d_radial_sym_O_array.reshape(-1,24))).reshape(-1,16,3)
    der_array_r_2 = np.reshape(der_array_r,(-1,16,3))

    return (der_array_r)



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


def symmetry_function(datapoints_list):
        
    atoms = (([Atom(atomtype,x,y,z) for (atomtype,x,y,z) in datapoints_list]))
    AtomType = np.array(datapoints_list)[:,0].reshape(-1,1)
    #print(AtomType)
    ###print(d_symmetry_fun_r(atoms))
    #print(len(datapoints_list))

    symmetry_fun_a_values = np.hstack((symmetry_fun_a(atoms,'O','O'),symmetry_fun_a(atoms,'Ti','Ti'),symmetry_fun_a(atoms,'Ti','O')))

    print(len(symmetry_fun_a_values))
    num  = len(symmetry_fun_a_values)

    print(np.hstack((symmetry_fun_r(atoms),symmetry_fun_a_values)).shape)
    #print(np.hstack((symmetry_fun_r(atoms),symmetry_fun_a_values)))
    sym_fun = np.hstack((symmetry_fun_r(atoms),symmetry_fun_a_values))
    #print(sym_fun)

    return sym_fun




tic = time.time()
'''----------------------------------------------------------------------------- working
path = './data_set_TiO2'
file_list = sorted(os.listdir(path))
f123 = open(os.path.join('./symmetry_functions','error.txt'),'w+')
for i,file in enumerate(file_list):
    if i==120:# and i<110:
      try:
          print('structure no =',i+1)
      
          datapoints_list = []
          _,_,datapoints_list=reader.xsf_reader(file)

          symm_fun = symmetry_function(datapoints_list)
          file_name = file[:-3]+'txt'
          f = open(os.path.join('./symmetry_functions','%s') %file_name,"w+") #os.path.join('./symmetry_functions',
          s = np.matrix(symm_fun)
          np.savetxt(os.path.join('./symmetry_functions','%s') %file_name, s)
      except:
        print('error',i+1)
        f123.write('error'+ str(i+1)+'\n')
        pass
          #print(np.loadtxt(os.path.join('./symmetry_functions','%s') %file_name))


toc = time.time()
print('Time taken =',str((toc-tic)) + 'sec')

#----------------------------------------------------------------------------'''


path = './data_set_TiO2'
file_list_required = sorted(os.listdir(path))
print(len(file_list_required))
file_list_required2 = [x[:-4] for x in file_list_required]
path = './symmetry_functions_all'
file_list_current = sorted(os.listdir(path))
print(len(file_list_current))
file_list_current2 = [x[:-4] for x in file_list_current]
missing = set(file_list_required2).difference(file_list_current2)
missing_i = (sorted([int(x[-4:]) for x in missing]))
final_xsf_list = sorted([x+'.xsf' for x in missing])
print(final_xsf_list)


path = './data_set_TiO2+outlier'
file_list = sorted(os.listdir(path))
f123 = open(os.path.join('./symmetry_functions_demo','error.txt'),'w+')
for i,file in enumerate(file_list):
    if i>8700 and i<9000:
      try:
          print('structure no =',file[-8:-4])
      
          datapoints_list = []
          _,_,datapoints_list=reader.xsf_reader(file)

          symm_fun = symmetry_function(datapoints_list)
          file_name = file[:-3]+'txt'
          f = open(os.path.join('./symmetry_functions_demo','%s') %file_name,"w+") #os.path.join('./symmetry_functions',
          print(file,file_name)
          s = np.matrix(symm_fun)
          np.savetxt(os.path.join('./symmetry_functions_demo','%s') %file_name, s)
      except:
        print('error',file[-8:-4])
        f123.write('error'+ str(i+1)+'\n')
        pass
      
exit(0)

path = './data_set_TiO2+outlier'
file_list = sorted(os.listdir(path))
f123 = open(os.path.join('./symmetry_functions_demo','error.txt'),'w+')
for i,file in enumerate(file_list):
    if  i==6762:
      try:
          print('structure no =',file[-8:-4])
      
          datapoints_list = []
          _,_,datapoints_list=reader.xsf_reader(file)

          symm_fun = symmetry_function(datapoints_list)
          file_name = file[:-3]+'txt'
          f = open(os.path.join('./symmetry_functions_demo','%s') %file_name,"w+") #os.path.join('./symmetry_functions',
          print(file,file_name)
          s = np.matrix(symm_fun)
          np.savetxt(os.path.join('./symmetry_functions_demo','%s') %file_name, s)
      except:
        print('error',file[-8:-4])
        f123.write('error'+ str(i+1)+'\n')
        pass
          #print(np.loadtxt(os.path.join('./symmetry_functions','%s') %file_name))


toc = time.time()
print('Time taken =',str((toc-tic)) + 'sec')