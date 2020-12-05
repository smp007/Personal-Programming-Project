import os
import numpy as np 
import time
import matplotlib.pyplot as plt 



path = './dataset_symm'
file_list = sorted(os.listdir(path))
#print((file_list))

number_of_atomss = []
energy_list = []

for file in file_list:

    with open(path+'/'+'%s'%(file)) as f:
        for i,line in enumerate(f):

            if i==0:  #energy line strips at =
                energy_list.append(float((line.strip().split('='))[1][:-3]))            

            if i==8:  #to get no: of atoms...goes to that line strips and splits first element
                no_of_atoms = line.strip().split(' ')
                n=int(no_of_atoms[0])
                number_of_atomss.append(n)

#print(collections.Counter(number_of_atomss))
#print(energy_list)


tic = time.time()
def test_train_split(filelist,energylist,split):
    '''Creates an empty array for test split and pops each element from total
dataset and append it to the test set simultaneously'''
    n_total_set = len(filelist)
    n_train_set = split/100 * n_total_set
    train_set = []   
    test_set = filelist
    train_energy = [] 
    test_energy = energylist
    while len(train_set) < n_train_set :
        index = np.random.randint(0,len(test_set))     #randrange(len(train_set))
        #print(index)
        train_set.append(test_set.pop(index))
        train_energy.append(test_energy.pop(index))
    return train_set,test_set,(train_energy),(test_energy)
np.random.seed(4)
#print(test_train_split(file_list,energy_list,80))

a,b,c,d = test_train_split(file_list,energy_list,80)
#print(len(c))
toc = time.time()
print('Time taken =',str((toc-tic)) + 'sec')

test_xy = ([(np.loadtxt(os.path.join('./symmetry_functions','%s') %(x[:-3]+'txt')),y)for x,y in zip(b,d)]) #np.loadtxt(os.path.join('./symmetry_functions','%s') %(x[:-3]+'txt'))
train_xy = ([(np.loadtxt(os.path.join('./symmetry_functions','%s') %(x[:-3]+'txt')),y)for x,y in zip(a,c)])

for val in train_xy:
    #print(val)
    pass
print(a[0],b[0])
print(train_xy[0][0],test_xy[0][0])

plt.plot(d)
plt.grid('True')
plt.show()