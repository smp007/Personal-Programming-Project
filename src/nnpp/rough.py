import reader
import os
import collections
import matplotlib.pyplot as plt
import numpy as np
path = './data_set_TiO2'
file_list = sorted(os.listdir(path))
#file = file_list[1]
#print(file_list)
number_of_atomss = []
energy_list = []

for file in file_list:

    with open(path+'/'+'%s'%(file)) as f:
        for i,line in enumerate(f):

            if i==0:  #energy line strips at =
                energy_list.append((line.strip().split('='))[1][:-3])            

            if i==8:  #to get no: of atoms...goes to that line strips and splits first element
                no_of_atoms = line.strip().split(' ')
                n=int(no_of_atoms[0])
                number_of_atomss.append(n)

print(collections.Counter(number_of_atomss))
print(energy_list)
x = np.linspace(1,50)
plt.plot(energy_list[:50],'o')
plt.show()


#{24: 2895, 6: 1827, 22: 1280, 23: 1260, 46: 480, 47: 28, 94: 24, 95: 21}