
import os 
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import reader


path = './data_set_TiO2_small'
file_list = sorted(os.listdir(path))
file = file_list[5]

_,_,datapoints_list=reader.xsf_reader(file)


print(datapoints_list)
fig = plt.figure(figsize = (10,7))
ax = plt.axes(projection = '3d')
datapoints = [(x,y,z) for (_,x,y,z) in datapoints_list]
colors = ['red' if element == 'Ti' else 'blue' for(element,_,_,_) in datapoints_list ]
for datapoint,color in zip(datapoints,colors):
     x,y,z = datapoint
     ax.scatter3D(x,y,z,c=color,s=500)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


class Atom:
    def __init__(self,atomtype,x,y,z):
        self.atomtype = atomtype
        self.position = (x,y,z)

    def __repr__(self):
        return f"Atom : {self.atomtype} at {self.position}"

    
print(type([Atom(atomtype,x,y,z) for (atomtype,x,y,z) in datapoints_list][0]))