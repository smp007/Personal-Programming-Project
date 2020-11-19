
import os 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import reader


path = './data_set_TiO2_small'
file_list = sorted(os.listdir(path))
file = file_list[0]
#print(file)
datapoints_list = []
_,_,datapoints_list=reader.xsf_reader(file)
                                                    #print(reader.xsf_reader(file))
# x_value = [x for (_,x,_,_) in datapoints_list]
# y_value = [y for (_,_,y,_) in datapoints_list]
# z_value = [z for (_,_,_,z) in datapoints_list]
# #print(x_value,y_value,z_value)
# x_values,y_values,z_values = [],[],[]

# x_values.append(x_value*2)
# #y_values.append(y_value*i)
# #z_values.append(z_value*i)

# print(x_values)

#print(datapoints_list)
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
#plt.show()

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(x_values,y_values,z_values)
# plt.show()





