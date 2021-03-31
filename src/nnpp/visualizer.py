"""
====================================================================================================
Visualizer module
----------------------------------------------------------------------------------------------------
Used to visualize the position of atoms at a particular timestep in the MD simulation using 
matplotlib
====================================================================================================
"""
# ==================================================================================================
# imports
# --------------------------------------------------------------------------------------------------
import os 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import reader
# ==================================================================================================

path = './dataset_TiO2'
file_list = sorted(os.listdir(path))
file = file_list[1000] #choosing a random file to visualize
datapoints_list = []
_,_,datapoints_list=reader.xsf_reader(file)

def visualize(datapoints_list):
     """
     Shows the atoms in a 3d plot using the data given to it in form of list of lists.
     Arguments:
     datapoints_list -- list of lists containing atomtype and the atomic positions(x,y,z)
     Returns :
     none -- no return value.only shows the plot
     """
     fig = plt.figure(figsize = (10,7))
     ax = plt.axes(projection = '3d')
     datapoints = [(x,y,z) for (_,x,y,z) in datapoints_list]
     colors = ['red' if element == 'Ti' else 'blue' for(element,_,_,_) in datapoints_list ] #to change color according to atomtype
     #size   = [1500 if element =='Ti' else 500 for(element,_,_,_) in datapoints_list]
     labels = ['Ti' if element == 'Ti'else 'O' for(element,_,_,_) in datapoints_list]
     for datapoint,color,label_ in zip(datapoints,colors,labels):
          x,y,z = datapoint
          ax.scatter3D(x,y,z,c=color,s=1500)

     ax.set_xlabel('X')
     ax.set_ylabel('Y')
     ax.set_zlabel('Z')
     ax.set_title('Atomic Visualization')
     #ax.legend()
     plt.show()


if __name__ == '__main__': 
     visualize(datapoints_list)





