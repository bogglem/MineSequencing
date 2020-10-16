# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 08:30:55 2020

@author: Tim Pelech
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class blockmodel():
     
    def __init__(self, blockmodel):
        
        #inputfile="BM_easy10x10x8.xlsx"
#        self.inputdata=pd.read_excel(inputfile)
#        self.data=self.inputdata
#        self.Imin=self.data._I.min()
#        self.Imax=self.data._I.max()
#        self.Jmin=self.data._J.min()
#        self.Jmax=self.data._J.max()
#        self.RLmin=self.data.RL.min()
#        self.RLmax=self.data.RL.max()
#        self.Ilen=self.Imax-self.Imin+1
#        self.Jlen=self.Jmax-self.Jmin+1
#        self.RLlen=self.RLmax-self.RLmin+1 #RL counts up as depth increases
#        self.channels = 2
        
#        self.geo_array= np.zeros([self.Ilen, self.Jlen, self.RLlen, self.channels], dtype=float)
        
#        for i in self.data.index:
#            self.geo_array[self.data._I[i]-1,self.data._J[i]-1,self.data.RL[i]-1,0]=self.data.H2O[i]
#            self.geo_array[self.data._I[i]-1,self.data._J[i]-1,self.data.RL[i]-1,1]=self.data.Tonnes[i]
               
#        self.bm=self.geo_array[:,:,:,0]
        
        self.bm = blockmodel
        
        self.Imax = blockmodel.shape[0]+1
        self.Jmax = blockmodel.shape[1]+1
        self.RLmax = blockmodel.shape[2]+1
        
    def normalize(self, arr):
        arr_min = np.min(arr)
        return (arr-arr_min)/(np.max(arr)-arr_min)
    
    def make_ax(self, grid=False):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.grid(grid)
        return ax
    
    
    def explode(self, arr):
        shape_arr = np.array(arr.shape)
        size = shape_arr*2 - 1
        exploded = np.zeros(size, dtype=arr.dtype)
        exploded[::2, ::2, ::2] = self.bm
        return exploded
    
    def expand_coordinates(self, indices):
        x, y, z = indices
        x[1::2, :, :] += 1
        y[:, 1::2, :] += 1
        z[:, :, 1::2] += 1
        return x, y, z
    
    def plot(self, angle=300):
        
        exparr=self.explode(self.bm)
        normarr = self.normalize(exparr)     
        facecolours = cm.viridis(normarr)
        ceilarr=np.ceil(normarr)
        facecolours[:,:,:,-1] = ceilarr
                
        filled = facecolours[:,:,:,-1] != 0
        x, y, z = self.expand_coordinates(np.indices(np.array(filled.shape) + 1))
    
        fig = plt.figure(figsize=(30/2.54, 30/2.54))
        ax = fig.gca(projection='3d')
        ax.view_init(30, angle)
        ax.set_xlim(right=self.Imax*2)
        ax.set_ylim(top=self.Jmax*2)
        ax.set_zlim(top=self.RLmax*2)
        ax.invert_zaxis()
        ax.voxels(x, y, z, filled, facecolors=facecolours, shade=False)
        plt.show()