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

class renderbm():
     
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
        
        self.exparr=self.explode(self.bm)
        shape=(self.exparr.shape[0],self.exparr.shape[1],self.exparr.shape[2],4)
        self.facecolours=np.zeros(shape)
        self.filled = self.facecolours[:,:,:,-1] != 0
        
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
    
    def expand_coordinates(self, indices): #remove gaps between voxels
        x, y, z = indices
        x[1::2, :, :] += 1
        y[:, 1::2, :] += 1
        z[:, :, 1::2] += 1
        return x, y, z
    
    
    def translate_to_exploded(self,x,y,z):
        
        exploded_x=x*2
        exploded_y=y*2
        exploded_z=z*2
        
        return exploded_x,exploded_y,exploded_z
        
    def update_mined(self,i,j,RL):
        
        #minedarr=explode(self.bm)
        exploded_x,exploded_y,exploded_z = self.translate_to_exploded(i,j,RL)
        
        self.facecolours[exploded_x,exploded_y,exploded_z] = [0.5,0.5,0.5,0.5]
        
        if exploded_z>0:
            self.facecolours[exploded_x,exploded_y,0:exploded_z-1] = [0.5,0.5,0.5,0] #hide previously mined blocks above
        
        

    def initiate_plot(self):   

        self.exparr=self.explode(self.bm)
        normarr = self.normalize(self.exparr)     
        self.facecolours = cm.plasma(normarr) # facecolours [x,y,z, (r,g,b,a)]
        ceilarr=np.ceil(normarr)
        normarr[normarr<=0.05]=0 #hide voxels less than 0.05
        
        self.facecolours[:,:,:,-1] = normarr #matches transperancy to normarr intensity
             
               
    
    def plot(self, angle=300):
        
        self.filled = self.facecolours[:,:,:,-1] != 0     #hide voxels not = 0
        eqscale=max(self.Imax,self.Jmax,self.RLmax)
        x, y, z = self.expand_coordinates(np.indices(np.array(self.filled.shape) + 1))
    
        fig = plt.figure(figsize=(30/2.54, 30/2.54))
        ax = fig.gca(projection='3d')
        ax.view_init(30, angle)
        ax.set_xlim(right=eqscale*2)
        ax.set_ylim(top=eqscale*2)
        ax.set_zlim(top=eqscale*2)
        ax.invert_zaxis()
        ax.voxels(x, y, z, self.filled, facecolors=self.facecolours, edgecolors='none', shade=False)
        plt.show()