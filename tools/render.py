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
        
        #input block model from environment.
        #adapted code from https://terbium.io/2017/12/matplotlib-3d/
        
        self.bm = blockmodel
        
        self.Imax = blockmodel.shape[0]+1
        self.Jmax = blockmodel.shape[1]+1
        self.RLmax = blockmodel.shape[2]+1
        
        self.exparr=self.explode(self.bm)
        shape=(self.exparr.shape[0],self.exparr.shape[1],self.exparr.shape[2],4)
        self.facecolours=np.zeros(shape)
        self.facecolours_x=np.zeros(shape)
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
        
        #scale=(RL+1)/(self.RLmax)
        self.facecolours[exploded_x,exploded_y,exploded_z] = [0.5,0.5,0.5,0]
        
        if exploded_z>0:
            self.facecolours[exploded_x,exploded_y,0:exploded_z-1] = [0.5,0.5,0.5,0] #hide previously mined blocks above
        
    def update_all_mined(self, ob_sample):
        
        Ilen=ob_sample.shape[0]
        Jlen=ob_sample.shape[1]
        RLlen=ob_sample.shape[2]
        
        for i in range(Ilen):
            for j in range(Jlen):
                for k in range(RLlen):
                    if ob_sample[i,j,k,1]==1:
                        #minedarr=explode(self.bm)
                        #scale=(k+1)/(RLlen+1)
                        exploded_x,exploded_y,exploded_z = self.translate_to_exploded(i,j,k)
                        
                        self.facecolours[exploded_x,exploded_y,exploded_z] = [0.5,0.5,0.5,0]
                        self.facecolours_x[exploded_x,exploded_y,exploded_z] = [0.5,0.5,0.5,0]
                        
                        if exploded_z>0:
                            self.facecolours[exploded_x,exploded_y,0:exploded_z-1] = [0.5,0.5,0.5,0] #hide previously mined blocks above
                            self.facecolours_x[exploded_x,exploded_y,0:exploded_z-1] = [0.5,0.5,0.5,0] 


    def initiate_plot(self, averagereward, transparency='on'):   

        self.exparr_x=self.explode(self.bm)
        self.exparr=np.where(self.exparr_x-averagereward<0, 0,self.exparr) #turns values below cutoff to 0 invisible waste
        normarr = self.normalize(self.exparr)     
        self.facecolours = cm.plasma(normarr) # facecolours [x,y,z, (r,g,b,a)]
        #ceilarr=np.ceil(normarr)
        #normarr[normarr<=0.05]=0 #hide voxels less than 0.05
        if transparency=='on':
            self.facecolours[:,:,:,-1] = normarr #matches transperancy to normarr intensity
            normarr_x = self.normalize(self.exparr_x)            
            self.facecolours_x = cm.plasma(normarr_x) # facecolours [x,y,z, (r,g,b,a)]
            #ceilarr=np.ceil(normarr)
            #normarr[normarr<=0.05]=0 #hide voxels less than 0.05
            
            self.facecolours_x[:,:,:,-1] = normarr #matches transperancy to normarr intensity     
            
            
            
        else:
            
            notransp=np.where(self.exparr_x-averagereward<0, 0,1)
            self.facecolours[:,:,:,-1] = notransp
            
            
            normarr_x = self.normalize(self.exparr_x)            
            self.facecolours_x = cm.plasma(normarr_x) # facecolours [x,y,z, (r,g,b,a)]
            #ceilarr=np.ceil(normarr)
            #normarr[normarr<=0.05]=0 #hide voxels less than 0.05

            self.facecolours_x[:,:,:,-1] = notransp #matches transperancy to 1 for ore 0 for waste        
           
        

        
    
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
        ax.voxels(x, y, z, self.filled, facecolors=self.facecolours, edgecolors='k', shade=False)
        ax.set(xlabel='X',ylabel='Y',zlabel='Z')

        plt.show()

        
    def plotx(self, xx,yy,zz):
        
        eqscale=max(self.Imax,self.Jmax,self.RLmax)                
        fig = plt.figure(figsize=(30/2.54, 30/2.54))
        ax = fig.gca(projection='3d')
        ax.set_xlim(right=eqscale*2)
        ax.set_ylim(top=eqscale*2)
        ax.set_zlim(top=eqscale*2)
        ax.set(xlabel='X',ylabel='Y',zlabel='Z')
        ax.invert_zaxis()
        
        if xx>0:
            self.filled = self.facecolours_x[:,:,:,-1]# != 0     #hide voxels not = 0
            angle=0


            filled, facecolours =  self.filled[(xx-2):xx,:,:], self.facecolours_x[(xx-2):xx,:,:,:]
            facecolours[:,:,:,3]=1
            x, y, z = self.expand_coordinates(np.indices(np.array(filled.shape) + 1))
        
            ax.view_init(0, angle)
            ax.voxels(x, y, z, filled, facecolors=facecolours, edgecolors='k', shade=False)

            plt.show()            
               
        elif yy>0:
            self.filled = self.facecolours_x[:,:,:,-1]# != 0     #hide voxels not = 0
            angle=90

            eqscale=max(self.Imax,self.Jmax,self.RLmax)
            filled, facecolours =  self.filled[:,(yy-2):yy,:], self.facecolours_x[:,(yy-2):yy,:,:]
            facecolours[:,:,:,3]=1
            x, y, z = self.expand_coordinates(np.indices(np.array(filled.shape) + 1))

            ax.view_init(0, angle)
            ax.voxels(x, y, z, filled, facecolors=facecolours, edgecolors='k', shade=False)

            plt.show()           
                        
        elif zz>0:
            self.filled = self.facecolours_x[:,:,:,-1]# != 0     #hide voxels not = 0
            angle=0

            eqscale=max(self.Imax,self.Jmax,self.RLmax)
            filled, facecolours =  self.filled[:,:,(zz-2):zz], self.facecolours_x[:,:,(zz-2):zz,:]
            facecolours[:,:,:,3]=1
            x, y, z = self.expand_coordinates(np.indices(np.array(filled.shape) + 1))

            ax.view_init(90, angle)
            ax.voxels(x, y, z, filled, facecolors=facecolours, edgecolors='k', shade=False)

            plt.show()             
                        

