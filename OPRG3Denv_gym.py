# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 12:04:45 2020

@author: Tim Pelech
"""
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
import gym
from gym import spaces
from render import renderbm
from createmodel import automodel

class environment(gym.Env):
    
    def __init__(self, x,y,z ,gamma, rendermode='off'):
        
        self.rendermode=rendermode

        #self.data=self.inputdata
        self.actionslist = list()
        self.turnore=0     
        self.discountedmined=0
        self.turncounter=1
        self.i=-1
        self.j=-1
        self.terminal=False
        self.gamma=gamma
        self.Imin=0
        self.Imax=x
        self.Jmin=0
        self.Jmax=y
        self.RLmin=0
        self.RLmax=z
        self.Ilen=self.Imax-self.Imin
        self.Jlen=self.Jmax-self.Jmin
        self.RLlen=self.RLmax-self.RLmin #RL counts up as depth increases
        
        #self.orebody=np.array([self.Ilen,self.Jlen,self.RLlen])
        #self.idxbody=np.array([self.Ilen,self.Jlen,self.RLlen])
        self.block_dic={}
        self.block_dic_init={}
        self.dep_dic={}
    
        #self.RL=self.RLlen-1
        self.channels = 3
        #self.geo_array= np.zeros([self.Ilen, self.Jlen, self.RLlen, self.channels], dtype=float)
        #self.state_size = self.geo_array.shape
        self.flatlen=self.Ilen*self.Jlen*self.RLlen*self.channels
        self.mined=-1
        
        self.callnumber=1
        
        self.automodel=automodel()
             
        self.build()
        self.turns=round(len(self.dep_dic)*0.5,0)
        
        
       #super(environment, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete((self.Ilen)*(self.Jlen))#Box(low=0, high=1,
                                        #shape=((self.Ilen)*(self.Jlen),), dtype=np.float64)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=-1, high=1,
                                        shape=(self.Ilen, self.Jlen, self.RLlen,self.channels), dtype=np.float64)

    def build(self):
                
        self.geo_array=self.automodel.buildmodel(self.Ilen,self.Jlen,self.RLlen)
        
        scaler=MinMaxScaler()
        H2O_init=self.geo_array[:,:,:,0]
        Tonnes_init=self.geo_array[:,:,:,1]
        State_init=self.geo_array[:,:,:,2]
        
        H2O_reshaped=H2O_init.reshape([-1,1])
        Tonnes_reshaped=Tonnes_init.reshape([-1,1])
        State_reshaped=State_init.reshape([-1,1])
        
        H2O_scaled=scaler.fit_transform(H2O_reshaped)
        Tonnes_scaled=scaler.fit_transform(Tonnes_reshaped)
        
        a=H2O_scaled.reshape([self.Ilen, self.Jlen, self.RLlen,1])
        b=Tonnes_scaled.reshape([self.Ilen, self.Jlen, self.RLlen,1])
        c=State_reshaped.reshape([self.Ilen, self.Jlen, self.RLlen,1])
               
        self.norm=np.append(a, b, axis=3)
        self.norm=np.append(self.norm,c, axis=3)
        #.reshape(1,self.Imax+1-self.Imin, self.Jmax+1-self.Jmin, self.RLmax+1-self.RLmin, self.channels)
        #self.norm=normalize(np.reshape(self.geo_array,((1,self.Imax+1-self.Imin, self.Jmax+1-self.Jmin, self.RLmax+1-self.RLmin, self.channels))),4)
        self.ob_sample=deepcopy(self.norm)
        self.construct_dep_dic()
        #construct_dependencies blocks with padding
        self.construct_block_dic()
        self.block_dic=deepcopy(self.block_dic_init)
        self.render_update = self.geo_array[:,:,:,0]
               
    
    def construct_block_dic(self):
       
        for i in range(-1,self.Ilen+1):
            for j in range(-1,self.Jlen+1):
                for k in range(self.RLlen):
                    
                    block=str(i)+str('_')+str(j)+str('_')+str(k)
                    self.block_dic_init["%s"% block]=0 
                    #if (i>=0 & i<self.Ilen)&(j>=0 & j<self.Jlen)&(k>=0 & k<self.RLlen): 
                        #self.block_dic_init["%s"% block]=0 # not mined yet
                    #else:
                    #    self.block_dic_init["%s"% block]=1 # mined or doesnt exist
                        
    
    def construct_dep_dic(self):    
    #construct_dependencies
        
        for i in range(self.Ilen):
            for j in range(self.Jlen):
                for k in range(self.RLlen):
                    
                    block=str(i)+str('_')+str(j)+str('_')+str(k)
                    if k==0: #if block is surface layer, then no dependency exists
                        
                        dep=list(['','','','','','','','',''])
                        self.dep_dic["%s"% block]=dep
                        
                    else:
                        dep0=str(i-1)+str('_')+str(j+1)+str('_')+str(k-1)
                        dep1=str(i)+str('_')+str(j+1)+str('_')+str(k-1)
                        dep2=str(i+1)+str('_')+str(j+1)+str('_')+str(k-1)
                        dep3=str(i-1)+str('_')+str(j)+str('_')+str(k-1)
                        dep4=str(i)+str('_')+str(j)+str('_')+str(k-1)
                        dep5=str(i+1)+str('_')+str(j)+str('_')+str(k-1)
                        dep6=str(i-1)+str('_')+str(j-1)+str('_')+str(k-1)
                        dep7=str(i)+str('_')+str(j-1)+str('_')+str(k-1)
                        dep8=str(i+1)+str('_')+str(j-1)+str('_')+str(k-1)
                        
                        dep=list([dep0,dep1,dep2,dep3,dep4,dep5,dep6,dep7,dep8])
                        self.dep_dic["%s"% block]=dep
               
    
    def actcoords(self, action):
        #map coords
        q=np.zeros((self.Ilen)*(self.Jlen))
        q[action]=1
        
        q2=q.reshape(self.Ilen,self.Jlen)
        action_coords=np.argwhere(q2.max()==q2)[0]
        
        #mapping q values to action coordinates
        self.i=action_coords[0]#+1
        self.j=action_coords[1]#+1    
    
    def select_block(self):
    
        for k in range(self.RLlen): #iterate through orebody at action location to find highest unmined block (reversed -top to bottom)
            check_block=str(self.i)+str('_')+str(self.j)+str('_')+str(k)
            
            if self.block_dic["%s"% check_block]==0:
                selected_block = check_block
                self.RL = k
                break
            
            if k==self.RLlen-1:
                selected_block = check_block            
            
        return selected_block #return string name of selected block to mine
        
    
    def isMinable(self, selected_block):
        
        deplist = self.dep_dic["%s"% selected_block]
        minablelogic=np.zeros(len(deplist))
        
        for d in range(len(deplist)):
            depstr=deplist[d]
            
            if depstr=='':
               minablelogic[d]=1
               
            else: #if not surface then check dependencies
               minablelogic[d]=self.block_dic["%s"% depstr]
        
        isMinable=int(np.prod(minablelogic))
                   
        return isMinable
      
    
    def step(self, action):        
        info={}
        self.actcoords(action)
        selected_block=self.select_block()
        minable=self.isMinable(selected_block)
        
        if (self.turncounter<self.turns):
            
            self.evaluate(selected_block, minable)
            self.update(selected_block)
            self.turncounter+=1
            self.render(self.rendermode)
        else:
            self.evaluate(selected_block, minable)
            self.update(selected_block)
            self.turncounter+=1
            self.render(self.rendermode)
            self.terminal =True
        
        #arr=np.ndarray.flatten(self.ob_sample) #used for MLP policy
        #out=arr.reshape([1,len(arr)])
                    
        return self.ob_sample, self.turnore, self.terminal, info    
    
                 
    def evaluate(self, selected_block, isMinable):
        
        if isMinable==0:             #penalising repetetive useless actions
            
            self.turnore=-1#/(self.gamma**(self.turncounter))

            
        else:
            
            H2O=self.geo_array[self.i,self.j,self.RL,0]
            Tonnes=self.geo_array[self.i, self.j,self.RL,1] 

            self.turnore=(H2O*Tonnes)
    
        self.discountedmined+=self.turnore*self.gamma**(self.turncounter)
        
    def update(self, selected_block):
    
        self.block_dic["%s"% selected_block]=1 #set to one (mined) for dependency logic multiplication
        self.ob_sample[self.i,self.j,self.RL,2]=1 #set to one (mined) for agent observation

    
    def reset(self):
        
        #self.block_dic=deepcopy(self.block_dic_init)
        #self.ob_sample=deepcopy(self.norm)
        self.build()
        self.turnore=0
        self.discountedmined=0
        self.turncounter=0
        self.terminal=False
        self.i=-1
        self.j=-1
        self.RL=-1
        self.actionslist=list()
        #self.render_update=deepcopy(self.geo_array[:,:,:,0])
        
        #arr=np.ndarray.flatten(self.ob_sample) #used for MLP policy
        #out=arr.reshape([1,len(arr)])
        return self.ob_sample
                    
    def render(self, mode):      
        
        if mode=='on':
             
             self.render_update[self.i, self.j, self.RL]=0
             bm=renderbm(self.render_update)
             bm.plot()
            
        pass
   
        
        
    #def close(self):
        
        
        
        
