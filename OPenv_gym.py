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
from render import blockmodel


class environment(gym.Env):
    
    def __init__(self,inputfile,gamma, rendermode='off'):
        
        self.rendermode=rendermode
        self.inputdata=pd.read_excel(inputfile)
        self.data=self.inputdata
        self.actionslist = list()
        self.turnore=0     
        self.discountedmined=0
        self.turncounter=1
        self.i=-1
        self.j=-1
        self.terminal=False
        self.gamma=gamma
        self.Imin=self.data._I.min()
        self.Imax=self.data._I.max()
        self.Jmin=self.data._J.min()
        self.Jmax=self.data._J.max()
        self.RLmin=self.data.RL.min()
        self.RLmax=self.data.RL.max()
        self.Ilen=self.Imax-self.Imin+1
        self.Jlen=self.Jmax-self.Jmin+1
        self.RLlen=self.RLmax-self.RLmin+1 #RL counts up as depth increases
        
        self.orebody=np.array([self.Ilen,self.Jlen,self.RLlen])
        self.idxbody=np.array([self.Ilen,self.Jlen,self.RLlen])
        self.block_dic={}
        self.block_dic_init={}
        self.dep_dic={}
    
        self.RL=self.RLlen-1
        self.channels = 2
        self.geo_array= np.zeros([self.Ilen, self.Jlen, self.RLlen, self.channels], dtype=float)
        #self.state_size = self.geo_array.shape
        self.flatlen=self.Ilen*self.Jlen*self.RLlen*self.channels
        self.mined=-1
        
      # normalising input space
        
        for i in self.data.index:
            self.geo_array[self.data._I[i]-1,self.data._J[i]-1,self.data.RL[i]-1,0]=self.data.H2O[i]
            self.geo_array[self.data._I[i]-1,self.data._J[i]-1,self.data.RL[i]-1,1]=self.data.Tonnes[i]
            #state space (mined/notmined)
            #self.geo_array[self.data._I[i]-1,self.data._J[i]-1,self.data.RL[i]-1,2]=1
            
        scaler=MinMaxScaler()
        H2O_init=self.geo_array[:,:,:,0]
        Tonnes_init=self.geo_array[:,:,:,1]
        #State_init=self.geo_array[:,:,:,2]
        
        H2O_reshaped=H2O_init.reshape([-1,1])
        Tonnes_reshaped=Tonnes_init.reshape([-1,1])
        #State_reshaped=State_init.reshape([-1,1])
        
        H2O_scaled=scaler.fit_transform(H2O_reshaped)
        Tonnes_scaled=scaler.fit_transform(Tonnes_reshaped)
        
        a=H2O_scaled.reshape([self.Ilen, self.Jlen, self.RLlen,1])
        b=Tonnes_scaled.reshape([self.Ilen, self.Jlen, self.RLlen,1])
        #c=State_reshaped.reshape([1,self.Ilen, self.Jlen, self.RLlen,1])
               
        self.norm=np.append(a, b, axis=3)
        #self.norm=np.append(self.norm,c, axis=4)
        #.reshape(1,self.Imax+1-self.Imin, self.Jmax+1-self.Jmin, self.RLmax+1-self.RLmin, self.channels)
        #self.norm=normalize(np.reshape(self.geo_array,((1,self.Imax+1-self.Imin, self.Jmax+1-self.Jmin, self.RLmax+1-self.RLmin, self.channels))),4)
        self.ob_sample=deepcopy(self.norm)
        self.construct_dep_dic()
        #construct_dependencies blocks with padding
        self.construct_block_dic()
        self.block_dic=deepcopy(self.block_dic_init)
        self.render_update = self.geo_array[:,:,:,0]
        self.turns=round(len(self.dep_dic)*0.5,0)
        
        
        super(environment, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete((self.Ilen)*(self.Jlen))
        # Example for using image as input:
        self.observation_space = spaces.Box(low=-1, high=1,
                                        shape=(self.flatlen,), dtype=np.float64)
        
       
    
    def construct_block_dic(self):
       
        for i in range(-1,self.Ilen+1):
            for j in range(-1,self.Jlen+1):
                for k in range(self.RLlen):
                    
                    block=str(i)+str('_')+str(j)+str('_')+str(k)
                    
                    if (i>=0 & i<self.Ilen)&(j>=0 & j<self.Jlen)&(k>=0 & k<self.RLlen): 
                        self.block_dic_init["%s"% block]=0 # not mined yet
                    else:
                        self.block_dic_init["%s"% block]=1 # mined or doesnt exist
                        
    
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
               minablelogic[d]=0
               
            else: #if not surface then check dependencies
               minablelogic[d]=self.block_dic["%s"% depstr]
        
        notMinable=int(np.prod(minablelogic))
        
        if notMinable == 0:
            isMinable=1
        else:
            isMinable=0
            
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
            self.terminal =True
        
        flatob=np.ndarray.flatten(self.ob_sample)
                    
        return flatob, self.turnore, self.terminal, info    
    
                 
    def evaluate(self, selected_block, isMinable):
        
        if isMinable==1:             #penalising repetetive useless actions
            
            self.turnore=-1#/(self.gamma**(self.turncounter))

            
        else:
            
            H2O=self.geo_array[self.i,self.j,self.RL,0]
            Tonnes=self.geo_array[self.i, self.j,self.RL,1] 

            self.turnore=(H2O*Tonnes)
    
        self.discountedmined+=self.turnore*self.gamma**(self.turncounter)
        
    def update(self, selected_block):
    
        self.block_dic["%s"% selected_block]=1 #set to one (mined) for dependency logic multiplication
        
    
    def reset(self):
        
        self.block_dic=deepcopy(self.block_dic_init)
        self.ob_sample=deepcopy(self.norm)
        self.turnore=0
        self.discountedmined=0
        self.turncounter=0
        self.terminal=False
        self.i=-1
        self.j=-1
        self.RL=-1
        self.actionslist=list()
        self.render_update=deepcopy(self.geo_array[:,:,:,0])
        
        
        return np.ndarray.flatten(self.ob_sample)
                    
    def render(self, mode):      
        
        if mode=='on':
             
             self.render_update[self.i, self.j, self.RL]=0
             bm=blockmodel(self.render_update)
             bm.plot()
            
        pass
    
        
        
    #def close(self):
        