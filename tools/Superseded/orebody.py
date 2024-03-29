# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 12:04:45 2020

@author: Tim Pelech
"""
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler

class orebody():

    def __init__(self,inputfile,gamma):
        
        self.inputdata=pd.read_excel(inputfile)
        self.data=self.inputdata
        self.actionslist = list()
        self.turnore=0     
        self.discountedmined=0
        self.turncounter=0
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
        self.dep_dic={}
        
        self.action_space=np.zeros((self.Ilen)*(self.Jlen))
        
        self.RL=self.RLlen-1
        self.channels = 2
        self.geo_array= np.zeros([self.Ilen, self.Jlen, self.RLlen, self.channels], dtype=float)
        self.state_size = self.geo_array.shape

        self.turns=(self.RLlen*self.Ilen*self.Jlen)
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
        
        a=H2O_scaled.reshape([1,self.Ilen, self.Jlen, self.RLlen,1])
        b=Tonnes_scaled.reshape([1,self.Ilen, self.Jlen, self.RLlen,1])
        #c=State_reshaped.reshape([1,self.Ilen, self.Jlen, self.RLlen,1])
               
        self.norm=np.append(a, b, axis=4)
        #self.norm=np.append(self.norm,c, axis=4)
        #.reshape(1,self.Imax+1-self.Imin, self.Jmax+1-self.Jmin, self.RLmax+1-self.RLmin, self.channels)
        #self.norm=normalize(np.reshape(self.geo_array,((1,self.Imax+1-self.Imin, self.Jmax+1-self.Jmin, self.RLmax+1-self.RLmin, self.channels))),4)
        self.ob_sample=deepcopy(self.norm)
        
        #construct_dependencies blocks with padding
       
        for i in range(-1,self.Ilen):
            for j in range(-1,self.Jlen):
                for k in range(self.RLlen):
                    
                    block=str(i)+str('_')+str(j)+str('_')+str(k)
                    self.block_dic["%s"% block]=1

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
    
        for k in reversed(range(self.RLlen)): #iterate through orebody at action location to find highest unmined block (reversed -top to bottom)
            check_block=str(self.i)+str('_')+str(self.j)+str('_')+str(k)
            
            if self.block_dic["%s"% check_block]==1:
                selected_block = check_block
                self.RL = k
                break
            
            if k==0:
                selected_block = check_block            
            
        return selected_block #return string name of selected block to mine
        
    
    def isMinable(self, selected_block):
        
        available = self.block_dic["%s"% selected_block] #identifies if block is already mined.
        deplist = self.dep_dic["%s"% selected_block]
        minablelogic=np.zeros(len(deplist))
        
        for d in range(len(deplist)):
            depstr=deplist[d]
            
            if depstr=='':
               isMinable=1
               break
            else: #if not surface then check dependencies
               minablelogic[d]=self.block_dic["%s"% depstr]
               isMinable=available*np.prod(minablelogic)
            
        return isMinable
      

    def step(self, action):        

        self.actcoords(action)
        selected_block=self.select_block()
        minable=self.isMinable(selected_block)
        
        if (self.turncounter<self.turns):
            
            self.evaluate(selected_block, minable)
            self.update(selected_block)
            self.turncounter+=1
            
        else: 
            self.terminal =True
                            
        return self.ob_sample, self.turnore, self.terminal, #info    
        
                 
    def evaluate(self, selected_block, isMinable):
        
        if isMinable==0:             #penalising repetetive useless actions

            H2O=-1
            Tonnes=1
            #State=-1
            
        else:
            
            H2O=self.geo_array[self.i,self.j,self.RL,0]
            Tonnes=self.geo_array[self.i, self.j,self.RL,1] 
            #State=self.mined

        self.turnore=(H2O*Tonnes)
        self.discountedmined+=self.turnore*self.gamma**(self.turncounter)
        
    def update(self, selected_block):

        self.ob_sample[0,self.i,self.j,self.RL,:]=self.mined #update State channel to mined "-1"
        self.block_dic["%s"% selected_block]=0 #set to zero for logic multiplication
        

    def reset(self):
        
        self.ob_sample=deepcopy(self.norm)
        self.turnore=0
        self.discountedmined=0
        self.turncounter=0
        self.terminal=False
        self.i=-1
        self.j=-1
        self.RL=-1
        self.actionslist=list()
        
        return self.ob_sample            
                    
                    
        