# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 18:30:30 2020

@author: Tim Pelech
"""
import pandas as pd


# -*- coding: utf-8 -*-
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
#import tensorflow as tf
from keras import Model
from keras.models import Sequential, clone_model, load_model
from keras.layers import Dense, Conv3D, MaxPooling3D, Flatten, Dropout, Input
from keras.optimizers import Adam
from copy import deepcopy
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler
#from tensorflow import Print

gamma=0.95
LR_actor=0.00001
LR_critic=0.001
batch_size=64
EPSINIT=10.0
inputfile="Ore blocks_sandbox3x3v2.xlsx"
epsilon_min=0.01
memcap=10000
EPISODES = 200
dropout=0
test='no-expmod'

start=time.time()
end=start+11.5*60*60

mined=-1

class environment: 

    def __init__(self):
        
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
        self.RLlen=self.RLmax-self.RLmin+1
        self.action_space=np.zeros((self.Ilen)*(self.Jlen))
        self.actioncounter=np.zeros((self.Ilen)*(self.Jlen))
        
        self.RL=self.RLlen-1
        self.channels = 3
        self.geo_array= np.zeros([self.Ilen, self.Jlen, self.RLlen, self.channels], dtype=float)
        self.state_size = self.geo_array.shape
        self.actionlimit=np.ones([self.Ilen, self.Jlen])      
        self.turns=(self.RLlen*self.Ilen*self.Jlen)
        #self.epsilonmod=np.zeros(self.turns)
        
      # normalising input space
        
        for i in self.data.index:
            self.geo_array[self.data._I[i]-1,self.data._J[i]-1,self.data.RL[i]-1,0]=self.data.H2O[i]
            self.geo_array[self.data._I[i]-1,self.data._J[i]-1,self.data.RL[i]-1,1]=self.data.Tonnes[i]
            #state space (mined/notmined)
            self.geo_array[self.data._I[i]-1,self.data._J[i]-1,self.data.RL[i]-1,2]=1
            
        scaler=MinMaxScaler()
        H2O_init=self.geo_array[:,:,:,0]
        Tonnes_init=self.geo_array[:,:,:,1]
        State_init=self.geo_array[:,:,:,2]
        
        H2O_reshaped=H2O_init.reshape([-1,1])
        Tonnes_reshaped=Tonnes_init.reshape([-1,1])
        State_reshaped=State_init.reshape([-1,1])
        
        H2O_scaled=scaler.fit_transform(H2O_reshaped)
        Tonnes_scaled=scaler.fit_transform(Tonnes_reshaped)
        
        a=H2O_scaled.reshape([1,self.Ilen, self.Jlen, self.RLlen,1])
        b=Tonnes_scaled.reshape([1,self.Ilen, self.Jlen, self.RLlen,1])
        c=State_reshaped.reshape([1,self.Ilen, self.Jlen, self.RLlen,1])
               
        self.norm=np.append(a, b, axis=4)
        self.norm=np.append(self.norm,c, axis=4)
        #.reshape(1,self.Imax+1-self.Imin, self.Jmax+1-self.Jmin, self.RLmax+1-self.RLmin, self.channels)
        #self.norm=normalize(np.reshape(self.geo_array,((1,self.Imax+1-self.Imin, self.Jmax+1-self.Jmin, self.RLmax+1-self.RLmin, self.channels))),4)
        self.ob_sample=deepcopy(self.norm)


    def actcoords(self, action):
        #map coords
        q=np.zeros((self.Ilen)*(self.Jlen))
        q[action]=1
        
        q2=q.reshape(self.Ilen,self.Jlen)
        action_coords=np.argwhere(q2.max()==q2)[0]
        
        #mapping q values to action coordinates
        
        self.i=action_coords[0]#+1
        self.j=action_coords[1]#+1
        
    #def possible_actions(self):
      
        
    def step(self, action):        

        self.actcoords(action)
        
        if (self.turncounter<self.turns): #& (data2.empty!=True): 
            
            #self.actionslist.append(action)
            self.evaluate()
            self.update()     
            #self.actioncounter[action]+=1
            #if max(self.actioncounter)>self.RLlen:
            #self.epsilonmod[self.turncounter]=round(max((self.actioncounter))/(self.RLlen),ndigits=2)
            self.turncounter+=1
            
        else: 
            self.terminal =True
                            
        return self.ob_sample, self.turnore, self.terminal, #info
    
    def evaluate(self):
        
        H2O=0
        Tonnes=0
        State=1
        for RLidx in reversed(range(self.RLlen)):
            
            if self.ob_sample[0,self.i,self.j,RLidx,2]!=mined: #if State unmined
                self.RL=RLidx
                H2O=self.geo_array[self.i,self.j,self.RL,0]
                Tonnes=self.geo_array[self.i, self.j,self.RL,1]
                #State=self.ob_sample[0,self.i, self.j,self.RL,2]              
                break
        
        if self.ob_sample[0,self.i,self.j,RLidx,2]==mined: #if all mined in i,j column
            #self.terminal=True
            self.actionlimit[self.i,self.j]=0
            #penalising repetetive useless actions
            H2O=1
            Tonnes=1
            State=mined
            
            if sum(sum(self.actionlimit))==0:
                self.termimal=True
           
        self.turnore=(H2O*Tonnes*State)
        self.discountedmined+=self.turnore*self.gamma**(self.turncounter)
        
    def update(self):
    
        #self.ob_sample[0,self.i,self.j,self.RL,0]=-1
        #self.ob_sample[0,self.i,self.j,self.RL,1]=-1
        self.ob_sample[0,self.i,self.j,self.RL,2]=mined #update State channel to mined "-1"
      
        

    def reset(self):
        
        self.ob_sample=deepcopy(self.norm)

        self.actionlimit=np.ones([self.Ilen, self.Jlen]) 
        
        self.turnore=0
        self.discountedmined=0
        self.turncounter=0
        self.terminal=False
        self.i=-1
        self.j=-1
        self.actionslist=list()
        self.actioncounter=np.zeros((self.Ilen)*(self.Jlen))
        
        return self.ob_sample
