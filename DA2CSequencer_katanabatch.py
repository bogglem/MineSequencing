#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 11:32:50 2020

@author: z3333990
"""

import pandas as pd

# -*- coding: utf-8 -*-
import sys
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

idx=int(sys.argv[1])

inputarray=pd.read_csv('A2C_job_input_array.csv')
 
LR_critic=inputarray.loc[idx].LR_critic
LR_actor=inputarray.loc[idx].LR_actor
batch_size=int(inputarray.loc[idx].batch_size)
memcap=int(inputarray.loc[idx].memcap)
inputfile=inputarray.loc[idx].inputfile
gamma=inputarray.loc[idx].gamma
dropout=float(inputarray.loc[idx].dropout)

#gamma=0.96
#LR_actor=0.0001
#LR_critic=0.00001
#batch_size=64

#inputfile="Ore blocks_sandbox4x4x3v2.xlsx"

#memcap=500
#EPISODES = 200
#dropout=0
test='DA2C'

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
        self.actionlimit=np.ones([self.Ilen, self.Jlen])      
        self.turns=(self.RLlen*self.Ilen*self.Jlen)

        
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
            
            self.actionslist.append(action)
            self.evaluate()
            self.update()     
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
    
        self.ob_sample[0,self.i,self.j,self.RL,2]=mined #update State channel to mined "-1"
      
        

    def reset(self):
        
        self.ob_sample=deepcopy(self.norm)       
        self.turnore=0
        self.discountedmined=0
        self.turncounter=0
        self.terminal=False
        self.i=-1
        self.j=-1
        self.actionslist=list()
        
        return self.ob_sample


#initialising
state = list([1])
action= list([1])

class DQNAgent:  
        
    def __init__(self, state_size, action_size):

        self.action_size = action_size
        self.memory = deque(maxlen=memcap)
        self.gamma = gamma   # discount rate
        self.batch_size = batch_size
        self.learning_rateC=LR_critic
        self.state_size=state_size
        self.policybatch=list()
        self.statebatch = deque(maxlen=batch_size)
        self.rewardbatch=list()
        self.advantagebatch=list()
        self.action_prbatch=list()
        self.advantage=float()
        self.action_size = action_size
        # setting the our created session as default session
        self.Amodel = self.build_Actor() 
        self.Vmodel = self.build_Critic()        
        self.trainingbatch=list()

    def memorize(self, state, action, reward, next_state, done):
     
        self.memory.append((state, action, reward, next_state, done))
        
        
    def a2c_replay(self):
        minibatch = random.sample(self.memory,self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            
            target = np.zeros((1, 1))
            advantages = np.zeros((1, self.action_size))
    
            value = self.Vmodel.predict(state)[0][0]
            next_value = self.Vmodel.predict(next_state)[0][0]
    
            if done:
                advantages[0][action] = reward - value
                target[0][0] = reward
            else:
                advantages[0][action] = reward + self.gamma * (next_value) - value
                target[0][0] = reward + self.gamma * next_value
    
            self.Amodel.fit(state, advantages, epochs=1, verbose=0)
            self.Vmodel.fit(state, target, epochs=1, verbose=0)
                
    
    def act(self, state):

        action_probs = self.Amodel.predict(state)
        critic_v = self.Vmodel.predict(state)[0][0]
        action = np.random.choice(self.action_size, p=np.squeeze(action_probs))
        action_onehot=np.zeros(self.action_size, dtype=float)
        action_onehot[action]=1
           
        return action, critic_v, action_onehot # returns action and critic value

#    def load(self, name):
#        self.model.load_weights(name)

#    def save(self, name):
#        self.model.save_weights(name)

    def build_Critic(self):

        model=Sequential()
        model.add(Conv3D(1, kernel_size=(1, 1, 1), activation='relu', kernel_initializer='he_uniform', input_shape=state_size, padding='valid'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse',
                  optimizer=Adam(lr=LR_critic))

        return model    

 
    def build_Actor(self):
        
        model=Sequential()
        model.add(Conv3D(1, kernel_size=(1, 1, 1), activation='relu', kernel_initializer='he_uniform', input_shape=state_size, padding='valid'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(optimizer=Adam(lr=LR_actor), loss='categorical_crossentropy')
                
        return model


     
if __name__ == "__main__":
    env = environment()
    state_size = env.geo_array.shape#[0]
    
    action_size = len(env.action_space)    #.n
    agent = DQNAgent(state_size, action_size)
    
    #agent.load("model.h5")
    done = False
    
    episodelist=list()
    scorelist=list()
    output=list()
    e=0
    #
    while time.time()<end:    
    #for e in range(EPISODES):
        e+=1
        agent.state = env.reset()

        while True:
            
            agent.action, critic_v, agent.act1hot = agent.act(agent.state) #, env.actionlimit
            next_state, reward, done = env.step(agent.action)
            agent.memorize(agent.state, agent.action, reward, next_state, done)
            agent.state = next_state
            
            if len(agent.memory)>= agent.batch_size:
                agent.a2c_replay()
            
            if done:
                #print("episode: {}/{}, score: {}, actions: {}"
                #      .format(e, EPISODES, env.discountedmined, env.actionslist))
                episodelist.append(e)
                scorelist.append(env.discountedmined)
                output.append([e,env.discountedmined, env.actionslist])
                
                break

   
    plt.plot(episodelist,scorelist)
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    #plt.show()
    
    scenario=str(f'{inputfile} t{test}, lr_a{LR_actor}, lr_c{LR_critic}, memory{memcap}, gamma{gamma}, batch{batch_size}')
    #agent.model.save(f'{scenario}_model.h5')
    plt.savefig(f'fig_{scenario}.png')
    outputdf=pd.DataFrame(output)
    outputdf.to_csv(f"output_{scenario}.csv")
    
