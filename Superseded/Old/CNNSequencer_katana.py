#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 17:10:02 2020

@author: z3333990
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:52:39 2020

@author: Tim Pelech

"""

import pandas as pd

# -*- coding: utf-8 -*-
import sys
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv3D, MaxPooling3D, Flatten, Dropout
from keras.optimizers import Adam
from keras.models import load_model
from copy import deepcopy
import keras.backend as tf

idx=int(sys.argv[1])

inputarray=pd.read_csv('job_input_array.csv')
 
LR=inputarray.loc[idx].LR
batch_size=inputarray.loc[idx].batch_size
EPSINIT=inputarray.loc[idx].EPSILON
inputfile=inputarray.loc[idx].inputfile
epsilon_min=inputarray.loc[idx].epsilon_min

start=time.time()
end=start+11.5*60*60

class environment: 

    def __init__(self):
        
        self.inputdata=pd.read_excel(inputfile)
        self.data=self.inputdata
        self.actionslist = list()
        self.maxI=max(self.data._I)
        self.maxJ=max(self.data._J) 
        self.turnore=0     
        self.discountedmined=0
        self.turncounter=0
        self.i=-1
        self.j=-1
        self.terminal=False
        self.action_space=np.zeros((self.maxI)*(self.maxJ))
        self.Imin=self.data._I.min()
        self.Imax=self.data._I.max()
        self.Jmin=self.data._J.min()
        self.Jmax=self.data._J.max()
        self.RLmin=self.data.RL.min()
        self.RLmax=self.data.RL.max()
        self.channels = 2
        self.geo_array= np.zeros([self.Imax+1-self.Imin, self.Jmax+1-self.Jmin, self.RLmax+1-self.RLmin, self.channels], dtype=float)
                
        for i in self.data.index:
            self.geo_array[self.data._I[i]-1,self.data._J[i]-1,self.data.RL[i]-1,0]=self.data.H2O[i]
            self.geo_array[self.data._I[i]-1,self.data._J[i]-1,self.data.RL[i]-1,1]=self.data.Tonnes[i]
            
        self.ob_sample=deepcopy(np.reshape(self.geo_array,((1,self.Imax+1-self.Imin, self.Jmax+1-self.Jmin, self.RLmax+1-self.RLmin, self.channels))))

        self.turns=(self.RLmax*self.Jmax*self.Imax)


    def actcoords(self, action):
        #map coords
        q=np.zeros((self.maxI)*(self.maxJ))
        q[action]=1
        
        q2=q.reshape(self.maxI,self.maxJ)
        action_coords=np.argwhere(q2.max()==q2)[0]
        
        #mapping q values to action coordinates
        
        self.i=action_coords[0]#+1
        self.j=action_coords[1]#+1      
        
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
        Tonnes=-1
        
        for RLidx in reversed(range(self.RLmin-1,self.RLmax)):
            
            if self.ob_sample[0,self.i,self.j,RLidx,1]!=-1: 
                self.RL=RLidx
                H2O=self.ob_sample[0,self.i,self.j,self.RL,0]
                Tonnes=self.ob_sample[0,self.i, self.j,self.RL,1]
                break

        self.turnore=(H2O*Tonnes)
        self.discountedmined+=self.turnore*(self.turns-self.turncounter)/self.turns
        
    def update(self):
    
        self.ob_sample[0,self.i,self.j,self.RL,0]=0
        self.ob_sample[0,self.i,self.j,self.RL,1]=-1
      
        

    def reset(self):
        
          
        self.ob_sample=deepcopy(np.reshape(self.geo_array,((1,self.Imax+1-self.Imin, self.Jmax+1-self.Jmin, self.RLmax+1-self.RLmin, self.channels))))

        
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
memory=list([1])

def episodic_loss(state, action, memory):
    maxq=list()
    alpha=0.5
    beta=0.5
    for state_s, action_s, reward_s, next_state_s, done_s in memory:
        if state == state_s:
            if action == action_s:
                maxq.append(reward)
    
    maxqdf=pd.DataFrame(maxq)
    if maxqdf.empty!=True:    
        q_memory=list(maxqdf.loc[maxqdf.idxmax().idxmax()])
    else:
        alpha=1
        beta=0
        q_memory=0
        
    def model_loss(q_values,q_predicted):

        return tf.sum([alpha*tf.square(q_values-q_predicted),beta*tf.square(q_values-q_memory)])
            
    return model_loss     


EPISODES = 100

class DQNAgent:  
        
    def __init__(self, state_size, action_size):

        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = 0.9   # discount rate
        self.epsilon = EPSINIT  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = 0.995
        self.learning_rate = LR
        self.batch_size = batch_size
        self.model = self.build_model()
        


    def build_model(self):
        # Neural Net for Deep-Q learning Model

        #if os.listdir().count('model.h5')>0:
         #   model=load_model('model.h5')
         #episodic_loss(state,action,self.memory)
        #else: a generator to produce such vector for 3D C
        

            model = Sequential()
            model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=state_size))
            model.add(MaxPooling3D(pool_size=(2, 2, 1)))
            model.add(Dropout(0.1))
            model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
            model.add(MaxPooling3D(pool_size=(2, 2, 2)))
            model.add(Dropout(0.1))
            model.add(Flatten())            
    
            #model.add(Dense(24, input_dim=self.state_size, activation='relu'))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))
            
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        #episodic_loss(state,action,self.memory)
            return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            q_update = reward
            if not done:
                q_update = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

#    def load(self, name):
#        self.model.load_weights(name)

#    def save(self, name):
#        self.model.save_weights(name)


if __name__ == "__main__":
    env = environment()
    state_size = env.geo_array.shape#[0]
    
    action_size = len(env.action_space)    #.n
    agent = DQNAgent(state_size, action_size)
    #agent.load("model.h5")
    done = False
    e=0
    episodelist=list()
    scorelist=list()
    output=list()
    
    while time.time()<end:
        e+=1
        agent.state = env.reset()

        while True:

            agent.action = agent.act(agent.state)
            next_state, reward, done = env.step(agent.action)
            agent.memorize(agent.state, agent.action, reward, next_state, done)
            agent.state = next_state
            if done:
                #print("episode: {}/{}, score: {}, e: {:.2}, actions: {}"
                #      .format(e, EPISODES, env.discountedmined, agent.epsilon, env.actionslist))
                
                episodelist.append(e)
                scorelist.append(env.discountedmined)
                output.append([e,env.discountedmined,agent.epsilon, env.actionslist])
                
                break
            if len(agent.memory) > agent.batch_size:
                agent.replay()
    
    agent.model.save("model.h5")
    plt.plot(episodelist,scorelist)
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    #plt.show()
    
    scenario=str(f'{inputfile}, epsilon{epsilon_min}, lr{LR}, batch{batch_size}')
    agent.model.save(f'{scenario}_model.h5')
    plt.savefig(f'fig_{scenario}.png')
    outputdf=pd.DataFrame(output)
    outputdf.to_csv(f"output_{scenario}.csv")
    
