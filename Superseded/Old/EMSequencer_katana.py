# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 13:30:17 2020

@author: Tim Pelech
"""

import pandas as pd
import os
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.models import load_model
import keras.backend as tf
import sys

idx=int(sys.argv[1])

inputarray=pd.read_csv('job_input_array.csv')
 
LR=inputarray.loc[idx].LR
batch_size=inputarray.loc[idx].batch_size
EPSINIT=inputarray.loc[idx].EPSILON
inputfile=inputarray.loc[idx].inputfile
epsilon_min=inputarray.loc[idx].epsilon_min

start=time.time()
end=start+11.75*60*60

class environment: 

    def __init__(self,inputfile):
        
        self.inputdata=pd.read_excel(inputfile)
        self.data=self.inputdata
        s=np.ones(len(self.data))
        self.data['State']=pd.Series(s, index=self.data.index)
        self.actionslist = list()

        self.minI=min(self.data._I)
        self.maxI=max(self.data._I)
        
        self.minJ=min(self.data._J)
        self.maxJ=max(self.data._J) 
        
        self.turnreward=0
        self.turns=len(self.data)
        self.discountedmined=0
        self.idm=-1
        self.turncounter=0
        self.i=-1
        self.j=-1
        
        self.terminal=False
        
        self.action_space=np.zeros((self.maxI)*(self.maxJ))
        self.observation_space=self.data.values.flatten()

    def actcoords(self, action):
        #map coords
        q=np.zeros((self.maxI)*(self.maxJ))
        q[action]=1
        
        q2=q.reshape(self.maxI,self.maxJ)
        action_coords=np.argwhere(q2.max()==q2)[0]
        
        #mapping q values to action coordinates
        
        self.i=action_coords[0]+1
        self.j=action_coords[1]+1      
        
    def step(self, action):        

        self.actcoords(action)
        data2=self.data[(self.data._I==self.i)&(self.data._J==self.j)].RL
        #&(self.data.State==1)
        if (self.turncounter<self.turns) & (data2.empty!=True): 
                        
            self.actionslist.append(action)
            self.idm=data2.idxmax()
            self.evaluate()
            self.update()
            self.observe()
            self.turncounter+=1      
               
        else: 
            self.terminal =True
            
            #info=""
                    
        return self.observation_space, self.turnore, self.terminal, #info
    
    def evaluate(self):
    
        H2O=self.data.loc[self.idm, 'H2O']
        Tonnes=self.data.loc[self.idm, 'Tonnes']
        State=self.data.loc[self.idm, 'State']
     
        self.turnore=(H2O*Tonnes*State)
        self.discountedmined+=(self.turnore*(len(self.data)-(self.turncounter)/5)/len(self.data))
        
    def update(self):
    
        self.data.loc[self.idm, 'State'] = self.data.loc[self.idm, 'State']-1

      
    def observe(self):
        
        self.observation_space=self.data.values.flatten()
        

    def reset(self):
        
        #initiate
        #print("New trial, initiating map...")
        #self.data=
        
        s=np.ones(len(self.data))
        self.data['State']=pd.Series(s, index=self.data.index)
        
        self.minJ=min(self.data._J)
        self.minI=min(self.data._I)
        self.maxI=max(self.data._I)
        self.maxJ=max(self.data._J) 
        
        #self.I=range(self.minI,self.maxI+1)
        #self.J=range(self.minJ,self.maxJ+1)
        
        self.turnore=0
        self.turns=len(self.data)
        self.discountedmined=0
        self.idm=-1
        self.turncounter=0
        self.terminal=False
        self.i=-1
        self.j=-1
        
        #self.issurface()
        self.observation_space=self.data.values.flatten()
        self.actionslist=list()
        return self.observation_space


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


class DQNAgent:  
        
    def __init__(self, state_size, action_size, LR, EPSINIT, epsilon_min):
        
        
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=9000000)
        self.gamma = 0.7   # discount rate
        self.epsilon = EPSINIT  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = 0.999
        self.learning_rate = LR
        self.model = self.build_model()
        


    def build_model(self):
        # Neural Net for Deep-Q learning Model

        #if os.listdir().count('model.h5')>0:
         #   model=load_model('model.h5')
         #episodic_loss(state,action,self.memory)
        #else:
            model = Sequential()
            model.add(Dense(256, input_dim=self.state_size, activation='relu'))
            model.add(Dense(256, activation='relu'))
            #model.add(Dropout(0.1))
            model.add(Dense(256, activation='relu'))
            #model.add(Dropout(0.1))
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss=episodic_loss(state,action,self.memory),
                      optimizer=Adam(lr=self.learning_rate))

            return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
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
    env = environment(inputfile)
    state_size = env.observation_space.shape[0]
    action_size = len(env.action_space)    #.n
    agent = DQNAgent(state_size, action_size, LR, EPSINIT, epsilon_min)
    #agent.load("model.h5")
    done = False
    e=0
    episodelist=list()
    scorelist=list()
    output=list()
    
    while time.time()<end:
        e+=1
        agent.state = env.reset()
        agent.state = np.reshape(agent.state, [1, state_size])
        while True:
            # env.render()
            agent.action = agent.act(agent.state)
            next_state, reward, done = env.step(agent.action)
            #reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(agent.state, agent.action, reward, next_state, done)
            agent.state = next_state
            if done:
                #print("episode: {}, score: {}, e: {:.2}, actions: {}"
                #      .format(e, env.discountedmined, agent.epsilon, env.actionslist))
                
                episodelist.append(e)
                scorelist.append(env.discountedmined)
                
                output.append([e,env.discountedmined,agent.epsilon, env.actionslist])
                
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        # if e % 10 == 0:
    
    
    plt.plot(episodelist,scorelist)
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    
    
    scenario=str(f'{inputfile}, epsilon{epsilon_min}, lr{LR}, batch{batch_size}')
    agent.model.save(f'{scenario}_model.h5')
    plt.savefig(f'fig_{scenario}.png')
    outputdf=pd.DataFrame(output)
    outputdf.to_csv(f"output_{scenario}.csv")
    #plt.show()