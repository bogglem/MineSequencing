# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:52:39 2020

@author: Tim Pelech
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 12:25:04 2020

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
import sys
from keras.models import Sequential, clone_model, load_model
from keras.layers import Dense, Conv3D, MaxPooling3D, Flatten, Dropout, Input
from keras.optimizers import Adam
from copy import deepcopy
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from OPenv import environment
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
test='DA2C-OPenv'

start=time.time()
end=start+10.5*60*60

mined=-1


#initialising
state = list([1])
action= list([1])
#memory=list([1])


class DQNAgent:  
        
    def __init__(self, state_size, action_size):

        self.action_size = action_size
        self.memory = deque(maxlen=memcap)
        self.gamma = gamma   # discount rate
        self.batch_size = batch_size
        self.learning_rateC=LR_critic
        self.state_size=state_size
        self.statebatch = 0
        self.targetbatch=0
        self.advantagebatch=0
        self.advantage=float()
        self.action_size = action_size
        # setting the our created session as default session
        self.Amodel = self.build_Actor() 
        self.Vmodel = self.build_Critic()      


    def memorize(self, state, action, reward, next_state, done):
        
        self.memory.append((state, action, reward, next_state, done))
        
             
    def a2c_train(self):
        self.minibatch = random.sample(self.memory,self.batch_size)
 
        for state, action, reward, next_state, done in self.memory:

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
        #for q_ (PER)
        action_probs = self.Amodel.predict(state)
        action = np.random.choice(self.action_size, p=np.squeeze(action_probs))

           
        return action # returns action and critic value

#    def load(self, name):
#        self.model.load_weights(name)

#    def save(self, name):
#        self.model.save_weights(name)

    def build_Critic(self):

            model=Sequential()
            model.add(Conv3D(1, kernel_size=(1, 1, 1), activation='relu', kernel_initializer='he_uniform', input_shape=state_size, padding='valid'))
            model.add(Flatten())    
            model.add(Dense(64, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(dropout))
            model.add(Dense(1, activation='linear'))
            
            #model = Model(input=[inputl], output=[output])
            model.compile(loss='mse',
                      optimizer=Adam(lr=LR_critic))

            return model    

 
    def build_Actor(self):
        
        model=Sequential()
        model.add(Conv3D(1, kernel_size=(1, 1, 1), activation='relu', kernel_initializer='he_uniform', input_shape=state_size, padding='valid'))
        #Input(shape=[1])
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(self.action_size, activation='softmax'))
        #model = Model(inputs=state_input, outputs=output)
        #model = Model(input=[state_input, advantage], output=[output])
        model.compile(optimizer=Adam(lr=LR_actor), loss='categorical_crossentropy')
                
        return model


     
if __name__ == "__main__":
    env = environment(inputfile, gamma)
    state_size = env.geo_array.shape#[0]
    
    action_size = len(env.action_space)    #.n
    agent = DQNAgent(state_size, action_size)
   #agent.batch_size=env.turns #for one pass on policy algorithms
    
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
        actionslist=list()

        while True:
            
            agent.action = agent.act(agent.state) #, env.actionlimit
            next_state, reward, done = env.step(agent.action)
            
            agent.memorize(agent.state, agent.action, reward, next_state, done)
            agent.state = next_state
            actionslist.append(agent.action)
            if len(agent.memory)>= agent.batch_size:
                agent.a2c_train()
            
            if done:
                #agent.buffer_train()
                #print("episode: {}, score: {}, actions: {}"
                 #     .format(e, env.discountedmined, actionslist))
                    # replay compares against a stationary model
                episodelist.append(e)
                scorelist.append(env.discountedmined)
                output.append([e,env.discountedmined, actionslist])

                break
            #if len(agent.memory.minibatch()) > agent.batch_size:
                
    
#    agent.model.save("model.h5")
    plt.plot(episodelist,scorelist)
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    #plt.show()
    
    scenario=str(f'{inputfile} t{test}, lr_a{LR_actor}, lr_c{LR_critic}, memory{memcap}, gamma{gamma}, batch{batch_size}')
    #agent.model.save(f'{scenario}_model.h5')
    plt.savefig(f'fig_{scenario}.png')
    outputdf=pd.DataFrame(output)
    outputdf.to_csv(f"output_{scenario}.csv")