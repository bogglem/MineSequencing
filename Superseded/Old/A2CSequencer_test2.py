# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 14:06:01 2020

@author: Tim Pelech
"""

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
        
        a=H2O_scaled.reshape([self.Ilen, self.Jlen, self.RLlen,1])
        b=Tonnes_scaled.reshape([self.Ilen, self.Jlen, self.RLlen,1])
        c=State_reshaped.reshape([self.Ilen, self.Jlen, self.RLlen,1])
               
        self.norm=np.append(a, b, axis=3)
        self.norm=np.append(self.norm,c, axis=3)
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
            
            if self.ob_sample[self.i,self.j,RLidx,2]!=mined: #if State unmined
                self.RL=RLidx
                H2O=self.geo_array[self.i,self.j,self.RL,0]
                Tonnes=self.geo_array[self.i, self.j,self.RL,1]
                #State=self.ob_sample[0,self.i, self.j,self.RL,2]              
                break
        
        if self.ob_sample[self.i,self.j,RLidx,2]==mined: #if all mined in i,j column
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
        self.ob_sample[self.i,self.j,self.RL,2]=mined #update State channel to mined "-1"
      
        

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
        self.policybatch=list()
        self.statebatch = deque(maxlen=batch_size)
        self.rewardbatch=list()
        self.advantagebatch=list()
        self.action_prbatch=list()
        
        self.action_size = action_size
        # setting the our created session as default session
        self.Amodel = self.build_Actor() 
        self.Vmodel = self.build_Critic()        
        self.trainingbatch=list()
        
        #self.VCritic = VCritic(state_size).model
        #self.PActor= PActor(self.action_size, state_size).model


    def memorize(self, state, action, reward, next_state, done):
     
        self.memory.append((state, action, reward, next_state, done))
        
        
    
    def a2c_replay(self):
        minibatch = random.sample(self.memory,self.batch_size)
        self.policybatch=list()
        self.statebatch=deque(maxlen=batch_size)
        self.rewardbatch=list()
        self.advantagebatch=list()
        self.action_pr=list()
        i=0
        #X = []
        #y = []
        #advantages = np.zeros(shape=(batch_size, action_size))
        for cur_state, action, reward, next_state, done in minibatch:
             
            if done:
                # If last state then advatage A(s, a) = reward_t - V(s_t)
                advantage = reward - self.Vmodel.predict(cur_state)[0][0]
            else:
                # If not last state the advantage A(s_t, a_t) = reward_t + DISCOUNT * V(s_(t+1)) - V(s_t)
                next_reward = self.Vmodel.predict(next_state)[0][0]
                advantage = reward + self.gamma * next_reward - self.Vmodel.predict(cur_state)[0][0]
                # Updating reward to trian state value fuction V(s_t)
                reward = reward + self.gamma * next_reward
            
            action_pr=self.Amodel.predict(cur_state)[0][action]
            self.statebatch.append(cur_state)
            self.advantagebatch.append(advantage)
            self.action_prbatch.append(action_pr)
            self.rewardbatch.append(reward)
            i+=1
            statebatch=np.array(self.statebatch)
            
        self.Amodel.fit(statebatch, self.action_prbatch, batch_size=self.batch_size, epochs=1,verbose=0)
        self.Vmodel.fit(statebatch, self.rewardbatch, batch_size=self.batch_size, epochs=1, verbose=0)

    
    def act(self, state):
        #for q_ (PER)
        action_probs = self.Amodel.predict([0, state])
        critic_v = self.Vmodel.predict([0, state])[0][0]
        action = np.random.choice(self.action_size, p=np.squeeze(action_probs))
           
        return action, critic_v # returns action and critic value

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
            model.add(Dense(1, activation='linear'))
            
            #model = Model(input=[inputl], output=[output])
            model.compile(loss='mse',
                      optimizer=Adam(lr=LR_critic))

            return model    

 
    def build_Actor(self):
        
        def policy_update(self):
          
            def customloss(y_true, y_pred):
                advantage=self.advantagebatch
                y_true=self.action_prbatch
                
                return K.sum(-K.log(y_true)*advantage)
                
            return customloss 
            
        
        model=Sequential()
        model.add(Conv3D(1, kernel_size=(1, 1, 1), activation='relu', kernel_initializer='he_uniform', input_shape=state_size, padding='valid'))
        #Input(shape=[1])
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(self.action_size, activation='softmax'))
        #model = Model(inputs=state_input, outputs=output)
        #model = Model(input=[state_input, advantage], output=[output])
        model.compile(optimizer=Adam(lr=LR_actor), loss=policy_update(self))
                
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
    #while time.time()<end:    
    for e in range(EPISODES):
      #  e+=1
        agent.state = env.reset()
         
        #stationary_model=clone_model(agent.model)


        while True:
            
            agent.action, critic_v = agent.act(agent.state) #, env.actionlimit
            next_state, reward, done = env.step(agent.action)
            agent.memorize(agent.state, agent.action, reward, next_state, done)
            agent.state = next_state
            
            if len(agent.memory)>= agent.batch_size:
                agent.a2c_replay()
            
            if done:
                print("episode: {}/{}, score: {}, actions: {}"
                      .format(e, EPISODES, env.discountedmined, env.actionslist))
                    # replay compares against a stationary model
                episodelist.append(e)
                scorelist.append(env.discountedmined)
                output.append([e,env.discountedmined, env.actionslist])
                
                break
            #if len(agent.memory.minibatch()) > agent.batch_size:
                
    
#    agent.model.save("model.h5")
    plt.plot(episodelist,scorelist)
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    #plt.show()
    
    scenario=str(f'{inputfile} test{test}, lr_a{LR_actor}, lr_a{LR_critic}, batch{batch_size}')
    agent.model.save(f'{scenario}_model.h5')
    plt.savefig(f'fig_{scenario}.png')
    outputdf=pd.DataFrame(output)
    outputdf.to_csv(f"output_{scenario}.csv")
    
