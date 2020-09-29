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
from keras.models import Sequential, clone_model, load_model
#from keras.utils import normalize
from keras.layers import Dense, Conv3D, MaxPooling3D, Flatten, Dropout
from keras.optimizers import Adam
from copy import deepcopy
import keras.backend as tf
from sklearn.preprocessing import MinMaxScaler
#from tensorflow import Print


LR=0.0001
batch_size=64
EPSINIT=1.0
inputfile="Ore blocks_sandbox4x4x3v2.xlsx"
epsilon_min=0.01
memcap=10000
EPISODES = 200
dropout=0.2
test='PER-dropout'

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
        self.epsilonmod=np.zeros(self.turns)
        
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
            
            self.actionslist.append(action)
            self.evaluate()
            self.update()     
            self.actioncounter[action]+=1
            #if max(self.actioncounter)>self.RLlen:
            self.epsilonmod[self.turncounter]=round(max((self.actioncounter))/(self.RLlen),ndigits=2)
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
        self.discountedmined+=self.turnore*(self.turns-self.turncounter)/self.turns
        
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


#initialising
state = list([1])
action= list([1])
#memory=list([1])

class SumTree:
    write = 0

    def __init__(self, memcap):
        self.capacity = memcap
        self.tree = np.zeros( 2*self.capacity - 1 )
        self.data = np.zeros( self.capacity, dtype=object )

    def _propagate(self, idx, delta):
        parent = (idx - 1) // 2

        self.tree[parent] += delta

        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        delta = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, delta)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, memcap):
        self.tree = SumTree(memcap)

    def _getPriority(self, TDerror):
        return (TDerror + self.e) ** self.a

    def add(self, TDerror, sample):
        p = self._getPriority(TDerror)
        self.tree.add(p, sample) 

    def minibatch(self, batch_size):
        batch = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append(data)
            #print(p)
        return batch

    def update(self, idx, TDerror):
        p = self._getPriority(TDerror)
        self.tree.update(idx, p)


def randomact(actionlimit):
        
    actnotemptydf=pd.DataFrame(actionlimit.flatten())
    actnotemptydf.columns=['a']
    b=actnotemptydf[actnotemptydf.a==1]
    actnotempty=list(b.index)
    act=random.sample(actnotempty,1)[0]
          
    return act


class DQNAgent:  
        
    def __init__(self, state_size, action_size):

        self.action_size = action_size
        self.memory = Memory(memcap)
        self.gamma = 0.95   # discount rate
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
#state_size[2]
            model = Sequential()
            model.add(Conv3D(1, kernel_size=(1, 1, 1), activation='relu', kernel_initializer='he_uniform', input_shape=state_size, padding='valid'))
            #model.add(MaxPooling3D(pool_size=(1, 1, 1)))
            #model.add(Dropout(0.1))
            #model.add(Conv3D(3, kernel_size=(3, 3, 3), strides=(2,2,2), activation='relu', kernel_initializer='he_uniform', padding='valid'))
            #model.add(Dropout(0.1))
            #model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            #model.add(Dropout(0.1))
            
            model.add(Flatten())    

            #model.add(Dense(64, activation='relu'))
            #model.add(Dropout(0.1))
            #model.add(Dense(24, input_dim=self.state_size, activation='relu'))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(dropout))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(dropout))
            #model.add(Dense(64, activation='relu'))
            #model.add(Dropout(0.5))
            model.add(Dense(self.action_size, activation='linear'))
            #episodic_loss(state,action,self.memory)
            model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))

        
            return model

    def memorize(self, state, action, reward, next_state, done, q_):

        TDerror = abs(reward-q_)
        
        self.memory.add(TDerror,[state, action, reward, next_state, done])
        
 
    def act(self, state, actionlimit, epsilonmod):
        #for q_ (PER)
        act_values = self.model.predict(state)

        if np.random.rand() <= self.epsilon:
            action = randomact(actionlimit) #random.randrange(self.action_size)  
  
        elif epsilonmod>1:
            
            if np.random.rand() <= 0.5:
                action = randomact(actionlimit) #random.randrange(self.action_size)
            else: 
                action = np.argmax(act_values[0])        
               
        elif epsilonmod<=1:
                   
                action = np.argmax(act_values[0])
        
        q_= max(act_values[0])
        return action , q_  # returns action and q value

    def replay(self, stationary_model):
        minibatch = self.memory.minibatch(self.batch_size) #random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            q_update = reward
            if not done:
                q_update = np.amax(stationary_model.predict(state)[0])+LR*(reward + self.gamma * 
                                  np.amax(stationary_model.predict(next_state)[0])-np.amax(stationary_model.predict(state)[0]))
            q_values = stationary_model.predict(state)
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
    
    episodelist=list()
    scorelist=list()
    output=list()
    e=0
    #
    #while time.time()<end:    
    for e in range(EPISODES):
      #  e+=1
        agent.state = env.reset()
         
        stationary_model=clone_model(agent.model)


        while True:
            
            agent.action, q_ = agent.act(agent.state, env.actionlimit, env.epsilonmod[env.turncounter-1])
            next_state, reward, done = env.step(agent.action)
            agent.memorize(agent.state, agent.action, reward, next_state, done, q_)
            agent.state = next_state
            agent.replay(stationary_model)
            
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}, actions: {}, expmod: {}"
                      .format(e, EPISODES, env.discountedmined, agent.epsilon, env.actionslist, env.epsilonmod))
                    # replay compares against a stationary model
                episodelist.append(e)
                scorelist.append(env.discountedmined)
                output.append([e,env.discountedmined,agent.epsilon, env.actionslist, deepcopy(env.epsilonmod)])
                
                break
            #if len(agent.memory.minibatch()) > agent.batch_size:
                
    
#    agent.model.save("model.h5")
    plt.plot(episodelist,scorelist)
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    #plt.show()
    
    scenario=str(f'{inputfile} test{test}, epsilon{epsilon_min}, lr{LR}, batch{batch_size}')
    agent.model.save(f'{scenario}_model.h5')
    plt.savefig(f'fig_{scenario}.png')
    outputdf=pd.DataFrame(output)
    outputdf.to_csv(f"output_{scenario}.csv")
    
