# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 13:11:40 2022

@author: Tim Pelech
"""

# Remove tensorflow version warnings
import os
# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging
tf.get_logger().setLevel(logging.ERROR)


#import modules
import time
import gym
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines import A2C
from stable_baselines import ACER
from tools.humanBMenv import environment
from tools.plotresults import plotresults
import numpy as np


class human():

    def __init__(self, env, humantraj): #HumanTrajectory
        
        self.traj = np.genfromtxt(str(f'{humantraj}.csv'), delimiter=',')  
        self.results=list()
        self.minable=list()
        self.env=env
        
    def run(self):
        self.env.reset()
        counter=0
        self.env.rendermode='off'
        
        for s in self.traj:
            
            obs, rewards, dones, info = env.step(int(s))         
            #self.env.render()         
            self.results.append(info[0])
            a=abs(info[1]-1)
            self.minable.append(a)
            
            counter+=1
        


class ai():

    def __init__(self, env):
    
        self.env=env
        # chose trained agent type
        x=env.Imax
        y=env.Jmax
        z=env.RLmax
        LR=0.001
        gamma=0.8
        turnspc=0.10
        ncpu=16
        self.episodetimesteps=round(x*y*z*turnspc)
        
        policyname='MlpPolicy' #change this name to change RL policy type (MlpPolicy/CnnPolicy)
        
        if policyname == 'CnnPolicy':
            
            policy=CnnPolicy
            test='CNNA2C'
        
        elif policyname =='MlpPolicy':
        
            policy=MlpPolicy
            test='MLPA2C'
        
        trialv='exerror'
        
        #prepare file naming strings
        LR_s=str("{:f}".format(LR)).split('.')[1]
        inputfile_s='%s_%s_%s' % (x,y,z)
        gamma_s=str(gamma).replace('.','_')
        turnspc_s=str(turnspc).split('.')[1]
        
        scenario=str(f'{trialv}_{inputfile_s}_t{test}_lr{LR_s}_g{gamma_s}')  #_cpu{ncpu}
        savepath='./output/%s' % scenario
               
       # env = environment(x,y,z,gamma, turnspc, policyname)
        
        if test=='CNNACER' or test=='MLPACER':
        
            # Instantiate the agent
            self.model = ACER(policy, env, gamma=gamma, learning_rate=LR,n_steps=self.episodetimesteps,   verbose=1)
            #model = DQN('MlpPolicy', env, learning_rate=LR, prioritized_replay=True, verbose=1)
            #
            # Load the trained agent
            self.model = ACER.load("%s/best_model" % savepath)
            #model = DQN.load("%s/best_model" % savepath)
            print('loaded agent %s' % savepath)

            
        else:
            # Instantiate the agent
            self.model = A2C(policy, env, gamma=gamma, learning_rate=LR,n_steps=self.episodetimesteps,   verbose=1)
            #model = DQN('MlpPolicy', env, learning_rate=LR, prioritized_replay=True, verbose=1)
            #
            # Load the trained agent
            self.model = A2C.load("%s/best_model" % savepath)
            #model = DQN.load("%s/best_model" % savepath)
            print('loaded agent %s' % savepath)
            
            

        # Evaluate the agent
        mean_reward, std_reward = evaluate_policy(self.model, self.env, n_eval_episodes=20, deterministic=False)
        print('mean_reward = %s +/- %s' %(mean_reward,std_reward))
        self.results=list()
        self.minable=list()
        
    def run(self):  # Enjoy trained agent
        cumreward=0
        obs = self.env.reset()
        self.env.rendermode='off'

        while self.env.terminal==False: #i in range(self.episodetimesteps):
            action, _states = self.model.predict(obs, deterministic=False)
            obs, rewards, dones, info = self.env.step(action)
            cumreward+=rewards
            print(action, rewards, dones, cumreward)
            if info[1]==1:
                self.results.append(info[0])
                a=abs(info[1]-1) #translating sequence errors to be positive, else zero
                self.minable.append(a)
            
            env.renderif('on')
            if dones == True:
                break
    
    #def plotr(self):
        
        
        
if __name__ == '__main__':
    
    envnum=1
    
    env = environment(15, 15, 4, 0.1, 'MlpPolicy', rg_prob=envnum)
    
    trajname='HumanTrajectory%s' % envnum
    human=human(env, trajname)
            
    ai=ai(env)
    
    human.run()
    
    ai.run()
    
    plotresults.subplot([human.results, ai.results], ['Human', 'Agent'], env.geo_array, [human.minable, ai.minable])
    
    