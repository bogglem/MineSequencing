# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 13:28:51 2021

@author: Tim Pelech
"""

#filter tensorflow warnings
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

import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines import A2C
from stable_baselines import DQN
from stable_baselines import ACER
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.evaluation import evaluate_policy
#from tools.plotresults import plotresults
import tools.plotresults as plotresults
#from tools.BMenv import environment
from tools.loadsaveBMenv_cutoff import environment
#from tools.evalBMenv2_exerror import environment
#from tools.Fuzzy3DBMenv_9action import environment

# Create environment
x=15
y=15
z=4
batch_size=64
LR=0.001
gamma=0.8
turnspc=0.10
episodetimesteps=round(x*y*z*turnspc)
ncpu=16
cutoffarray=[0.0,0.02, 0.05,0.08,0.1,0.15]

policyname='MlpPolicy' #change this name to change RL policy type (MlpPolicy/CnnPolicy)

if policyname == 'CnnPolicy':
    
    policy=CnnPolicy
    test='CNNA2C'

elif policyname =='MlpPolicy':

    policy=MlpPolicy
    test='MLPA2C'

trialv='exerror'
#'loadsave10'

#prepare file naming strings
LR_s=str("{:f}".format(LR)).split('.')[1]
inputfile_s='%s_%s_%s' % (x,y,z)
gamma_s=str(gamma).replace('.','_')
#cutoff_s=str(cutoffpenaltyscalar).split('.')[0]
#rg_s=max(str(float(rg_prob)).split('.'))
turnspc_s=str(turnspc).split('.')[1]

scenario=str(f'{trialv}_{inputfile_s}_t{test}_lr{LR_s}_g{gamma_s}')  #_cpu{ncpu}
savepath='./output/%s/%s' % (scenario, 'eval')

turns=round(x*y*z*turnspc)



store=list()
labels=list()
cutoff_i=0

for cutoff in cutoffarray:
    
    env = environment(x,y,z, cutoff, turnspc, policyname, rg_prob='loadenv')
        
    if test=='CNNACER' or test=='MLPACER':
    
        # Instantiate the agent
        model = ACER(policy, env, gamma=gamma, learning_rate=LR,n_steps=episodetimesteps,   verbose=1)
        #model = DQN('MlpPolicy', env, learning_rate=LR, prioritized_replay=True, verbose=1)
        #
        # Load the trained agent
        model = ACER.load("%s/best_model" % savepath)
        #model = DQN.load("%s/best_model" % savepath)
        print('loaded agent %s' % savepath)
    
        
    else:
        # Instantiate the agent
        model = A2C(policy, env, gamma=gamma, learning_rate=LR,n_steps=episodetimesteps,   verbose=1)
        #model = DQN('MlpPolicy', env, learning_rate=LR, prioritized_replay=True, verbose=1)
        #
        # Load the trained agent
        model = A2C.load("%s/best_model" % savepath)
        #model = DQN.load("%s/best_model" % savepath)
        print('loaded agent %s' % savepath)
    
    
    for i in range(8): #number of repeats for non deterministic agent (MC method)
        
        
        # Evaluate the agent
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, deterministic=False)
        print('mean_reward = %s +/- %s' %(mean_reward,std_reward))
        
        # Enjoy trained agent
        env.rendermode='off'
        obs = env.reset()
        cumreward=0
        results=list()
        results_cutoff=list()
        minable=list()
        timestep=list()
        t=0
        while True:
        #for i in range(turns):
            action, _states = model.predict(obs, deterministic=False)
            obs, rewards, dones, info = env.step(action)
            cumreward+=rewards
            print(action, rewards, dones, cumreward)
            results.append(info[0])
            if info[0]>0:
                results_cutoff.append(info[0])
                timestep.append(t)
                t+=1
            a=abs(info[1]-1) #translating sequence errors to be positive, else zero
            minable.append(a)
            
            # if info[1]==1:
            #     results.append(info[0])
            #     a=abs(info[1]-1) #translating sequence errors to be positive, else zero
            #     minable.append(a)
            #env.renderif('on')
            if dones == True:
                while t<90:
                    results_cutoff.append(np.nan)
                    timestep.append(t)
                    t+=1
                break
        
            if len(results)>500:
                while t<90:
                    results_cutoff.append(np.nan)
                    timestep.append(t)
                    t+=1
                break
        
        label='Cutoff Grade {:.1f}%'.format(cutoff*100)
        #store.append([label, cutoff, i, env.geo_array, results, results_cutoff, minable])
        store.append([cutoff_i, timestep, results_cutoff])
    
    labels.append(label)    
    cutoff_i+=1

storearray=np.array(store)

#store=np.array(store)
plotresults.batchplot(storearray, labels)


    