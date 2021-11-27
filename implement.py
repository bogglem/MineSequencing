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

from stable_baselines import A2C
from stable_baselines import ACER
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.evaluation import evaluate_policy
from tools.Fuzzy3DBMenv import environment
#from tools.RG3DBMenv import environment

# Create environment
x=20
y=20
z=6
batch_size=64
LR=0.0001
gamma=0.95
turnspc=0.05
episodetimesteps=round(x*y*z*turnspc)

policyname='MlpPolicy' #change this name to change RL policy type (MlpPolicy/CnnPolicy)

if policyname == 'CnnPolicy':
    
    policy=CnnPolicy
    test='CNNACER'

elif policyname =='MlpPolicy':

    policy=MlpPolicy
    test='MLPACER'

trialv='13kenv-20cpu'

#prepare file naming strings
LR_s=str(LR).split('.')[1]
inputfile_s='Fuzzy_%s_%s_%s' % (x,y,z)
gamma_s=str(gamma).replace('.','_')
#cutoff_s=str(cutoffpenaltyscalar).split('.')[0]
#rg_s=max(str(float(rg_prob)).split('.'))
turnspc_s=str(turnspc).split('.')[1]

scenario=str(f'{inputfile_s}_t{test}_lr{LR_s}_g{gamma_s}_{trialv}')  
savepath='./output/%s' % scenario

turns=round(x*y*z*turnspc)

env = environment(x,y,z,gamma, turnspc, savepath, policyname)

# Instantiate the agent
model = ACER(policy, env, gamma=gamma, learning_rate=LR,n_steps=episodetimesteps,   verbose=1)
#
# Load the trained agent
model = ACER.load("%s/best_model" % savepath)
print('loaded agent %s' % savepath)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, deterministic=False)
print('mean_reward = %s +/- %s' %(mean_reward,std_reward))

# Enjoy trained agent
obs = env.reset()
cumreward=0
for i in range(turns):
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = env.step(action)
    cumreward+=rewards
    print(action, rewards, dones, cumreward)
    env.renderif('on')
    if dones == True:
        break