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
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.evaluation import evaluate_policy
from tools.RG3DBMenv import environment

# Create environment
x=15
y=15
z=6
batch_size=64
LR=0.0001
gamma=0.99
cutoffpenaltyscalar=1.0
rg_prob=0.0
turnspc=0.6
test='CNNA2C'
trialv='single_env'

#prepare file naming strings
LR_s=str(LR).split('.')[1]
inputfile_s='RG_%s_%s_%s' % (x,y,z)
gamma_s=str(gamma).split('.')[1]
cutoff_s=str(cutoffpenaltyscalar).split('.')[0]
rg_s=str(float(rg_prob)).split('.')[1]
turnspc_s=str(turnspc).split('.')[1]

scenario=str(f'{inputfile_s}_t{test}_lr{LR_s}_rg{rg_s}_cutoff{cutoff_s}_{trialv}')  
savepath='./output/%s' % scenario

turns=round(x*y*z*turnspc)

env = environment(x,y,z,gamma, cutoffpenaltyscalar, rg_prob, turnspc, savepath)

# Instantiate the agent
model = A2C(CnnPolicy, env, gamma=gamma, n_steps=batch_size, learning_rate=LR,  verbose=1)

# Load the trained agent
model = A2C.load("./%s/best_model" % savepath)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

# Enjoy trained agent
obs = env.reset()
for i in range(turns):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.renderif('on')