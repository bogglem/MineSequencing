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
from OPRG3Dp1env_gym import environment

# Create environment
x=15
y=15
z=6
batch_size=64
LR=0.0004
gamma=0.9
cutoffpenaltyscalar=1.0
rg_prob=0.1

env = environment(x,y,z,gamma, cutoffpenaltyscalar, rg_prob)

# Instantiate the agent
model = A2C(CnnPolicy, env, gamma=gamma, n_steps=batch_size, learning_rate=LR,  verbose=1)

# Load the trained agent
model = A2C.load("best_model")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

# Enjoy trained agent
obs = env.reset()
for i in range(200):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()