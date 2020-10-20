# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 18:45:45 2020

@author: Tim Pelech

"""

# Filter tensorflow version warnings
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


import time
import gym
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines import A2C
from OPenv_gym import environment

start=time.time()
end=start+2*60*60
inputfile="BM_central15x15x5.xlsx"
LR=0.001
LR2=0.000001
gamma=0.95
batch_size=64
#n_steps=5
inspectenv = environment(inputfile, gamma)
test='A2C'


episodetimesteps=int(inspectenv.turns)

eval_env=environment(inputfile,gamma, rendermode="on")

loaded_model=A2C.load("best_model")
ob=eval_env.reset()
cum_reward=0

for a in range(round(eval_env.flatlen*0.5)):
    
    action = loaded_model.predict(ob)
        
    ob, reward, terminal, _= eval_env.step(action)
    
    cum_reward+=reward    
    
    print(cum_reward)    
    
    
    
    
    
    
