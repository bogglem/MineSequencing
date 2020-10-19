# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 18:45:45 2020

@author: Tim Pelech
"""

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
inputfile="BM_parametric15x15x5.xlsx"
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
    
    
    
    
    
    
