#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 09:35:16 2020

@author: z3333990
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

from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines.common.schedules import PiecewiseSchedule
from stable_baselines import SAC, A2C
from OPRG3Denv_gym import environment

test='test'


#idx=int(sys.argv[1])

#inputarray=pd.read_csv('SB_job_input_array.csv')
 
#LR_critic=inputarray.loc[idx].LR_critic
#LR=inputarray.loc[idx].LR
#batch_size=int(inputarray.loc[idx].batch_size)
#memcap=int(inputarray.loc[idx].memcap)
#inputfile=inputarray.loc[idx].inputfile
#gamma=inputarray.loc[idx].gamma
#dropout=float(inputarray.loc[idx].dropout)


start=time.time()
end=start+2*60*60
#inputfile="BM_easy10x10x8.xlsx"
LR=0.00001
LR2=0.000001
gamma=0.95
batch_size=32
#n_steps=5
x=15
y=15
z=6

#inspectenv = environment(x,y,z, gamma)

episodetimesteps=round(x*y*z*0.5)#int(inspectenv.turns)



class TimeLimit(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, episodetimesteps):
        super(TimeLimit, self).__init__()
        self.check_freq = episodetimesteps
        self.incomplete = True
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if time.time()<end:
                self.incomplete = True
            else:
                self.incomplete = False
                 
        
        return self.incomplete
    
    

def make_env(x,y,z, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = environment(x,y,z, gamma)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


#points_values=list([[0,LR1],[1000000,LR2]])

#Sched=PiecewiseSchedule(points_values, outside_value=LR2)

if __name__ == '__main__':

    num_cpu = 1  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(x,y,z, i) for i in range(num_cpu)])
    eval_env=environment(x,y,z, gamma)
    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)
    scenario=str(f'RG_t{test}_lr{LR}_gamma{gamma}_batch{batch_size}')    
    callbacklist=CallbackList([TimeLimit(episodetimesteps), EvalCallback(eval_env, log_path=scenario, deterministic=False)])
    

        
    model = A2C(CnnPolicy, env, gamma=gamma, verbose=1)#, tensorboard_log=scenario)
    model.learn(total_timesteps=episodetimesteps**99, callback=callbacklist)
