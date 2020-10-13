#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 17:49:36 2020

@author: z3333990
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 17:38:18 2020

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

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines import ACKTR
from OPenv_gym import environment

test='ACKTR'


idx=int(sys.argv[1])

inputarray=pd.read_csv('SB_job_input_array.csv')
 
#LR_critic=inputarray.loc[idx].LR_critic
LR=inputarray.loc[idx].LR
batch_size=int(inputarray.loc[idx].batch_size)
#memcap=int(inputarray.loc[idx].memcap)
inputfile=inputarray.loc[idx].inputfile
gamma=inputarray.loc[idx].gamma
#dropout=float(inputarray.loc[idx].dropout)


start=time.time()
end=start+11.5*60*60
#inputfile="BM_easy6x6x4.xlsx"
#LR=0.000001
#gamma=0.995
#batch_size=32
#n_steps=5
inspectenv = environment(inputfile, gamma)

episodetimesteps=int(inspectenv.turns)


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

def make_env(inputfile, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = environment(inputfile,gamma)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


if __name__ == '__main__':

    num_cpu = 12  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(inputfile, i) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)
    
    timelimit=TimeLimit(episodetimesteps)
    
    scenario=str(f'{inputfile}_t{test}_lr{LR}_gamma{gamma}_batch{batch_size}')
        
    model = ACKTR(MlpPolicy, env, gamma=gamma, n_steps=batch_size, learning_rate=LR, lr_schedule='constant', verbose=1, tensorboard_log=scenario)
    model.learn(total_timesteps=episodetimesteps**99, callback=timelimit)
    
    
    
    