#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 17:38:18 2020

@author: z3333990
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
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines import ACER
from tools.Fuzzy3DBMenv import environment

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

#idx=int(sys.argv[1]) #array row number. required for batch runs on pbs katana
idx=2

#prepare input parameters
inputarray=pd.read_csv('jobarrays/Fuzzy_drstrange_job_input.csv')

#block model (environment) dimensions
x=inputarray.loc[idx].x
y=inputarray.loc[idx].y
z=inputarray.loc[idx].z #must be greater than 6 for CNN

policyname=inputarray.loc[idx].policyname  #change this name to change RL policy type (MlpPolicy/CnnPolicy)

if policyname == 'CnnPolicy':
    
    policy=CnnPolicy
    test='CNNACER'

elif policyname =='MlpPolicy':

    policy=MlpPolicy
    test='MLPACER'

trialv=inputarray.loc[idx].trialv 
#LR_critic=inputarray.loc[idx].LR_critic
LR=inputarray.loc[idx].LR
#batch_size=int(inputarray.loc[idx].batch_size)
#memcap=int(inputarray.loc[idx].memcap)
#inputfile=inputarray.loc[idx].inputfile
gamma=inputarray.loc[idx].gamma
#dropout=float(inputarray.loc[idx].dropout)
runtime=inputarray.loc[idx].runtime
#cutoffpenaltyscalar=inputarray.loc[idx].cutoffpenaltyscalar #not currently implemented
#rg_prob=inputarray.loc[idx].rg_prob
turnspc=inputarray.loc[idx].turnspc
ncpu=inputarray.loc[idx].ncpu

start=time.time()
end=start+runtime
episodetimesteps=round(x*y*z*turnspc)

#prepare file naming strings
LR_s=str(LR).split('.')[1]
inputfile_s='Fuzzy_%s_%s_%s' % (x,y,z)
gamma_s=str(gamma).replace('.','_')
#cutoff_s=str(cutoffpenaltyscalar).split('.')[0]
#rg_s=rg_prob #max(str(float(rg_prob)).split('.'))
turnspc_s=str(turnspc).split('.')[1]
storagefolder='output'
scenario=str(f'{inputfile_s}_t{test}_lr{LR_s}_g{gamma_s}_{trialv}')    
savepath='./%s/%s' % (storagefolder ,scenario)
#savepath='%s/environment' % (savepath)

if (os.path.exists(savepath)!=True):
    os.mkdir(savepath) #make directory prior to multiprocessing to avoid broken pipe error

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
                model.save("%s/final_model" % savepath)
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
        
        env = environment(x, y, z, gamma, turnspc, savepath, policyname)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


if __name__ == '__main__':

    num_cpu = ncpu # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(x,y,z, i) for i in range(num_cpu)])
    eval_env=environment(x, y, z, gamma, turnspc, savepath, policyname)
    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)

    
    #create callbacks to record data, initiate events during training.
    callbacklist=CallbackList([TimeLimit(episodetimesteps), EvalCallback(eval_env, log_path=savepath, n_eval_episodes=20
                                                                         ,eval_freq=2000, deterministic=False, best_model_save_path=savepath)])
    
    if (os.path.exists("%s/best_model.zip" % savepath)):
        # Instantiate the agent
        model = ACER(policy, env, gamma=gamma, n_steps=episodetimesteps, learning_rate=LR,  buffer_size=500*episodetimesteps,  verbose=1)
        # Load the trained agent
        model = ACER.load("%s/best_model" % savepath, env=env)
        print('loaded agent')
        model.learn(total_timesteps=episodetimesteps**50, callback=callbacklist) #total timesteps set to very large number so program will terminate based on runtime parameter)
        
        
    else:
        #create model with Stable Baselines package.
        model = ACER(policy, env, gamma=gamma, n_steps=episodetimesteps, learning_rate=LR,  buffer_size=500*episodetimesteps,  verbose=1)#, tensorboard_log=scenario)
        #model = ACER.load("%s/best_model" % savepath, env)
        model.learn(total_timesteps=episodetimesteps**50, callback=callbacklist) #total timesteps set to very large number so program will terminate based on runtime parameter)
            
    
    #create learning curve plot
    evaluations = './%s/%s/evaluations.npz' % (storagefolder,scenario)
    data=np.load(evaluations)
    results=data['results']
    y=np.average(results, axis=1)
    timesteps=data['timesteps']
    plt.plot(timesteps,y)
    
    plt.xlabel('Timesteps')
    plt.ylabel('Score')
    #plt.show() 
    
    #save learning curve plot
    figsavepath='./%s/%s/fig_%s' % (storagefolder ,scenario, scenario)
    plt.savefig(figsavepath)
    
    
    