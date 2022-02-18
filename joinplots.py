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
from stable_baselines import A2C
from tools.loadsaveBMenv_excludeerror import environment
from tools.evalBMenv2 import environment as evalenv

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#idx=int(sys.argv[1]) #array row number. required for batch runs on pbs katana



#prepare input parameters
inputarray=pd.read_csv('jobarrays/general_katana_job_input.csv')

#for idx in range(len(inputarray)):
idx=1
#block model (environment) dimensions
x=inputarray.loc[idx].x
y=inputarray.loc[idx].y
z=inputarray.loc[idx].z #must be greater than 6 for CNN

policyname=inputarray.loc[idx].policyname  #change this name to change RL policy type (MlpPolicy/CnnPolicy)

if policyname == 'CnnPolicy':
    
    policy=CnnPolicy
    test='CNNA2C'

elif policyname =='MlpPolicy':

    policy=MlpPolicy
    test='MLPA2C'

trialv=inputarray.loc[idx].trialv 
#LR_critic=inputarray.loc[idx].LR_critic
LR=inputarray.loc[idx].LR
#batch_size=int(inputarray.loc[idx].batch_size)
#memcap=int(inputarray.loc[idx].memcap)
#inputfile=inputarray.loc[idx].inputfile
gamma=inputarray.loc[idx].gamma
#dropout=float(inputarray.loc[idx].dropout)
runtime=inputarray.loc[idx].runtime
#penaltyscalar=inputarray.loc[idx].penaltyscalar #not currently implemented
#rg_prob=inputarray.loc[idx].rg_prob
turnspc=inputarray.loc[idx].turnspc
ncpu=inputarray.loc[idx].ncpu

start=time.time()
end=start+runtime
episodetimesteps=round(x*y*z*turnspc)

#prepare file naming strings
LR_s=str("{:f}".format(LR)).split('.')[1]
inputfile_s='%s_%s_%s' % (x,y,z)
gamma_s=str(gamma).replace('.','_')
#cutoff_s=str(cutoffpenaltyscalar).split('.')[0]
#rg_s=rg_prob #max(str(float(rg_prob)).split('.'))
turnspc_s=str(turnspc).split('.')[1]
storagefolder='output'
scenario=str(f'{trialv}_{inputfile_s}_t{test}_lr{LR_s}_g{gamma_s}')    #_cpu{ncpu}
savepath='./%s/%s' % (storagefolder ,scenario)
evpath='./%s/%s/eval' % (storagefolder ,scenario)
prevpath='./%s/%s/eval/prev' % (storagefolder ,scenario)
#savepath='%s/environment' % (savepath)



#create learning curve plot for training


numprevs=len([name for name in os.listdir(prevpath) if os.path.isfile(os.path.join(prevpath, name))])

#joined_results=np.empty((0,100,1))
#joined_timesteps=np.empty((0,))

for loadid in range(1,numprevs):

    data=np.load("%s/evaluations_%s.npz"% (prevpath, loadid))
    
    results=data['results']
    timesteps=data['timesteps']
    
    if loadid==1:
        joined_results=results
        joined_timesteps=timesteps
    else:
        joined_results=np.concatenate((joined_results, results))
        joined_timesteps=np.concatenate((joined_timesteps, timesteps+joined_timesteps[-1]))
    
    
    
    
    

#data=np.load(evaluations)
#results=data['results']
evaluations_current= './%s/%s/eval/evaluations.npz' % (storagefolder,scenario)

data=np.load(evaluations_current)
results=data['results']

joined_results=np.concatenate((joined_results, results))

timesteps=data['timesteps']

joined_timesteps=np.concatenate((joined_timesteps, timesteps+joined_timesteps[-1]))


y=np.average(joined_results, axis=1)
#timesteps=data['timesteps']
plt.plot(joined_timesteps,y)

plt.xlabel('Timesteps')
plt.ylabel('Score')
#plt.show() 

#save learning curve plot
figsavepath='./%s/%s/joinfig_%s' % (storagefolder ,scenario, scenario)
plt.savefig(figsavepath)
#plt.clf() #clear plot

# #create learning curve plot for evaluation
# evaluations='./%s/evaluations.npz' % (evpath)
# data=np.load(evaluations)
# results=data['results']
# y=np.average(results, axis=1)
# timesteps=data['timesteps']
# plt.plot(timesteps,y)

# plt.xlabel('Timesteps')
# plt.ylabel('Score')
# #plt.show() 

# #save learning curve plot
# figsavepath='./%s/%s/evfig_%s' % (storagefolder ,scenario, scenario)
# plt.savefig(figsavepath)    



# def save_evals():
    
#     tr= './%s/%s/evaluations.npz' % (storagefolder,scenario)
#     ev='./%s/evaluations.npz' % (evpath)
#     prevtr= './%s/%s/prev' % (storagefolder,scenario)
#     prevev='./%s/prev' % (evpath)

#     if (os.path.exists(prevtr)!=True):
#         folder='./%s/%s/prev' %(storagefolder,scenario)    
#         os.mkdir(folder)    
        
#     try:
#         savenumber=len([name for name in os.listdir(prevtr) if os.path.isfile(os.path.join(prevtr, name))])+1
#     except:
#         savenumber=1
    
#     if (os.path.exists(tr)==True):
#         dest_dir = prevtr
#         new_name = 'evaluations_%s.npz' % savenumber
#         current_file_name = tr
#         os.rename(current_file_name, os.path.join(dest_dir, new_name))

#     if (os.path.exists(prevev)!=True):
#         folder='./%s/prev' % (evpath)    
#         os.mkdir(folder)      
        
#     try:   
#         savenumber=len([name for name in os.listdir(prevev) if os.path.isfile(os.path.join(prevev, name))])+1
#     except:
#         savenumber=1
    
#     if (os.path.exists(ev)==True):
#         dest_dir = prevev
#         new_name = 'evaluations_%s.npz' % savenumber
#         current_file_name = ev
#         os.rename(current_file_name, os.path.join(dest_dir, new_name))


# class TimeLimit(BaseCallback):
#     """
#     Callback for saving a model (the check is done every ``check_freq`` steps)
#     based on the training reward (in practice, we recommend using ``EvalCallback``).

#     :param check_freq: (int)
#     :param log_dir: (str) Path to the folder where the model will be saved.
#       It must contains the file created by the ``Monitor`` wrapper.
#     :param verbose: (int)
#     """
#     def __init__(self, episodetimesteps):
#         super(TimeLimit, self).__init__()
#         self.check_freq = episodetimesteps
#         self.incomplete = True
#         self.starttime=time.time()
#         self.prev=1
        
#     def _on_step(self) -> bool:
#         if self.n_calls % self.check_freq == 0:
#             if time.time()<end:
#                 self.incomplete = True
#             else:
#                 model.save("%s/final_model" % savepath)
#                 self.incomplete = False
#                 trainingplot()
                
#             if np.ceil((time.time() - self.starttime)/(60*60*12))-self.prev>=1: #every 12 hours make plot. previous difference will =1
#                 trainingplot()
                
#             self.prev=np.ceil((time.time() - self.starttime)/(60*60*12)) 
#         return self.incomplete

# def make_env(x,y,z, rank, seed=0):
#     """
#     Utility function for multiprocessed env.

#     :param env_id: (str) the environment ID
#     :param num_env: (int) the number of environments you wish to have in subprocesses
#     :param seed: (int) the inital seed for RNG
#     :param rank: (int) index of the subprocess
#     """
#     def _init():
        
#         env = environment(x, y, z, gamma, turnspc, policyname)
#         env.seed(seed + rank)
#         return env
#     set_global_seeds(seed)
#     return _init


# if __name__ == '__main__':

#     num_cpu = ncpu  # Number of processes to use
#     # Create the vectorized environment
#     env = SubprocVecEnv([make_env(x,y,z, i) for i in range(num_cpu)])
#     eval_env=evalenv(x, y, z, gamma, turnspc, policyname)
#     env1 =environment(x, y, z, gamma, turnspc, policyname) #env annealreate/ numturns*eval_freq
#     # Stable Baselines provides you with make_vec_env() helper
#     # which does exactly the previous steps for you:
#     # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)

    
#     #create callbacks to record data, initiate events during training.
#     callbacklist=CallbackList([TimeLimit(episodetimesteps), EvalCallback(eval_env, log_path=evpath, n_eval_episodes=100, eval_freq=50000
#                                                                          , deterministic=False, best_model_save_path=evpath), EvalCallback(env1, log_path=savepath, n_eval_episodes=20, eval_freq=50000
#                                                                          , deterministic=False, best_model_save_path=savepath)])
#     if (os.path.exists("%s/best_model.zip" % savepath)):
#         # Instantiate the agent
#         model = A2C(policy, env, gamma=gamma, n_steps=episodetimesteps, learning_rate=LR,  verbose=1, n_cpu_tf_sess=num_cpu)
#         # Load the trained agent
#         model = A2C.load("%s/best_model" % savepath, env=env)
#         print('loaded agent')
#         save_evals()
        
#         model.learn(total_timesteps=episodetimesteps**50, callback=callbacklist) #total timesteps set to very large number so program will terminate based on runtime parameter)
        
        
#     else:
#         #create model with Stable Baselines package.
#         model = A2C(policy, env, gamma=gamma, n_steps=episodetimesteps, learning_rate=LR,  verbose=1, n_cpu_tf_sess=num_cpu)#, tensorboard_log=scenario)
#         #model = ACER.load("%s/best_model" % savepath, env)
#         save_evals()
        
#         model.learn(total_timesteps=episodetimesteps**50, callback=callbacklist)  #total timesteps set to very large number so program will terminate based on runtime parameter)
            
    

    
    
    
    