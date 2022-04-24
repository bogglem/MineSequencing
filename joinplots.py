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

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ma=ret[n - 1:] / n
    
    var=np.zeros(len(ma))
    for i in range(len(ma)):
        var[i]=np.var(a[i:i+n])
    
    return ma, var

#prepare input parameters
inputarray=pd.read_csv('jobarrays/general_katana_job_input_loop.csv')

for idx in range(len(inputarray)):
    #idx=2
    #block model (environment) dimensions
    x=inputarray.loc[idx].x
    y=inputarray.loc[idx].y
    z=inputarray.loc[idx].z #must be greater than 6 for CNN
    
    policyname=inputarray.loc[idx].policyname  #change this name to change RL policy type (MlpPolicy/CnnPolicy)
    architecture=inputarray.loc[idx].architecture
    gamma=inputarray.loc[idx].gamma
    
    if policyname == 'CnnPolicy':
        
        policy=CnnPolicy
        
        if architecture == 'ACER':
            test='CNNACER'
            algo='A2C Gamma %s' %gamma
        else:
            test='CNNA2C'
            algo='A2C  Gamma %s' %gamma
    
    elif policyname =='MlpPolicy':
    
        policy=MlpPolicy    
    
        if architecture == 'ACER':
            test='MLPACER'
            algo='ACER Gamma %s' %gamma
        else:
            test='MLPA2C'
            algo='A2C Gamma %s' %gamma
    trialv=inputarray.loc[idx].trialv 
    #LR_critic=inputarray.loc[idx].LR_critic
    LR=inputarray.loc[idx].LR
    #batch_size=int(inputarray.loc[idx].batch_size)
    #memcap=int(inputarray.loc[idx].memcap)
    #inputfile=inputarray.loc[idx].inputfile

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
   # plt.plot(joined_timesteps,y)
    
    plt.xlabel('Timesteps')
    plt.ylabel('Score')
    #plt.show() 
    
    #save learning curve plot
    figsavepath='./%s/%s/joinfig_%s' % (storagefolder ,scenario, scenario)
    plt.savefig(figsavepath)
    
    
    n=15 #moving average points
    ma,var=moving_average(y,n)
    t=joined_timesteps[n-1:]
    #fig, ax = plt.subplots(1)
    plt.plot(t, ma, lw=2, label=algo)
    plt.fill_between(t, ma+var, ma-var, alpha=0.4)
    
    #plt.plot(timesteps,y, label=label, linewidth=1.2)
    plt.legend(loc='lower right')
    
    title="Agent Training"
    plt.title(title)
    plt.xlabel('Timesteps')
    plt.ylabel('Evaluation Score')



title="Agent Training"      
#save learning curve plot
figsavepath='./%s/%s' % (storagefolder, title)
plt.savefig(figsavepath, dpi=300)
