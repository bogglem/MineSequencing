# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 09:05:54 2021

@author: Tim Pelech
"""

#import modules
import os
import time
import gym
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import SubprocVecEnv
# from stable_baselines.common import set_global_seeds, make_vec_env
# from stable_baselines.common.callbacks import BaseCallback, CallbackList, EvalCallback
# from stable_baselines import ACER
# from tools.loadsaveBMenv import environment
# from tools.evalBMenv import environment as evalenv

#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

#prepare input parameters
inputarray=pd.read_csv('jobarrays/LR_job_input.csv')

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ma=ret[n - 1:] / n
    
    var=np.zeros(len(ma))
    for i in range(len(ma)):
        var[i]=np.var(a[i:i+n])
    
    return ma, var

# def plot(x, y, n):
    
#     ma,var=moving_average(y,n)
#     t=x[n-1:]
#     fig, ax = plt.subplots(1)
#     ax.plot(t, ma, lw=2, label='mean population 1')
#     ax.fill_between(t, ma+var, ma-var, alpha=0.4) #facecolor='C0'


for t in range(len(inputarray)):
    
    idx=t#int(sys.argv[1]) #array row number. required for batch runs on pbs katana
    #idx=5
    try:
    
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
        #cutoffpenaltyscalar=inputarray.loc[idx].cutoffpenaltyscalar #not currently implemented
        #rg_prob=inputarray.loc[idx].rg_prob
        turnspc=inputarray.loc[idx].turnspc
        ncpu=inputarray.loc[idx].ncpu
        equipf=inputarray.loc[idx].equipf
        
        start=time.time()
        end=start+runtime
        episodetimesteps=round(x*y*z*turnspc)
        
        #prepare file naming strings
        LR_s=str("{:f}".format(LR)).split('.')[1]
        inputfile_s='%s_%s_%s' % (x,y,z)
        gamma_s=str(gamma).replace('.','_')
        scalar_s=str(equipf).replace('.','_')
        #cutoff_s=str(cutoffpenaltyscalar).split('.')[0]
        #rg_s=rg_prob #max(str(float(rg_prob)).split('.'))
        turnspc_s=str(turnspc).split('.')[1]
        storagefolder='output'
        scenario=str(f'{trialv}_{inputfile_s}_t{test}_lr{LR_s}_g{gamma_s}_f{scalar_s}')    #_s{scalar_s}
        savepath='./%s/%s' % (storagefolder ,scenario)
        evpath='./%s/%s/eval' % (storagefolder ,scenario)
        #savepath='%s/environment' % (savepath)
        
        evaluations='./%s/evaluations.npz' % (evpath)
        data=[]
        data=np.load(evaluations)
        results=[]
        results=data['results']
        y=np.average(results, axis=1)
        timesteps=[]
        timesteps=data['timesteps']
        label='Failure Probability %s' % equipf
        
        n=25 #moving average points
        ma,var=moving_average(y,n)
        t=timesteps[n-1:]
        #fig, ax = plt.subplots(1)
        plt.plot(t, ma, lw=2, label=label)
        plt.fill_between(t, ma+var, ma-var, alpha=0.2)
        
        #plt.plot(timesteps,y, label=label, linewidth=1.2)
        plt.legend(loc='lower right')
        
        title="A2C Equipment Failure Parameter Testing"
        plt.title(title)
        plt.xlabel('Timesteps')
        plt.ylabel('Evaluation Score')
        #plt.show() 
        
        
    except:
        break

title="A2C Equipment Failure Parameter Testing"      
#save learning curve plot
figsavepath='./%s/%s' % (storagefolder, title)
plt.savefig(figsavepath, dpi=300)
#plt.clf()
    
