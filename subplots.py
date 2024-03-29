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
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines import ACER
# from tools.loadsaveBMenv import environment
# from tools.evalBMenv import environment as evalenv

#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

#prepare input parameters
inputarray=pd.read_csv('jobarrays/A2C_katana_job_input.csv')

#initate subplots

l=len(inputarray)
cols=2
rows=int(np.ceil(l/cols))
fig, axes = plt.subplots(rows, cols, figsize=(12,12))
#fig.tight_layout()


# """ Iterate column's axes"""
# def subplot(cols, x,y, label):
#     for col in cols:
#         col
idx=0

for ax in axes.flatten():

    #idx=t#int(sys.argv[1]) #array row number. required for batch runs on pbs katana
    #idx=5
    
    #subplot coords
    # if t%2==1:
    #     col=1
    # else:
    #     col=0
    
    # if t%2==0:
    #     row=int(t/2)  
    # else:
    #     row=int((t-1)/2)
    
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
        scalar=inputarray.loc[idx].scalar
        
        start=time.time()
        end=start+runtime
        episodetimesteps=round(x*y*z*turnspc)
        
        #prepare file naming strings
        LR_s=str("{:f}".format(LR)).split('.')[1]
        inputfile_s='%s_%s_%s' % (x,y,z)
        gamma_s=str(gamma).replace('.','_')
        scalar_s=str(scalar).replace('.','_')
        #cutoff_s=str(cutoffpenaltyscalar).split('.')[0]
        #rg_s=rg_prob #max(str(float(rg_prob)).split('.'))
        turnspc_s=str(turnspc).split('.')[1]
        storagefolder='output'
        scenario=str(f'{trialv}_{inputfile_s}_t{test}_lr{LR_s}_g{gamma_s}_s{scalar_s}') #   
        savepath='./%s/%s' % (storagefolder ,scenario)
        evpath='./%s/%s/eval' % (storagefolder ,scenario)
        #savepath='%s/environment' % (savepath)
        
        evaluations='%s/evaluations.npz' % (evpath)
        data=[]
        data=np.load(evaluations)
        results=[]
        results=data['results']
        y=np.average(results, axis=1)
        timesteps=[]
        timesteps=data['timesteps']
        label='Scalar %s' % scalar
        #plt.plot(timesteps,y, label=label, linewidth=1.2)
       # plt.legend()
        
        #
        #plt.xlabel('Timesteps')
        #plt.ylabel('Evaluation Score')
        #plt.show() 
        
        
        """ Iterate row's axes"""
        #subplot(row, col, timesteps, y, label)
        #ax=axes[row,col]
        ax.set_ylim([-30,10])
        ax.set_xlim([0,7e7])
           
        ax.grid(axis='y')
        ax.plot(timesteps, y, label=label, linewidth=1.2)    
        ax.legend(loc='lower right')
        #fig.title("A2C Learning Rate Tuning on Eval Environment Set")
        
        idx+=1
        
        
    except:
        break
        
fig.subplots_adjust(hspace = .3)     
fig.text(0.51, 0.91, 'A2C Penalty Scalar Tuning', ha='center', va='center', fontsize='xx-large')
fig.text(0.51, 0.08, 'Timesteps', ha='center', va='center', fontsize='xx-large')
fig.text(0.08, 0.5, 'Evaluation Score', ha='center', va='center', rotation='vertical', fontsize='xx-large')


#save learning curve plot
figsavepath='./%s/%s_%s Tuning_subplots' % (storagefolder, test, trialv)
plt.savefig(figsavepath, dpi=400)
#plt.clf()
    
