# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 11:14:46 2021

@author: Tim Pelech
"""
import os
import time
import gym
import numpy as np
import sys
import pandas as pd
from tools.BMenv import environment

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

idx=0

#prepare input parameters
inputarray=pd.read_csv('jobarrays/Fuzzy_drstrange_generateenv_input.csv')

#block model (environment) dimensions
x=inputarray.loc[idx].x
y=inputarray.loc[idx].y
z=inputarray.loc[idx].z #must be greater than 6 for CNN

policyname=inputarray.loc[idx].policyname  #change this name to change RL policy type (MlpPolicy/CnnPolicy)

if policyname == 'CnnPolicy':
    
    #policy=CnnPolicy
    test='CNNACER'

elif policyname =='MlpPolicy':

    #policy=MlpPolicy
    test='MLPACER'

trialv=inputarray.loc[idx].trialv 
LR=inputarray.loc[idx].LR
gamma=inputarray.loc[idx].gamma
runtime=inputarray.loc[idx].runtime
#cutoffpenaltyscalar=inputarray.loc[idx].cutoffpenaltyscalar #not currently implemented
#rg_prob=inputarray.loc[idx].rg_prob
turnspc=inputarray.loc[idx].turnspc

start=time.time()
end=start+runtime
episodetimesteps=round(x*y*z*turnspc)

#prepare file naming strings
LR_s=str(LR).split('.')[1]
inputfile_s='%s_%s_%s' % (x,y,z)
gamma_s=str(gamma).split('.')[1]
#cutoff_s=str(cutoffpenaltyscalar).split('.')[0]
#rg_s=max(str(float(rg_prob)).split('.'))
turnspc_s=str(turnspc).split('.')[1]
storagefolder='environments/15x15x4'
scenario=str(f'{trialv}_{inputfile_s}_t{test}_lr{LR_s}_{policyname}')
savepath='./%s' % (storagefolder)
#savepath='%s/environment' % (savepath)


env = environment(x,y,z,gamma, turnspc, policyname, envpath=savepath)


for i in range(3000):
 
    env.build()
    env.save()

    
    
    
    
    