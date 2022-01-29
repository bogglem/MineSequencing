# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 13:28:51 2021

@author: Tim Pelech
"""

#filter tensorflow warnings
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

import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines import A2C
from stable_baselines import DQN
from stable_baselines import ACER
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.evaluation import evaluate_policy
#from tools.BMenv import environment
#from tools.loadBMenv import environment
#from tools.SingleBMenv_curricturnspc import environment
from tools.BMenv import environment
#from tools.Fuzzy3DBMenv_9action import environment

# # Create environment
# x=15
# y=15
# z=4
# batch_size=64
# LR=0.0005
# gamma=0.99
# turnspc=0.10
# episodetimesteps=round(x*y*z*turnspc)

# policyname='MlpPolicy' #change this name to change RL policy type (MlpPolicy/CnnPolicy)

# if policyname == 'CnnPolicy':
    
#     policy=CnnPolicy
#     test='CNNA2C'

# elif policyname =='MlpPolicy':

#     policy=MlpPolicy
#     test='MLPA2C'

# trialv='loadsave10'

# #prepare file naming strings
# LR_s=str("{:f}".format(LR)).split('.')[1]
# inputfile_s='%s_%s_%s' % (x,y,z)
# gamma_s=str(gamma).replace('.','_')
# #cutoff_s=str(cutoffpenaltyscalar).split('.')[0]
# #rg_s=max(str(float(rg_prob)).split('.'))
# turnspc_s=str(turnspc).split('.')[1]

# scenario=str(f'{trialv}_{inputfile_s}_t{test}_lr{LR_s}_g{gamma_s}')  
# savepath='./output/%s' % scenario

# turns=round(x*y*z*turnspc)

# env = environment(x,y,z,gamma, turnspc, policyname)

# # Instantiate the agent
# model = A2C(policy, env, gamma=gamma, learning_rate=LR,n_steps=episodetimesteps,   verbose=1)
# #model = DQN('MlpPolicy', env, learning_rate=LR, prioritized_replay=True, verbose=1)
# #
# # Load the trained agent
# model = A2C.load("%s/best_model" % savepath)
# #model = DQN.load("%s/best_model" % savepath)
# print('loaded agent %s' % savepath)

# # Evaluate the agent
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, deterministic=False)
# print('mean_reward = %s +/- %s' %(mean_reward,std_reward))

# # Enjoy trained agent
# obs = env.reset()
# env.rendermode='on'
# cumreward=0
# results=list()
# for i in range(turns):
#     action, _states = model.predict(obs, deterministic=False)
#     obs, rewards, dones, info = env.step(action)
#     cumreward+=rewards
#     print(action, rewards, dones, cumreward)
#     results.append(info)
#     #env.renderif('on')
#     if dones == True:
#         break

class plotresults():
           
    def singleplot(results, labels, geo_array):
        
        resultsarray=np.array(results)
        grades=geo_array[:,:,:,0]
        gradesf=np.ndarray.flatten(grades)
        avgrade=np.average(grades)
        
        fig1=plt.figure(1)
        plt.hist(gradesf,40)
        plt.axvline(np.average(resultsarray[0:20]), color='g', linestyle='dashed', linewidth=1)
        plt.axvline(np.average(resultsarray[20:40]), color='r', linestyle='dashed', linewidth=1)
        plt.axvline(np.average(resultsarray), color='k', linestyle='dashed', linewidth=1)
        min_ylim, max_ylim = plt.ylim()
        
        plt.text(np.average(resultsarray[0:20])*1.05, max_ylim*0.4, '0-20 Grade: {:.3f}'.format(np.average(resultsarray[0:20])), rotation=45, color='g')
        plt.text(np.average(resultsarray[20:40])*1.05, max_ylim*0.6, '20-40 Grade: {:.3f}'.format(np.average(resultsarray[20:40])), rotation=45, color='r')
        plt.text(np.average(resultsarray)*1.05, max_ylim*0.8, 'Total Mined Grade: {:.3f}'.format(np.average(resultsarray)), rotation=45)
        
        plt.figure(2)
        plt.plot(resultsarray)    
        plt.axhline(np.max(resultsarray), color='k', linestyle='dashed', linewidth=1)
        min_xlim, max_xlim = plt.xlim()
        plt.text(max_xlim*0.7, np.max(resultsarray)*0.9, 'Max Grade: {:.3f}'.format(np.max(resultsarray)))
        plt.text(max_xlim*0.4, np.max(resultsarray)*0.9, labels)

        
    def subplot(results, labels, geo_array):
       
        fig1, (f1ax1, f1ax2) = plt.subplots(1,2)
        fig2, (f2ax1, f2ax2) = plt.subplots(2)        
        
        resultsarray1=np.array(results[0])
        resultsarray2=np.array(results[1])
        grades=geo_array[:,:,:,0]
        gradesf=np.ndarray.flatten(grades)
        avgrade=np.average(grades)

        labels1, labels2 = labels
        
        #Histogram 1
        
        #fig1=plt.figure(1)
        f1ax1.hist(gradesf,40)
        f1ax1.axvline(np.average(resultsarray1[0:20]), color='g', linestyle='dashed', linewidth=1)
        f1ax1.axvline(np.average(resultsarray1[20:40]), color='r', linestyle='dashed', linewidth=1)
        f1ax1.axvline(np.average(resultsarray1), color='k', linestyle='dashed', linewidth=1)
        min_ylim, max_ylim = f1ax1.get_ylim()
        min_xlim, max_xlim = f1ax1.get_xlim()
        
        # f1ax1.text(np.average(resultsarray1[0:20])*1.05, max_ylim*0.4, '0-20 Grade: {:.3f}'.format(np.average(resultsarray1[0:20])), rotation=45, color='g')
        # f1ax1.text(np.average(resultsarray1[20:40])*1.05, max_ylim*0.6, '20-40 Grade: {:.3f}'.format(np.average(resultsarray1[20:40])), rotation=45, color='r')
        # f1ax1.text(np.average(resultsarray1)*1.05, max_ylim*0.8, 'Total Mined Grade: {:.3f}'.format(np.average(resultsarray1)), rotation=45)
        
        f1ax1.text(max_xlim*0.35, max_ylim*0.4, '0-20 Gr: {:.3f}'.format(np.average(resultsarray1[0:20])), rotation=0, color='g')
        f1ax1.text(max_xlim*0.35, max_ylim*0.5, '20-40 Gr: {:.3f}'.format(np.average(resultsarray1[20:40])), rotation=0, color='r')
        f1ax1.text(max_xlim*0.35, max_ylim*0.6, 'Total Gr: {:.3f}'.format(np.average(resultsarray1)), rotation=0)
        f1ax1.text(max_xlim*0.4, max_ylim*1.02, labels1, rotation=0)    
        
        #Histogram 2
        
        #fig1=plt.figure(1)
        f1ax2.hist(gradesf,40)
        f1ax2.axvline(np.average(resultsarray2[0:20]), color='g', linestyle='dashed', linewidth=1)
        f1ax2.axvline(np.average(resultsarray2[20:40]), color='r', linestyle='dashed', linewidth=1)
        f1ax2.axvline(np.average(resultsarray2), color='k', linestyle='dashed', linewidth=1)
        min_ylim, max_ylim = f1ax2.get_ylim()
        min_xlim, max_xlim = f1ax2.get_xlim()
        
        # f1ax2.text(np.average(resultsarray2[0:20])*1.05, max_ylim*0.4, '0-20 Grade: {:.3f}'.format(np.average(resultsarray2[0:20])), rotation=45, color='g')
        # f1ax2.text(np.average(resultsarray2[20:40])*1.05, max_ylim*0.6, '20-40 Grade: {:.3f}'.format(np.average(resultsarray2[20:40])), rotation=45, color='r')
        # f1ax2.text(np.average(resultsarray2)*1.05, max_ylim*0.8, 'Total Mined Grade: {:.3f}'.format(np.average(resultsarray2)), rotation=45)
        f1ax2.text(max_xlim*0.35, max_ylim*0.4, '0-20 Gr: {:.3f}'.format(np.average(resultsarray2[0:20])), rotation=0, color='g')
        f1ax2.text(max_xlim*0.35, max_ylim*0.5, '20-40 Gr: {:.3f}'.format(np.average(resultsarray2[20:40])), rotation=0, color='r')
        f1ax2.text(max_xlim*0.35, max_ylim*0.6, 'Total Gr: {:.3f}'.format(np.average(resultsarray2)), rotation=0)       
        f1ax2.text(max_xlim*0.45, max_ylim*1.02, labels2, rotation=0)     
        
        #Grade sequence 1
    
        #plt.figure(2)
        f2ax1.plot(resultsarray1)    
        f2ax1.axhline(np.max(resultsarray1), color='k', linestyle='dashed', linewidth=1)
        min_xlim, max_xlim = f2ax1.get_xlim()
        min_ylim, max_ylim = f2ax1.get_ylim()
        f2ax1.text(max_xlim*0.7, np.max(resultsarray1)*0.9, 'Max Grade: {:.3f}'.format(np.max(resultsarray1)))
        f2ax1.text(max_xlim*0.05, max_ylim*0.5, labels1, rotation=0)     

        #Grade sequence 2
   
        #plt.figure(2)
        f2ax2.plot(resultsarray2)    
        f2ax2.axhline(np.max(resultsarray2), color='k', linestyle='dashed', linewidth=1)
        min_xlim, max_xlim = f2ax2.get_xlim()
        min_ylim, max_ylim = f2ax2.get_ylim()
        f2ax2.text(max_xlim*0.7, np.max(resultsarray2)*0.9, 'Max Grade: {:.3f}'.format(np.max(resultsarray2)))    
        f2ax2.text(max_xlim*0.05, max_ylim*0.5, labels2, rotation=0)
    