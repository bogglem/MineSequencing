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

#class plotresults():
    
    
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ma=ret[n - 1:] / n
    
    var=np.zeros(len(ma))
    for i in range(len(ma)):
        var[i]=np.var(a[i:i+n])

    return ma, var

       
def singleplot(results, labels, geo_array, minable):
    
    resultsarray=np.array(results)*100
    resultsarray_cutoff=resultsarray[resultsarray>0.0]
    grades=geo_array[:,:,:,0]*100
    gradesf=np.ndarray.flatten(grades)
    avgrade=np.average(grades)
    cumresults=np.cumsum(resultsarray_cutoff)
    
    minablearray=np.array(minable)*np.average(resultsarray)
    maskedminablearray = np.ma.masked_where(minablearray==0, minablearray)
    
    #grade histogram
    
    fig1=plt.figure(1)
    plt.hist(gradesf,40)
    plt.axvline(np.average(resultsarray_cutoff[0:20]), color='g', linestyle='dashed', linewidth=1)
    plt.axvline(np.average(resultsarray_cutoff[20:40]), color='r', linestyle='dashed', linewidth=1)
    plt.axvline(np.average(resultsarray_cutoff), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    
    plt.text(np.average(resultsarray_cutoff[0:20])*1.05, max_ylim*0.4, '0-20 Grade: {:.1f}'.format(np.average(resultsarray_cutoff[0:20])), rotation=45, color='g')
    plt.text(np.average(resultsarray_cutoff[20:40])*1.05, max_ylim*0.6, '20-40 Grade: {:.1f}'.format(np.average(resultsarray_cutoff[20:40])), rotation=45, color='r')
    plt.text(np.average(resultsarray_cutoff)*1.05, max_ylim*0.8, 'Total Mined Grade: {:.1f}'.format(np.average(resultsarray_cutoff)), rotation=45)
    
    #Sequence Plot
    
    plt.figure(2)
    plt.plot(resultsarray)    
    plt.plot(maskedminablearray, color='r',linestyle='none', marker='x', label='Sequence Error')
    plt.axhline(np.max(resultsarray), color='k', linestyle='dashed', linewidth=1)
    min_xlim, max_xlim = plt.xlim()
    plt.text(max_xlim*0.7, np.max(resultsarray)*0.9, 'Max Grade: {:.1f}'.format(np.max(resultsarray)))
    plt.text(max_xlim*0.4, np.max(resultsarray)*0.9, labels)
    
    plt.figure(3)
   # plt.text(0.51, 0.02, 'Timestep', ha='center', va='center', fontsize='large')
   # plt.text(0.02, 0.5, 'Cumulative H2O Extracted (kg)', ha='center', va='center', rotation='vertical', fontsize='large')
    
    #plt.figure(2)
    plt.plot(cumresults, label=labels)    
    #min_xlim, max_xlim = plt.get_xlim()
    #min_ylim, max_ylim = plt.get_ylim()
   # plt.text(max_xlim*0.31, max_ylim*1.05, 'Cumulative Grade', rotation=0)


# def variance(store):
    
#     storea=np.array(store)
#     x=len(storea[:,0])
#     maxi=max(storea[:,2])
#     duplicates=x/maxi
#     cumresults_cutoff=list() 
    
#     for i in range(int(x)):
#         cumresults_cutoff.append(np.cumsum(storea[i][5],0))
           
           

   
 
def batchplot(storearray, labels):
    
    # x,tests=store.shape
    # maxi=np.max(store[:,2])
    
    # duplicates=x/maxi
    
    # for n in range(int(duplicates)):
        
    #     labels=store[int(n*maxi):int(duplicates+n*maxi),0]
    #     cutoff=store[int(n*maxi):int(duplicates+n*maxi),1]
    #     i=store[int(n*maxi):int(duplicates+n*maxi),2]
    #     #geo_array=store[int(n*maxi):int(duplicates+n*maxi),3]
    #     results=store[int(n*maxi):int(duplicates+n*maxi),4]
    #     results_cutoff=store[int(n*maxi):int(duplicates+n*maxi),5]
    #     #minable=store[int(n*maxi):int(duplicates+n*maxi),6]
        
    # for i in range(len(store)):
    #     np.var[results_cutoff[i][5]]
        #resultsarray=np.array(results)*100
        
        #dimx,dimy,dimz=storearray.shape
        
        #storearray[:,1,:],0
        fig=plt.figure(1, figsize=(6, 6), dpi=300)
        i=storearray[:,0]
        plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
        cutoff_i=max(i)
        
        
        for i in range(cutoff_i):
            
            storesliced=storearray[storearray[:,0]==i]
            t=storesliced[:,1][0]
            results = np.vstack(storesliced[:,2]).astype(np.float)
            cum=np.cumsum(results,1)
            var=np.var(cum,0)*100
            a=np.average(cum,0)*100
            
           # p=5 #moving average points
            #ma,var=moving_average(results_cutoff,p)
            #t=timesteps[n-1:]
            #fig, ax = plt.subplots(1)
            plt.plot(t, a, lw=2, label=labels[i]) #label=labels[0
            plt.fill_between(t, a+var, a-var, alpha=0.4)
    
            #plt.plot(cumresults, label=labels)    
            plt.legend(loc='lower right')
    
        fig.text(.5, .05, 'Timestep', ha='center', va='center', fontsize='large')
        fig.text(.04,.5, 'H2O Produced (kg)', ha='center', va='center', rotation='vertical', fontsize='large')
        #fig.grid()#color='gray', linestyle='-', linewidth=.4)   
    
def subplot(results, labels, geo_array, minable):
    
   # fig1, (f1ax1, f1ax2, f1ax3) = plt.subplots(3,1, figsize=(10, 10), dpi=360)
    fig1, (f1ax1, f1ax2) = plt.subplots(1,2, figsize=(6, 4), dpi=300)
    
    fig2, (f2ax1, f2ax2) = plt.subplots(2, figsize=(6, 4), dpi=300)        
    fig2.subplots_adjust(hspace = .4)     
    
    fig3, f3ax1 = plt.subplots(1, figsize=(6, 8), dpi=300)  
    
    resultsarray1=np.array(results[0])*100
    resultsarray2=np.array(results[1])*100
    
    minablearray1=np.array(minable[0])*np.average(resultsarray1)
    maskedminablearray1 = np.ma.masked_where(minablearray1==0, minablearray1)
    
    minablearray2=np.array(minable[1])*np.average(resultsarray2)
    maskedminablearray2=np.ma.masked_where(minablearray2==0, minablearray2)
    
    cumresults1=np.cumsum(resultsarray1)
    cumresults2=np.cumsum(resultsarray2)
    grades=geo_array[:,:,:,0]*100
    gradesf=np.ndarray.flatten(grades)
    avgrade=np.average(grades)

    labels1, labels2 = labels
    
    #Histogram 1
    
    fig1.text(0.54, 0.02, 'Grade %H2O', ha='center', va='center', fontsize='large')
    fig1.text(0.02, 0.5, '# of available blocks', ha='center', va='center', rotation='vertical', fontsize='large')

    
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
    
    f1ax1.text(max_xlim*0.4, max_ylim*0.4, '0-20 %H2O: {:.1f}'.format(np.average(resultsarray1[0:20])), rotation=0, color='g')
    f1ax1.text(max_xlim*0.4, max_ylim*0.5, '20-40 %H2O: {:.1f}'.format(np.average(resultsarray1[20:40])), rotation=0, color='r')
    f1ax1.text(max_xlim*0.4, max_ylim*0.6, 'Total %H2O: {:.1f}'.format(np.average(resultsarray1)), rotation=0)
    f1ax1.text(max_xlim*0.4, max_ylim*1.02, labels1, rotation=0)    
    f1ax1.grid(color='gray', linestyle='-', linewidth=.4) 
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
    f1ax2.text(max_xlim*0.4, max_ylim*0.4, '0-20 %H2O: {:.1f}'.format(np.average(resultsarray2[0:20])), rotation=0, color='g')
    f1ax2.text(max_xlim*0.4, max_ylim*0.5, '20-40 %H2O: {:.1f}'.format(np.average(resultsarray2[20:40])), rotation=0, color='r')
    f1ax2.text(max_xlim*0.4, max_ylim*0.6, 'Total %H2O: {:.1f}'.format(np.average(resultsarray2)), rotation=0)       
    f1ax2.text(max_xlim*0.45, max_ylim*1.02, labels2, rotation=0)     
    f1ax2.grid(color='gray', linestyle='-', linewidth=.4) 
    #Grade sequence 1

    fig2.text(0.51, 0.02, 'Timestep', ha='center', va='center', fontsize='large')
    fig2.text(0.02, 0.5, 'Grade Extracted %H2O', ha='center', va='center', rotation='vertical', fontsize='large')
    
    #plt.figure(2)
    f2ax1.plot(resultsarray1)
    f2ax1.plot(maskedminablearray1, color='r',linestyle='none', marker='x', label='Sequence Error')
    f2ax1.axhline(np.max(resultsarray1), color='k', linestyle='dashed', linewidth=1)
    min_xlim, max_xlim = f2ax1.get_xlim()
    min_ylim, max_ylim = f2ax1.get_ylim()
    #f2ax1.text(max_xlim*0.7, np.max(resultsarray1)*0.9, 'Max Grade: {:.1f}'.format(np.max(resultsarray1)))
    f2ax1.text(max_xlim*0.42, max_ylim*1.05, labels1, rotation=0)     

    #Grade sequence 2
   
    #plt.figure(2)
    f2ax2.plot(resultsarray2)    
    f2ax2.plot(maskedminablearray2, color='r',linestyle='none', marker='x')
    f2ax2.axhline(np.max(resultsarray2), color='k', linestyle='dashed', linewidth=1)
    min_xlim, max_xlim = f2ax2.get_xlim()
    min_ylim, max_ylim = f2ax2.get_ylim()
    #f2ax2.text(max_xlim*0.7, np.max(resultsarray2)*0.9, 'Max Grade: {:.1f}'.format(np.max(resultsarray2)))    
    f2ax2.text(max_xlim*0.42, max_ylim*1.05, labels2, rotation=0)
    
    fig2.legend(loc='lower right')
    
    #Cumulative plot 1

    fig3.text(0.51, 0.02, 'Timestep', ha='center', va='center', fontsize='large')
    fig3.text(0.02, 0.5, 'Cumulative H2O Extracted (kg)', ha='center', va='center', rotation='vertical', fontsize='large')
    
    #plt.figure(2)
    f3ax1.plot(cumresults1, label=labels1)    
    min_xlim, max_xlim = f3ax1.get_xlim()
    min_ylim, max_ylim = f3ax1.get_ylim()
    f3ax1.text(max_xlim*0.31, max_ylim*1.05, 'Cumulative Grade', rotation=0)

    #Cumulative plot 2

    #plt.figure(2)
    f3ax1.plot(cumresults2, label=labels2)    
    
    f3ax1.legend(loc='lower right')
    f3ax1.grid(color='gray', linestyle='-', linewidth=.4) 


def geohist(geo_array):

    
    fig1, (f1ax1) = plt.subplots(1,1)

    grades=geo_array[:,:,:,0]*100
    gradesf=np.ndarray.flatten(grades)
    avgrade=np.average(grades)    


    #Histogram 1
    
    fig1.text(0.54, 0.02, 'Grade %H2O', ha='center', va='center', fontsize='large')
    fig1.text(0.02, 0.5, '# of available blocks', ha='center', va='center', rotation='vertical', fontsize='large')
    
    #fig1=plt.figure(1)
    f1ax1.hist(gradesf,40)

    min_ylim, max_ylim = f1ax1.get_ylim()
    min_xlim, max_xlim = f1ax1.get_xlim()
    