# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 13:11:40 2022

@author: Tim Pelech
"""
from tools.humanBMenv import environment

import numpy as np


class human():

    traj = np.genfromtxt('HumanTrajectory.csv', delimiter=',')
    
    env = environment(15, 15, 4, 0.9, 0.1, 'MlpPolicy')
    
    output=np.zeros(len(traj))
    counter=0
    
    for s in traj:
        
        a,b,c,info = env.step(int(s))
        
        env.render()
        
        output[counter]=info
        
        counter+=1
    


class ai():
    
    env = environment(15, 15, 4, 0.9, 0.1, 'MlpPolicy')   
    output=np.zeros(len(traj))
    counter=0
    
    
    