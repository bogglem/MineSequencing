# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 12:39:28 2022

@author: Tim Pelech
"""
import numpy as np
import matplotlib.pyplot as plt
import random

class envconc(): 
    
    def __init__(self, length):
        
        self.i=0
        self.t=0
        self.s=0
        self.n=0
        self.container= np.ones(length)
        self.arecord=list() #1
        self.brecord=list() #2
        self.crecord=list() #3
        self.drecord=list() #4
        self.erecord=list() #greater than 4
                
        self.length=length
    
    def change(self, ratio):
        
        #type a = fresh environment
        #type be = used environemnt
        prob=random.random()
        
        threshold=0.00001
        
        if (prob<ratio*threshold): #every 1 00 000 steps randomly save environment 
            self.container[self.i] += 1
            self.s+=1
        if (prob<threshold): #probability of resetting environment
            self.container[np.random.randint(0,self.length-1)] = 1
            self.n+=1
            
        #elif (self.i%90==0)  and (random.random()<0.05):

    
    def step(self, ratio):
        
        self.change(ratio)
                        
       
        
        if self.t%10000000==0:   #total timesteps
             a=len(self.container[self.container==1])
             b=len(self.container[self.container==2])
             c=len(self.container[self.container>2])
             #d=len(self.container[self.container==4])
             #e=len(self.container[self.container>4])
             self.arecord.append(a)
             self.brecord.append(b)
             self.crecord.append(c)
        #self.drecord.append(b)
        #self.erecord.append(b)
        
        self.i+=1
        self.t+=1
        if self.i>=self.length: #cyclical timesteps
            self.i=0
        
        
    def show(self):
        
        x=np.array(range(len(self.arecord)))*1000000
        
        ya=np.array(self.arecord)
        yb=np.array(self.brecord)
        yc=np.array(self.crecord)
        #yd=np.array(self.brecord)
        #ye=np.array(self.brecord)
        
        
        plt.plot(x,ya, label='Unmined', linewidth=1.2)
        plt.plot(x,yb, label='Part Mined Once', linewidth=1.2)
        plt.plot(x,yc, label='Part Mined >Once', linewidth=1.2)
        #plt.plot(x,yd, label='Used Env 3', linewidth=1.2)
        #plt.plot(x,ye, label='Used Env >4', linewidth=1.2)
                
        plt.legend(loc='upper right')
        
        title="Environment types in folder"
        plt.title(title)
        plt.xlabel('Timestep')
        plt.ylabel('# of Environments in Folder')

        plt.savefig('envconcentrations2', dpi=300)
        
        
                