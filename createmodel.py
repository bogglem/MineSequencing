# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 14:08:08 2020

@author: Tim Pelech
"""
import numpy as np
import pandas as pd
import random


_I=15
_J=15
RL=5
index=0

names=np.array(['_I','_J','RL','H2O','Tonnes'])

data=pd.DataFrame(columns=names)

for i in range(_I-1):
    for j in range(_J-1):
        for k in range(RL-1):
            dic={}
            
            if (10>i>5)&(10>j>5):
                
                H2O=(random.randrange(10,40)/100)*0.5*(RL+0.001)
            
            else: 
                H2O=(random.randrange(1,10)/100)*0.5*(RL+0.001)
            
                
            dic={'_I':i+1,'_J':j+1,'RL':k+1,'H2O':H2O,'Tonnes':10}
            data = data.append(dic, True)
            
            
            
data.to_excel("BM_central15x15x5.xlsx")

            
            
            
                
        
    
