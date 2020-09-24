# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 15:53:06 2020

@author: Tim Pelech
"""
import time
import numpy as np

tic=time.time()


a=np.zeros([100,100,100])

i=0
j=0
k=0

for i in range(99):
    for j in range(99):
        for k in range(99):
            
            b=a[i,j,k]


toc=time.time()


print(toc-tic)