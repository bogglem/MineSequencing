# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 19:58:48 2022

@author: Tim Pelech
"""
import numpy as np
import matplotlib.pyplot as plt

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ma=ret[n - 1:] / n
    
    var=np.zeros(len(ma))
    for i in range(len(ma)):
        var[i]=np.var(a[i:i+n])
    
    return ma, var



def smoothplot(x, y, n):
    
    
    ma,var=moving_average(y,n)
    t=x[n-1:]
    fig, ax = plt.subplots(1)
    ax.plot(t, ma, lw=2, label='mean population 1')
    ax.fill_between(t, ma+var, ma-var, alpha=0.4) #facecolor='C0'

