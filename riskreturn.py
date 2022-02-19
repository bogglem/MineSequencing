# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 15:20:48 2022

@author: Tim Pelech
"""
import numpy as np
import matplotlib.pyplot as plt


class riskreturn():
    
    def __init__(self, failureprob, tests):
        
       # self.doall()
    
    def eplength(self,  failureprob, tests):
        
        eplength=list()
        length=list()
        
        for test in range(tests):
            i=0
            while i<90:
                
                failure=np.random.uniform()            
                
                if failure<failureprob:
                    length=i
                    i=90
                    
                i+=1  
                    
            eplength.append(length)
            
        return np.array(eplength)
    
    
    
    def accumulate(self, failurprob, resultsarray, eplength):
        
        cumresults=np.cumsum(resultsarray)
        
        accumulated=cumresults[eplength[::]]
        
        
        return accumulated
        
        
    def plothist2(self, a, b):
        
        
        plt.hist([a,b],50)
        
        
        
    def doall(self, failureprob, humanresults, airesults):
        
        
        eplength=self.eplength(failureprob, 5000)
        
        human = self.accumulate(failureprob, humanresults,eplength)
        ai = self.accumulate(failureprob, airesults, eplength)
        
        self.plothist2(human,ai)
        
        