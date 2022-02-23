# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 15:20:48 2022

@author: Tim Pelech
"""
import numpy as np
import matplotlib.pyplot as plt
import random

import scipy.stats as stats

class riskreturn():
    
    def __init__(self, failureprob, tests, humanresults, airesults):
        
        self.failureprob =failureprob
        
        self.tests = tests
        
        self.humanresults=humanresults
        self.airesults=airesults
        
       # self.doall()
    
    def eplength(self):
        
        eplength=list()
        length=list()
        
        for test in range(self.tests):
            i=0
            while (i<90):
                
                failure=random.random()            
                #print(failure)
                if failure<self.failureprob*i:
                    length=i
                    eplength.append(length)
                   # print('failure')
                    i=100

                else:    
                    if i==89:
                        length=i
                        eplength.append(length)
                    i+=1  
                    
                   # print(i)
            
            
        return np.array(eplength)
    
    
    
    def accumulate(self, resultsarray, eplength):
        
        cumresults=np.cumsum(resultsarray)*100
        
        accumulated=cumresults[eplength-1]
        
        
        return accumulated
        
        
    def plothist2(self, a, b, eplength):
        
        fig1, ax1 =plt.subplots(1)
        fig2, ax2 =plt.subplots(1)
        #fig2=plt.figure(2)
        
        #binwidth = 8 / 40
       #scale_factor = len(a) * binwidth
        #gaussian_kde_zi = 
        #ax.plot(x, gaussian_kde_zi(x)*scale_factor
        
        ax1.hist([a,b],20, label=['Human','AI'])
        ax1.legend(loc='upper right')
        
        kde_a = stats.gaussian_kde(a)
        kde_b = stats.gaussian_kde(b)
        
        ax2.plot(a, kde_a(a), label='KDE Human', linestyle='None', marker='x')
        ax2.plot(b, kde_b(b), label='KDE AI', linestyle='None',  marker='x')
        
        #ax2.hist([eplength],89, label=['eplength'])
        ax2.legend(loc='upper right')
        
    def plotkde(self, humansamples, aisamples):
        
        
        
       # hsamples = human#np.concatenate([np.random.normal(np.random.randint(-8, 8), size=n)*np.random.uniform(.4, 2) for i in range(4)])
         
        
        # plot the histogram
        #fig1 = plt.figure(figsize=(8, 6), dpi=)
        fig1, (f1ax1, f1ax2, f1ax3) = plt.subplots(3,1, figsize=(10, 10), dpi=360)
        fig1.subplots_adjust(hspace = .5)    
        #fig2, (f2ax1, f2ax2) = plt.subplots(2,1)
        
        h,e = np.histogram(humansamples, bins=40, density=True)

        #plt.figure(figsize=(8,6))
        f1ax1.bar(e[:-1], h, width=np.diff(e), ec='k', align='edge', label='Human Performance pdf')
        # plot the real KDE
        kde1 = stats.gaussian_kde(humansamples)
        f1ax1.plot(e, kde1.pdf(e), c='C1', lw=2, label='KDE')
        #plot CDF
        cum=np.cumsum(h*np.diff(e))
        f1ax3.plot(e[:-1],cum, label='Human Performance cdf')
        
        f1ax1.legend(loc='upper right')
       # f1ax2.legend(loc='upper right')
        
        fig1.text(0.51, 0.02, 'Probable Performance kg H2O', ha='center', va='center', fontsize='large')
        fig1.text(0.02, 0.5, 'Probability', ha='center', va='center', rotation='vertical', fontsize='large')
        
        
        ################

        h,e = np.histogram(aisamples, bins=40, density=True)
       # x = np.linspace(e.min(), e.max())
        
        #plt.figure(figsize=(8,6))
        f1ax2.bar(e[:-1], h, width=np.diff(e),color='r', ec='k', align='edge', label='AI Performance pdf')
        # plot the real KDE
        kde2 = stats.gaussian_kde(aisamples)
        f1ax2.plot(e, kde2.pdf(e), c='C2', lw=2, label='KDE')
        #plot CDF
        cum=np.cumsum(h*np.diff(e))
        f1ax3.plot(e[:-1],cum, color='r', label='AI Performance cdf')
        
        f1ax2.legend(loc='upper right')
        f1ax3.legend(loc='lower right' )
       # f1ax4.legend(loc='upper right')
        
       # ax.set_ylim([-30,10])
        xlim=800
        f1ax1.set_xlim([0,xlim])
        f1ax2.set_xlim([0,xlim])
        f1ax3.set_xlim([0,xlim])
        
        f1ax1.grid(color='gray', linestyle='-', linewidth=.4)
        f1ax2.grid(color='gray', linestyle='-', linewidth=.4)
        f1ax3.grid(color='gray', linestyle='-', linewidth=.4)
       # f1ax4.set_xlim([0,xlim])
        
        #fig2.text(0.51, 0.02, 'Probable Performance kg H2O', ha='center', va='center', fontsize='large')
        #fig2.text(0.02, 0.5, 'Probability', ha='center', va='center', rotation='vertical', fontsize='large')
        
        
        
                
    def dokde(self):
        
        
        eplength=self.eplength()
        
        
        human = self.accumulate(self.humanresults, eplength)
        ai = self.accumulate(self.airesults, eplength)
        
        # if data=='human':
        #     self.plotkde(human)
        
        # elif data =='ai':
        #     self.plotkde(ai)
            
        # else:
        self.plotkde(human, ai)
        #     self.plotkde(ai)
        

        
    def doall(self):
        
        
        eplength=self.eplength()
        
        
        human = self.accumulate(self.humanresults, eplength)
        ai = self.accumulate(self.airesults, eplength)
        
        self.plothist2(human,ai, eplength)
        
        