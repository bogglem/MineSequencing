# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 12:04:45 2020

@author: Tim Pelech
"""
import numpy as np

class orebody:

    def __init__(self):
        self.ilen=3
        self.jlen=3 
        self.klen=3 #0 is bottom layer
    
        self.orebody=np.array([self.ilen,self.jlen,self.klen])
        self.idxbody=np.array([self.ilen,self.jlen,self.klen])
        self.block_dic={}
        self.dep_dic={}
        
        
    def construct(self):
        
        for i in self.ilen-1:
            for j in self.jlen-1:
                for k in self.klen-1:
                    block=str(i)+str('_')+str(j)+str('_')+str(k)
                    self.block_dic["id_%s"% block]=1
                    #id_i_j_k=str('id','_',i,'_',j,'_',k)
                                      
                    
                    if k-self.klen+1==0: #if block is surface layer, then no dependency exists
                        #self.orebody[i,j,k]=list(id_i_j_k)
                        break
                    
                    else:
                         "\ dep /"
                          "\ore/"
                        dep0=str(i-1)+str('_')+str(j+1)+str('_')+str(k+1)
                        dep1=str(i)+str('_')+str(j+1)+str('_')+str(k+1)
                        dep2=str(i+1)+str('_')+str(j+1)+str('_')+str(k+1)
                        dep3=str(i-1)+str('_')+str(j)+str('_')+str(k+1)
                        dep4=str(i)+str('_')+str(j)+str('_')+str(k+1)
                        dep5=str(i+1)+str('_')+str(j)+str('_')+str(k+1)
                        dep6=str(i-1)+str('_')+str(j-1)+str('_')+str(k+1)
                        dep7=str(i)+str('_')+str(j-1)+str('_')+str(k+1)
                        dep8=str(i+1)+str('_')+str(j-1)+str('_')+str(k+1)
                        
                        dep=list([dep0,dep1,dep2,dep3,dep4,dep5,dep6,dep7,dep8])
                        self.dep_dic["id_%s_dep"% block]=dep
                           
    
    def mine_block(self, i, j):
    
        k= #iterate through orebody at action location to find highest unmined block
        
        selected_block = str(i)+str('_')+str(j)+str('_')+str(k)
        self.block_dic["id_%s"% selected_block]=0
        
        return mine_block_str #return string name of mined block
        
        
    
    def isMinable(self, i,j,k):
        
        #self.orebody[i,j,k][0]=1
        selected_block = str(i)+str('_')+str(j)+str('_')+str(k)
        available = self.block_dic["id_%s"% selected_block]
        deplist = self.dep_dic["id_%s"% selected_block]
        minablelogic=np.zeros(len(deplist))            
               
        for d in range(len(deplist)):
            depstr=deplist[d]
            minablelogic[d]=self.block_dic["id_%s"% depstr]
            isMinable=available*np.prod(minablelogic)
            
    return isMinable



    def blockvalue(self, mine_block_str, isMineable):
        
        

        
        
                 
                    
                    
                    
        