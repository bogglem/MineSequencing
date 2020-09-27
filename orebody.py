# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 12:04:45 2020

@author: Tim Pelech
"""
import numpy as np
import pandas as pd

class orebody:

    def __init__(self):
        self.ilen=3
        self.jlen=3 
        self.klen=3 #0 is bottom layer
    
        self.orebody=np.array([self.ilen,self.jlen,self.klen])
        self.idxbody=np.array([self.ilen,self.jlen,self.klen])
        orebodydf=pd.DataFrame()
        orebodydf.columns(id_i_j_k, dep0,dep1,dep2,dep3,dep4,dep5,dep6,dep7,dep8)
        
        
    def construct(self):
        block_dic = {}
        dep_dic={}
        
        for i in self.ilen-1:
            for j in self.jlen-1:
                for k in self.klen-1:
                    s=str(i)+str('_')+str(j)+str('_')+str(k)
                    block_dic["id_%s"% s]=1
                    #id_i_j_k=str('id','_',i,'_',j,'_',k)
                                      
                    
                    if k-self.klen+1==0:
                    
                        self.orebody[i,j,k]=list(id_i_j_k)
                        break
                    
                    else:
                        
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
                        dep_dic["id_%s_dep"% s]=dep
                        
                    #df=pd.DataFrame(list([id_i_j_k, dep0,dep1,dep2,dep3,dep4,dep5,dep6,dep7,dep8]))
                    #orebodydf.append(df)
                    #self.orebody[i,j,k]=list([id_i_j_k, dep0,dep1,dep2,dep3,dep4,dep5,dep6,dep7,dep8])
        
    
    
    def mine_block(self, i, j, k):
    
        current_block = str(i)+str('_')+str(j)+str('_')+str(k)
        block_dic["id_%s"% current_block]=0
           

        
    
    def isMinable(self, i,j,k):
        
        #self.orebody[i,j,k][0]=1
        current_block = str(i)+str('_')+str(j)+str('_')+str(k)
        available = block_dic["id_%s"% current_block]
        deplist = dep_dic["id_%s"% current_block]
        minablelogic=np.zeros(len(deplist))            
               
        for d in range(len(deplist)):
            depstr=deplist[d]
            
            minablelogic[d]=block_dic["id_%s"% depstr]
            isMinable=available*np.prod(minablelogic)
            
    return isMinable
                    

    def minableRL(self,i,j):
        
        if 
        
        else:
            k=-1
        
        return k
        
        
                 
                    
                    
                    
        