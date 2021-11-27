# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 14:08:08 2020

@author: Tim Pelech
"""
import numpy as np
import pandas as pd
import random


class manualmodel():
    
    #deprecated
    
    def __init__(self,ilen,jlen,depth):

        self._I=ilen
        self._J=jlen
        self.RL=depth
        
        names=np.array(['_I','_J','RL','H2O','Tonnes'])

        self.data=pd.DataFrame(columns=names)

    def create(self):
        
        for i in range(self._I):
            for j in range(self._J):
                for k in range(self.RL):
                    dic={}
                    
                    if (10>i>5)&(10>j>5):
                        
                        H2O=(random.randrange(10,40)/100)*0.5*(self.RL+0.001)
                    
                    else: 
                        H2O=(random.randrange(1,10)/100)*0.5*(self.RL+0.001)
                    
                        
                    dic={'_I':i,'_J':j,'RL':k,'H2O':H2O,'Tonnes':10}
                    self.data = self.data.append(dic, True)
                    
                    
                    
        self.data.to_excel("BM_central15x15x5.xlsx")

    
class automodel():
    
    #generate a block model based on seed blocks and interpolation.
    
    def __init__(self,ilen,jlen,depth):

        self._I=ilen
        self._J=jlen
        self.RL=depth
        self.seeds=np.zeros(1)

        
    def seedlocations(self, numseedlocations):
        
        #input parameters for seeds.
        
        mu = 0#mean location (-1-0.1)
        kappa =4 #dilation y (range 1-3)
        alpha = 20 #dilation x (range 10-20)
        self.seeds=np.zeros([numseedlocations,4])
        
        for s in range(numseedlocations):
            
            i = random.randint(2, self._I-2)
            j = random.randint(2, self._J-2)
            RL = random.randint(1, self.RL-2)
            H2O= random.vonmisesvariate(mu, kappa)/alpha #ore grade distribution
            self.seeds[s,::]=[i,j,RL,H2O] #,10 #last column = 10 tonnes per block.
            
    
    def I2D(self,x,y,z):
        
        #interpolation function for block (x,y,z)
        value=0
        gradesum=0
        dsum=0
        n_seeds=len(self.seeds)
        incrementalvalue = np.zeros([n_seeds,2])
        
        for v in range(n_seeds):
            
            d=np.sqrt((self.seeds[v,0]-x)**2+(self.seeds[v,1]-y)**2+(self.seeds[v,2]-z)**2) #cartesian distance for interpolation
            
            if d == 0:
                incrementalvalue[::,::] =[self.seeds[v,3], 1/n_seeds]
                break

            elif d<(np.sqrt(self._I**2+self._J**2+self.RL**2)/2):
                incrementalvalue[v,::]=[(self.seeds[v,3]/d), 1/d] #v/d interpolation
                           
            else:
                incrementalvalue[v,::]=[0.001,1]
                
            
        value=np.dot(incrementalvalue[:,0],incrementalvalue[:,1])
        
        return value
       
     
    def interpolate_excel(self):
        
        #deprecated
        
        names=np.array(['_I','_J','RL','H2O','Tonnes'])
        data=pd.DataFrame(columns=names)

        for i in range(self._I):
            for j in range(self._J):
                for k in range(self.RL):
                    
                    H2O=self.I2D(i,j,k)
                    dic={'_I':i,'_J':j,'RL':k,'H2O':H2O,'Tonnes':10}
                    data = data.append(dic, True)
                                        
        return data
    
    
    def interpolate(self):
        
        #apply interpolation to create block model.
        
        geo_array=np.zeros([self._I,self._J,self.RL,2]) #2 channels one for h20, one for mined state.
        
        for i in range(self._I):
            for j in range(self._J):
                for k in range(self.RL):
                    
                    H2O=self.I2D(i,j,k)                  
                    geo_array[i,j,k,0]=H2O #grade value interpolated
                    #geo_array[i,j,k,1]=10 #tonnes value kept constant
                                                         
        return geo_array    
                  

    def buildmodel_excel(self, ilen,jlen,depth,callnumber):
        
        #deprecated
        
        self._I=ilen
        self._J=jlen
        self.RL=depth
        maxseeds=np.ceil(ilen*jlen*depth/150)
        self.seedlocations(random.randint(8,maxseeds))
        data=self.interpolate_excel()        
        
        filename= 'BM_auto%s.xlsx' % callnumber
        data.to_excel(filename)     
      
 
    def buildmodel(self):
        
        #handle function
        
        maxseeds=np.ceil(self._I*self._J*self.RL/100)
        self.seedlocations(random.randint(1,maxseeds))
        geo_array=self.interpolate()        
        
        return geo_array
                
        
class fuzzymodel():
    
    #generate a block model with interpolated uncertainty distributions represting grades.
    
    def __init__(self,ilen,jlen,depth):

        self._I=ilen
        self._J=jlen
        self.RL=depth
        self.seeds=np.zeros(1)

        
    def seedlocations(self, numseedlocations):
        
        #input parameters for seeds.
        
        mu = 0#mean location (-1-0.1)
        kappa =4 #dilation y (range 1-3)
        alpha = 20 #dilation x (range 10-20)
        self.seeds=np.zeros([numseedlocations,4])
        self.trueseeds=np.zeros([numseedlocations,4])
        
        for s in range(numseedlocations):
            
            i = random.randint(2, self._I-2)
            j = random.randint(2, self._J-2)
            RL = random.randint(1, self.RL-2)
            H2O= random.vonmisesvariate(mu, kappa)/alpha #ore grade distribution
            self.seeds[s,::]=[i,j,RL,H2O] #,10 #last column = 10 tonnes per block.
            self.trueseeds[s,::]=[i,j,RL,max(0,np.random.normal(H2O,0.3))]
            
    
    def I2D(self, x,y,z,seedarray, seedpc):
        
        #interpolation function for block (x,y,z)
        value=0
        gradesum=0
        dsum=0
        
        n_seeds=round(len(seedarray)*seedpc) #using only 75% of seeds for fuzzzy model. remainder employed in discretisation.
        incrementalvalue = np.zeros([n_seeds,2])
        
        for v in range(n_seeds):
            
            d=np.sqrt((seedarray[v,0]-x)**2+(seedarray[v,1]-y)**2+(seedarray[v,2]-z)**2) #cartesian distance for interpolation
            
            if d == 0:
                incrementalvalue[::,::] =[seedarray[v,3], 1/n_seeds]
                break

            elif d<(np.sqrt(self._I**2+self._J**2+self.RL**2)/2):
                incrementalvalue[v,::]=[(seedarray[v,3]/d), 1/d] #v/d interpolation
                           
            else:
                incrementalvalue[v,::]=[0.001,1]
                
            
        value=np.dot(incrementalvalue[:,0],incrementalvalue[:,1])
        
        return value
      
    def ISD(self,x,y,z, seedarray, seedpc):
        
        #interpolation function for standard deviation of block (x,y,z)
        value=0
        sdsum=0
        dsum=0
        n_seeds=round(len(seedarray)*seedpc) #using only 75% of seeds for fuzzzy model. remainder employed in discretisation.
        incrementalvalue = np.zeros([n_seeds,2])
        
        for v in range(n_seeds):
            
            d=np.sqrt((seedarray[v,0]-x)**2+(seedarray[v,1]-y)**2+(seedarray[v,2]-z)**2) #cartesian distance for interpolation
            
            if d == 0:
                dsum=0
                break          
            else:
                dsum=dsum+d #cartesian distance for interpolation
            
        averaged=dsum/n_seeds     
        value=2*averaged**2 #assumption: average distance squared proportional to normal dist standard deviation for H2O
        
        return value        
     
    
    def interpolate(self):
        
        #apply interpolation to create block model.
        
        geo_array=np.zeros([self._I,self._J,self.RL,3]) #3 channels (expected mean H2O, mined state, standard deviation H2O)
        seedarray=self.seeds
        for i in range(self._I):
            for j in range(self._J):
                for k in range(self.RL):
                    
                    H2O=self.I2D(i,j,k,seedarray, 0.8) #using only 75% of seeds for fuzzzy model. remainder employed in discretisation.
                    SD=self.ISD(i,j,k,seedarray, 0.8)
                    geo_array[i,j,k,0]=H2O #expected grade value interpolated
                    geo_array[i,j,k,2]=SD #standard deviation of H2O value
                                                         
        return geo_array    

    def discretize(self):
        

        seedarray=self.trueseeds

        truth_array=np.zeros([self._I,self._J,self.RL,3]) #3 channels (expected mean H2O, mined state, standard deviation H2O)
        
        for i in range(self._I):
            for j in range(self._J):
                for k in range(self.RL):
                    
                    H2O=self.I2D(i,j,k,seedarray,1) #using only 100% of seeds for fuzzzy model. remainder employed in discretisation.
                    #SD=self.ISD(i,j,k)
                    truth_array[i,j,k,0]=H2O #expected grade value interpolated
                    #truth_array[i,j,k,2]=0 #standard deviation of H2O value 
        
        return truth_array


    def buildmodel(self):
        
        #handle function
        
        maxseeds=np.ceil(self._I*self._J*self.RL/100)
        self.seedlocations(random.randint(1,maxseeds))
        geo_array=self.interpolate()        
        truth_array=self.discretize()
        
        return geo_array, truth_array
    
    

class fuzzymodel2():
    
    #generate a block model with interpolated uncertainty distributions represting grades.
    
    def __init__(self,ilen,jlen,depth):

        self._I=ilen
        self._J=jlen
        self.RL=depth
        self.seeds=np.zeros(1)

        
    def seedlocations(self, numseedlocations):
        
        #input parameters for seeds.
        
        mu = 0#mean location (-1-0.1)
        kappa =4 #dilation y (range 1-3)
        alpha = 20 #dilation x (range 10-20)
        self.seeds=np.zeros([numseedlocations,4])
        self.trueseeds=np.zeros([numseedlocations,4])
        
        for s in range(numseedlocations):
            
            i = random.randint(2, self._I-2)
            j = random.randint(2, self._J-2)
            RL = random.randint(1, self.RL-2)
            H2O= random.vonmisesvariate(mu, kappa)/alpha #ore grade distribution
            self.seeds[s,::]=[i,j,RL,H2O] #,10 #last column = 10 tonnes per block.
            self.trueseeds[s,::]=[i,j,RL,max(0,np.random.normal(H2O,0.3))]
            
    
    def I2D(self, x,y,z,seedarray, seedpc):
        
        #interpolation function for block (x,y,z)
        value=0
        gradesum=0
        dsum=0
        
        n_seeds=round(len(seedarray)*seedpc) #using only 75% of seeds for fuzzzy model. remainder employed in discretisation.
        incrementalvalue = np.zeros([n_seeds,2])
        
        for v in range(n_seeds):
            
            d=np.sqrt((seedarray[v,0]-x)**2+(seedarray[v,1]-y)**2+(seedarray[v,2]-z)**2) #cartesian distance for interpolation
            
            if d == 0:
                incrementalvalue[::,::] =[seedarray[v,3], 1/n_seeds]
                break

            elif d<(np.sqrt(self._I**2+self._J**2+self.RL**2)/2):
                incrementalvalue[v,::]=[(seedarray[v,3]/d), 1/d] #v/d interpolation
                           
            else:
                incrementalvalue[v,::]=[0.001,1]
                
            
        value=np.dot(incrementalvalue[:,0],incrementalvalue[:,1])
        
        return value
      
    def ISD(self,x,y,z, seedarray, seedpc):
        
        #interpolation function for standard deviation of block (x,y,z)
        value=0
        sdsum=0
        dsum=0
        n_seeds=round(len(seedarray)*seedpc) #using only 75% of seeds for fuzzzy model. remainder employed in discretisation.
        incrementalvalue = np.zeros([n_seeds,2])
        
        for v in range(n_seeds):
            
            d=np.sqrt((seedarray[v,0]-x)**2+(seedarray[v,1]-y)**2+(seedarray[v,2]-z)**2) #cartesian distance for interpolation
            
            if d == 0:
                dsum=0
                break          
            else:
                dsum=dsum+d #cartesian distance for interpolation
            
        averaged=dsum/n_seeds     
        value=2*averaged**2 #assumption: average distance squared proportional to normal dist standard deviation for H2O
        
        return value        
     
    
    def interpolate(self):
        
        #apply interpolation to create block model.
        
        geo_array=np.zeros([self._I,self._J,self.RL,2]) #3 channels (expected mean H2O, mined state, standard deviation H2O)
        seedarray=self.seeds
        for i in range(self._I):
            for j in range(self._J):
                for k in range(self.RL):
                    
                    H2O=self.I2D(i,j,k,seedarray, 0.8) #using only 75% of seeds for fuzzzy model. remainder employed in discretisation.
                    #SD=self.ISD(i,j,k,seedarray, 0.8)
                    geo_array[i,j,k,0]=H2O #expected grade value interpolated
                    #geo_array[i,j,k,2]=SD #standard deviation of H2O value
                                                         
        return geo_array    

    def discretize(self):
        

        seedarray=self.trueseeds

        truth_array=np.zeros([self._I,self._J,self.RL,2]) #3 channels (expected mean H2O, mined state, standard deviation H2O)
        
        for i in range(self._I):
            for j in range(self._J):
                for k in range(self.RL):
                    
                    H2O=self.I2D(i,j,k,seedarray,1) #using only 100% of seeds for fuzzzy model. remainder employed in discretisation.
                    #SD=self.ISD(i,j,k)
                    truth_array[i,j,k,0]=H2O #expected grade value interpolated
                    #truth_array[i,j,k,2]=0 #standard deviation of H2O value 
        
        return truth_array


    def buildmodel(self):
        
        #handle function
        
        maxseeds=np.ceil(self._I*self._J*self.RL/100)
        self.seedlocations(random.randint(1,maxseeds))
        geo_array=self.interpolate()        
        truth_array=self.discretize()
        
        return geo_array, truth_array            
            
            
            