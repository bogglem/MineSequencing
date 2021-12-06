# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 12:04:45 2020

@author: Tim Pelech
"""
import os.path
import numpy as np
import pandas as pd
import random
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
import gym
from gym import spaces
from tools.render import renderbm
from tools.createmodel import automodel


#inherits gym.Env to create a new gym environment (orebody block model) type for stable-baselines

class environment(gym.Env):
    
    def __init__(self, x,y,z ,gamma, turnspc, policy, rg_prob='rg', rendermode='off'):
        
        self.rendermode=rendermode # on/off display block model in matplotlib
       # self.cutoffpenaltyscalar=penaltyscalar #scaling parameter for changing the penalty for taking no action (cutoff).
        self.rg_prob=rg_prob #rg for randomly generated, loadenv for loading premade envionments
       # self.savepath=savepath
        envpath='./environments'
        self.savedgeo='%s/geology' % envpath
        # self.savedtruth='%s/truth' % envpath
        self.savedenv='%s/environment' % envpath
        self.saveddepdic='%s/depdict' % envpath
        self.savedeffdic='%s/effdict' % envpath
        self.policy=policy
        
        #initiating values
        self.framecounter=0
        self.actionslist = list()
        self.reward=0
        self.discountedmined=0
        self.turncounter=1
        self.i=-1
        self.j=-1
        self.terminal=False
        self.gamma=gamma #discount factor exponential (reward*turn^discount factor)
        self.Imin=0
        self.Imax=x
        self.Jmin=0
        self.Jmax=y
        self.RLmin=0
        self.RLmax=z
        self.mined=-1
        self.callnumber=1
        self.savenumber=0
        self.episodecounter=0
        

        
        try:
            self.maxloadid=len([name for name in os.listdir(self.savedgeo) if os.path.isfile(os.path.join(self.savedgeo, name))])
        except:
            self.maxloadid=1
            
        #sizing the block model environment
        self.Ilen=self.Imax-self.Imin 
        self.Jlen=self.Jmax-self.Jmin
        self.RLlen=self.RLmax-self.RLmin #RL (z coordinate) counts up as depth increases
        self.channels = 2 #H2O mean, mined state, Standard deviation
        self.flatlen=self.Ilen*self.Jlen*self.RLlen*self.channels
        
        #initiating block dependency dictionaries
        #self.block_dic={}
        self.block_dic_init={}
        self.dep_dic={}
        self.dep_dic_init={}
        self.eff_dic_init={}
        
        #create block model
        self.model=automodel(self.Ilen,self.Jlen,self.RLlen)
        self.build()
        
        self.startingturnspc=0.04
        self.turns=round(len(self.dep_dic)*self.startingturnspc,0) #set max number of turns (actions) in each episode based on percentage of block model size.
        self.maxturnspc=turnspc-self.startingturnspc

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        #actions are taken on a 2D checkerboard style view of the environement. progress will be made downwards in 3D over time.
        
        self.action_space = spaces.Discrete((self.Ilen)*(self.Jlen))#+1) #+1 action for choosing terminal state.
        
        if self.policy=='CnnPolicy':
        
            #observations are made of the entire environment (3D model with 3 channels, 1 channel represents mined state)
            self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(self.Ilen, self.Jlen, self.RLlen,self.channels), dtype=np.float64)
            
        elif self.policy=='MlpPolicy':
            #observations are made of the entire environment (3D model with 3 channels, 1 channel represents mined state)
            self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(self.flatlen,), dtype=np.float64)
            
                    
        #self.init_cutoffpenalty=self.cutoffpenalty() #experimental parameter function. penalises agent for not mining (do nothing), reward for taking action.

    def save(self):

        #create dir        
        #create dir        
        if (os.path.exists('./environments')!=True):
            os.mkdir('./environments')
        if (os.path.exists('%s' %self.savedgeo)!=True):
            os.mkdir('%s' %self.savedgeo)
        # if (os.path.exists('%s' %self.savedtruth)!=True):
        #     os.mkdir('%s' %self.savedtruth)
        if (os.path.exists('%s' %self.savedenv)!=True):
            os.mkdir('%s' %self.savedenv)
        if (os.path.exists('%s' %self.saveddepdic)!=True):
            os.mkdir('%s' %self.saveddepdic)
        if (os.path.exists('%s' %self.savedeffdic)!=True):
            os.mkdir('%s' %self.savedeffdic)        
        
        self.savenumber=len([name for name in os.listdir(self.savedgeo) if os.path.isfile(os.path.join(self.savedgeo, name))])+1
      
        
        #save geo array   
        if (os.path.exists(self.savedgeo)):
            np.save("%s/%s_geo_array"% (self.savedgeo, self.savenumber), self.geo_array)
          
        
        elif (os.path.exists(self.savedgeo)!=True):
            os.mkdir(self.savedgeo)
            np.save("%s/%s_geo_array"% (self.savedgeo, self.savenumber), self.geo_array)
        
        #save normalised ob_sample       
        if (os.path.exists(self.savedenv)):
            np.save("%s/%s_ob_sample"% (self.savedenv, self.savenumber), self.ob_sample)
          
        
        elif (os.path.exists(self.savedenv)!=True):
            os.mkdir(self.savedenv)
            np.save("%s/%s_ob_sample"% (self.savedenv, self.savenumber), self.ob_sample)   
      
        #save dep_dic  
        if (os.path.exists(self.saveddepdic)):
            np.save("%s/%s_dep_dic"% (self.saveddepdic, self.savenumber), self.dep_dic)
          
        
        elif (os.path.exists(self.saveddepdic)!=True):
            os.mkdir(self.saveddepdic)
            np.save("%s/%s_dep_dic"% (self.saveddepdic, self.savenumber), self.dep_dic)
        
        
         #save eff_dic   
        if (os.path.exists(self.savedeffdic)):
            np.save("%s/%s_eff_dic"% (self.savedeffdic, self.savenumber), self.eff_dic)
          
        
        elif (os.path.exists(self.savedeffdic)!=True):
            os.mkdir(self.savedeffdic)
            np.save("%s/%s_eff_dic"% (self.savedeffdic, self.savenumber), self.eff_dic)   
        
    
        
    def load(self, loadid):
        
        try:
            #self.geo_array=np.load("%s.npy"% self.savedenv)
            self.geo_array=np.load("%s/%s_geo_array.npy"% (self.savedgeo, loadid))
            self.ob_sample=np.load("%s/%s_ob_sample.npy"% (self.savedenv, loadid))
            # self.truth_array=np.load("%s/%s_geo_array.npy"% (self.savedgeo, loadid))
            self.dep_dic=np.load("%s/%s_dep_dic.npy"% (self.saveddepdic, loadid), allow_pickle='True').flat[0]
            self.eff_dic=np.load("%s/%s_eff_dic.npy"% (self.savedeffdic, loadid), allow_pickle='True').flat[0]
        
        except:
            self.geo_array=np.load("%s/%s_geo_array.npy"% (self.savedgeo, loadid+1))
            self.ob_sample=np.load("%s/%s_ob_sample.npy"% (self.savedenv, loadid+1))
            # self.truth_array=np.load("%s/%s_geo_array.npy"% (self.savedgeo, loadid+1))
            self.dep_dic=np.load("%s/%s_dep_dic.npy"% (self.saveddepdic, loadid+1), allow_pickle='True').flat[0]
            self.eff_dic=np.load("%s/%s_eff_dic.npy"% (self.savedeffdic, loadid+1), allow_pickle='True').flat[0]            

        self.averagereward=np.average(self.geo_array[:,:,:,0])
        
      
    def curriculum(self):
        
        self.maxloadid=len([name for name in os.listdir(self.savedgeo) if os.path.isfile(os.path.join(self.savedgeo, name))])
        n=2000
        
        if self.maxloadid>=n:
            
            self.savenumber=np.random.randint(2,n)
            
            if (os.path.exists(self.savedgeo)):
                 np.save("%s/%s_geo_array"% (self.savedgeo, self.savenumber), self.geo_array)

            #save normalised ob_sample       
            if (os.path.exists(self.savedenv)):
                 np.save("%s/%s_ob_sample"% (self.savedenv, self.savenumber), self.ob_sample)

            #save dep_dic  
            if (os.path.exists(self.saveddepdic)):
                 np.save("%s/%s_dep_dic"% (self.saveddepdic, self.savenumber), self.dep_dic)

             #save eff_dic   
            if (os.path.exists(self.savedeffdic)):
                 np.save("%s/%s_eff_dic"% (self.savedeffdic, self.savenumber), self.eff_dic)

        else:
            self.save()
        

    def build(self):
        
        #builds block model and mining sequence constraints dictionary (eg. top must be mined first)         
        if (self.rg_prob=='loadenv'):# and self.maxloadid>0: 
            
            if (random.random()>0.1) or (self.maxloadid<=1):
                loadid =1
            else:
                loadid = np.random.randint(1,self.maxloadid)      
            self.load(loadid)
        
        else:
            #self.geo_array, self.truth_array=self.model.buildmodel()
            self.geo_array=self.model.buildmodel()
            #self.save_env(self.savedenv,self.geo_array)
        
            
            scaler=MinMaxScaler()
            H2O_init=self.geo_array[:,:,:,0]
           # Tonnes_init=self.geo_array[:,:,:,1]
            State_init=self.geo_array[:,:,:,1]
           # SDev_init=self.geo_array[:,:,:,2]
            
            H2O_reshaped=H2O_init.reshape([-1,1])
            #Tonnes_reshaped=Tonnes_init.reshape([-1,1])
            State_reshaped=State_init.reshape([-1,1])
            #SDev_reshaped=SDev_init.reshape([-1,1])
            
            H2O_scaled=scaler.fit_transform(H2O_reshaped)
            #SDev_scaled=scaler.fit_transform(SDev_reshaped)
            
            a=H2O_scaled.reshape([self.Ilen, self.Jlen, self.RLlen,1])
            b=State_reshaped.reshape([self.Ilen, self.Jlen, self.RLlen,1])
            #c=SDev_scaled.reshape([self.Ilen, self.Jlen, self.RLlen,1])
            
            self.averagereward=np.average(self.geo_array[:,:,:,0])
             
            self.norm=np.append(a, b, axis=3)
           # self.norm=np.append(self.norm,c, axis=3)
            
            
            self.ob_sample=deepcopy(self.norm)
            self.construct_dep_dic()
            self.dep_dic=deepcopy(self.dep_dic_init)
            self.construct_eff_dic()
            self.eff_dic=deepcopy(self.eff_dic_init)
            
        #construct_dependencies blocks with zeros padding to avoid errors around environment edges.
        self.construct_block_dic()
        self.block_dic=deepcopy(self.block_dic_init) #deepcopy so dictionary doesnt have to be rebuilt for every new environment.
        self.render_update = self.geo_array[:,:,:,0] #provides data sliced for render function
        self.bm=renderbm(self.render_update)

        # #save environment if random generation disabled
        # if self.rg_prob==0.0 and not (os.path.isfile('%s.npy' % self.savedenv)):
        #     self.save_env(self.savedenv,self.geo_array)
    
    def construct_block_dic(self):
       
        #each block has a string reference in the dictionary bassed on i,j,k coordinates
        #extra 0 dependencies (padding) around edges to avoid errors.
        
        for i in range(-1,self.Ilen+1):
            for j in range(-1,self.Jlen+1):
                for k in range(self.RLlen):
                    
                    block=str(i)+str('_')+str(j)+str('_')+str(k)
                    self.block_dic_init["%s"% block]=0 
                           
    
    def construct_dep_dic(self):    
    
        #construct_dependencies
        #each block has a list of dependencies (other blocks) which must be removed prior to mining that block.
        
        for i in range(self.Ilen):
            for j in range(self.Jlen):
                for k in range(self.RLlen):
                    
                    block=str(i)+str('_')+str(j)+str('_')+str(k)
                    if k==0: #if block is surface layer, then no dependency exists
                        
                        dep=list(['','','','','','','','',''])
                        self.dep_dic_init["%s"% block]=dep
                        
                    else:
                        dep0=str(i-1)+str('_')+str(j+1)+str('_')+str(k-1)
                        dep1=str(i)+str('_')+str(j+1)+str('_')+str(k-1)
                        dep2=str(i+1)+str('_')+str(j+1)+str('_')+str(k-1)
                        dep3=str(i-1)+str('_')+str(j)+str('_')+str(k-1)
                        dep4=str(i)+str('_')+str(j)+str('_')+str(k-1)
                        dep5=str(i+1)+str('_')+str(j)+str('_')+str(k-1)
                        dep6=str(i-1)+str('_')+str(j-1)+str('_')+str(k-1)
                        dep7=str(i)+str('_')+str(j-1)+str('_')+str(k-1)
                        dep8=str(i+1)+str('_')+str(j-1)+str('_')+str(k-1)
                        
                        dep=list([dep0,dep1,dep2,dep3,dep4,dep5,dep6,dep7,dep8])
                        self.dep_dic_init["%s"% block]=dep
               
    def construct_eff_dic(self):    
    #construct_dependencies to encourange algorithm in digging deeper. experimental function, not currently used.
        
        for i in range(self.Ilen):
            for j in range(self.Jlen):
                for k in range(self.RLlen):
                    
                    block=str(i)+str('_')+str(j)+str('_')+str(k)
                       
                    dep9=str(i-1)+str('_')+str(j+1)+str('_')+str(k)
                    dep10=str(i)+str('_')+str(j+1)+str('_')+str(k)
                    dep11=str(i+1)+str('_')+str(j+1)+str('_')+str(k)
                    dep12=str(i-1)+str('_')+str(j)+str('_')+str(k)
                    dep13=str(i+1)+str('_')+str(j)+str('_')+str(k)
                    dep14=str(i-1)+str('_')+str(j-1)+str('_')+str(k)
                    dep15=str(i)+str('_')+str(j-1)+str('_')+str(k)
                    dep16=str(i+1)+str('_')+str(j-1)+str('_')+str(k)

                        
                    dep=list([dep9,dep10,dep11,dep12,dep13,dep14,dep15,dep16])
                    self.eff_dic_init["%s"% block]=dep
                    
    
    def actcoords(self, action):
        #map coords
        q=np.zeros((self.Ilen)*(self.Jlen))
        q[action]=1
        
        q2=q.reshape(self.Ilen,self.Jlen)
        action_coords=np.argwhere(q2.max()==q2)[0]
        
        #mapping q values to action coordinates
        self.i=action_coords[0]#+1
        self.j=action_coords[1]#+1    
    
    def select_block(self):
    
        #function identifies which block will be mined based on the current action (top to bottom mining).
        
        for k in range(self.RLlen): #iterate through orebody at action location to find highest unmined block (reversed -top to bottom)
            check_block=str(self.i)+str('_')+str(self.j)+str('_')+str(k)
            
            if self.block_dic["%s"% check_block]==0:
                selected_block = check_block
                self.RL = k
                break
            
            if k==self.RLlen-1:
                selected_block = check_block            
            
        return selected_block #return string name of selected block to mine
        
    
    def isMinable(self, selected_block):
        
        #find out if it is possible to mine selected block via dependency list.
        
        deplist = self.dep_dic["%s"% selected_block]
        minablelogic=np.zeros(len(deplist))
        
        for d in range(len(deplist)):
            depstr=deplist[d]
            
            if depstr=='':
               minablelogic[d]=1
               
            else: #if not surface then check dependencies
               minablelogic[d]=self.block_dic["%s"% depstr]
        
        isMinable=int(np.prod(minablelogic)) #logic 1,0 (is minable, not minable)
                   
        return isMinable
    
    def isEfficient(self,selected_block):
        
        #experimental indicator function to encourage mining adjacent blocks rather than spread out.
        
        deplist = self.eff_dic["%s"% selected_block]
        efficientlogic=np.zeros(len(deplist))
        
        for d in range(len(deplist)):
            depstr=deplist[d]
            
            if depstr=='':
                efficientlogic[d]=1
               
            else: #if not surface then check dependencies
                efficientlogic[d]=self.block_dic["%s"% depstr]
        
        isEfficient=efficientlogic.max() #if any blocks are mined (value 1), the adjacent block is considered efficient excavation.
                   
        return isEfficient        
        
    
    def unminedOre(self):
        
        #caluclates penalty for terminating episode early while remaining ore is unmined (for future use to determine the cutoff grade)
        
        blocks=np.multiply(self.ob_sample[:,:,:,0],self.ob_sample[:,:,:,1])
        #remaining=np.multiply(blocks,self.ob_sample[:,:,:,1])
        #cutoff=np.add(blocks,self.init_cutoffpenalty) #
        abandonreward=np.sum(np.where(blocks>self.averagereward,blocks,0))/np.sum(self.ob_sample[:,:,:,1]) #this indicator needs work. will be a focus of research.
        
        # mined=np.multiply(self.ob_sample[:,:,:,2],ore) #mined blocks updated to 1, (blocks-0.5)*x translates states to cause penalty for not mining, reward for mining.
        # unmined=np.subtract(ore,mined)
        
        # blocks=np.multiply(self.geo_array[:,:,:,0],self.geo_array[:,:,:,1])
        
        
        return abandonreward
    
    
    
    def equip_failure(self):
        #x=self.turncounter
        #prob_fail= #1-np.exp(-x*0.00001)
        
        if random.random()>0.9995**self.turncounter: #probability of success
            self.terminal=True
        else:
            self.terminal=False
        return
    
    def step(self, action):        
        
        info={} #required for gym.Env class output
       
        # if (random.random()<0.00001): #every 10 000 steps randomly save environment 
        #     self.maxloadid+=1
        #     self.save()
        
        if sum(sum(sum(self.ob_sample[:,:,:,1])))>=self.ob_sample[:,:,:,1].size: #if all blocks are mined, end episode
            self.terminal=True
            observation=self.ob_sample
               
        elif (self.turncounter>=self.turns): #if number of turns exceeds limit, end episode
            self.terminal=True
            self.reward = 0
            observation=self.ob_sample
        
        elif action>=((self.Ilen)*(self.Jlen)):
            self.terminal=True
            self.reward = -self.unminedOre()  
            observation=self.ob_sample
        
        else:   #normal step process
            self.actcoords(action)
            selected_block=self.select_block()
            isMinable=self.isMinable(selected_block)
            isEfficient=self.isEfficient(selected_block)
            
            self.evaluate(selected_block, isMinable, isEfficient)
            self.update(selected_block)
            self.turncounter+=1
            self.renderif(self.rendermode)
            #if random.random()>0.9999:
             #   self.curriculum()
            #self.equip_failure() #terminates episode based on random failure of equipment
            
            
        if self.policy=='MlpPolicy':
            arr=np.ndarray.flatten(self.ob_sample) #uncomment line for MLP (not CNN) policy
            observation=arr.reshape([len(arr)]) #uncomment line for MLP (not CNN) policy
                
        else:
            observation=self.ob_sample
        
        
        
        return observation, self.reward, self.terminal, info    
    
                 
    def evaluate(self, selected_block, isMinable, isEfficient):
        
        if isMinable==0:             #penalising repetetive useless actions
            
            ore=-self.averagereward
            
        elif isEfficient==0: #penalising high entropy policies spreading out and randomly picking.
            ore=-self.averagereward
                
        else:
            
            H2O=self.geo_array[self.i,self.j,self.RL,0]
            #Tonnes=self.geo_array[self.i, self.j,self.RL,1] 

            # if (H2O*Tonnes)+self.init_cutoffpenalty>=0: #to be used for experimental determination of cutoff grade
            ore=H2O
            # else:
            #     self.reward=self.init_cutoffpenalty
                
        self.reward=ore*10
        
    def update(self, selected_block):
    
        #updates observation environment and minable block dependencies.
        
        self.block_dic["%s"% selected_block]=1 #set to one (mined). required for dependency logical multiplication
        self.ob_sample[self.i,self.j,self.RL,1]=1 #set to one (mined) for agent observation.
   
    def reset(self):
        
        #start new episode.
            
        # loadid = int(np.ceil(random.random()*self.maxloadid))
        # self.load(loadid)
        
        #else:
        self.build()
            
        self.reward=0
        self.discountedmined=0
        self.turncounter=0
        self.episodecounter+=1
        #increase number of turns available as training progresses
        self.turns=min(round((len(self.dep_dic)*(self.maxturnspc*self.episodecounter/100000+self.startingturnspc))),round((len(self.dep_dic)*(self.maxturnspc))))
        self.terminal=False
        self.i=-1
        self.j=-1
        self.RL=-1
        #self.actionslist=list()
        
        if self.policy=='MlpPolicy':
            arr=np.ndarray.flatten(self.ob_sample) #uncomment line for MLP (not CNN) policy
            observation=arr.reshape([len(arr)]) #uncomment line for MLP (not CNN) policy
                
        else:
            observation=self.ob_sample

        return observation
                    
    
    def renderif(self, mode):      
        
        #create 3D plots if set 'on'
        
        if (mode=='on'): 
            self.framecounter +=1
        
            if self.framecounter<=1:
                self.render_update[self.i, self.j, self.RL]=0
    
                self.bm.initiate_plot(self.averagereward)
                self.bm.plot()
                
            self.bm.update_mined(self.i, self.j, self.RL)
            self.render_update[self.i, self.j, self.RL]=0 #not really required
                    
            if (self.framecounter % 5 == 0): #replot every 2 action frames.
                              
                 self.bm.plot()
        pass
   
    def render(self):      
    
        #create 3D plot
        
        # if geotruth=='truth':
       
        #     r=renderbm(self.truth_array[:,:,:,0])
            
        # else:
               
        r=renderbm(self.geo_array[:,:,:,0])
        
        r.initiate_plot(self.averagereward)
        r.plot()

    def renderx(self):      
    
        #create 3D plot
        
        # if geotruth=='truth':
       
        #     r=renderbm(self.truth_array[:,:,:,0])
            
        # else:
               
        r=renderbm(self.geo_array[:,:,:,0])
        
        r.initiate_plot(self.averagereward)
        r.plotx(20,0,0)                


        
