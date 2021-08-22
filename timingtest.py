# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 19:28:54 2021

@author: Tim Pelech
"""

import time
from tools.RG3DBMenv import environment 

a=time.time()

env = environment(15,15,6,0.9, 1, 0.1, 0.2, 'test', 'MlpPolicy')

b=time.time()-a
c=time.time()

e=env.load_env('testenvironment')

d=time.time()-c

e=time.time()
f=env.build()

g=time.time()-e

print(b)
print(d)
print(g)