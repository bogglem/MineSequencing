# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:07:33 2020

@author: Tim Pelech
"""
from keras.layers import Dense, Input,Conv3D, Dropout, Flatten, MaxPool3D
from keras.models import Model, Sequential


state_size=[15,15,6,3]
action_size=15*15

model2=Sequential()

model2.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=state_size, padding='valid'))
#model2.add(MaxPool3D((2,2,2), (1,1,1), padding='valid'))
model2.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=state_size, padding='valid'))
model2.add(MaxPool3D((2,2,2), (1,1,1), padding='valid'))

model2.add(Flatten())
model2.add(Dense(32, activation='relu'))
model2.add(Dense(64, activation='relu'))
model2.add(Dense(action_size, activation='relu'))

model2.summary()