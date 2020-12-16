# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:07:33 2020

@author: Tim Pelech
"""
from keras.layers import Dense, Input,Conv3D, Dropout, Flatten, MaxPool3D
from keras.models import Model, Sequential


state_size=[15,15,7,3]
action_size=15*15

model2=Sequential(name='cnn')

model2.add(Conv3D(16, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=state_size, padding='valid'))
#model2.add(MaxPool3D((2,2,2), (1,1,1), padding='valid'))
#model2.add(MaxPool3D((2,2,2), (1,1,1), padding='valid'))
model2.add(Conv3D(16, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=state_size, padding='valid'))
model2.add(MaxPool3D((2,2,2), (1,1,1), padding='valid'))

model2.add(Flatten())
model2.add(Dense(16, activation='relu'))
model2.add(Dense(16, activation='relu'))
model2.add(Dense(32, activation='relu'))
model2.add(Dense(action_size, activation='relu'))

model2.summary()


model3=Sequential(name='mlp')

model3.add(Dense(64, activation='relu', input_shape=state_size))
model3.add(Dense(64, activation='relu'))
model3.add(Dense(action_size, activation='relu'))

model3.summary()