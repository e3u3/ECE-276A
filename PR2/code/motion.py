#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 02:01:04 2022

@author: parth
"""

'''
This script gets the motion information 
from the gyroscope and wheel encoders 
to get the path of the car.
'''

import numpy as np
from pr2_utils import read_data_from_csv
from matplotlib import pyplot as plt

TS_e, encoder_data = read_data_from_csv('data/sensor_data/encoder.csv')
TS_FOG, FOG_data = read_data_from_csv('data/sensor_data/fog.csv')


# Parameters
# Wheel diameter
WHEEL_DIAMETER = 0.62

# Freq calculations
f_e = (max(TS_e) - min(TS_e))/TS_e.shape[0]
f_FOG = (max(TS_FOG) - min(TS_FOG))/TS_FOG.shape[0]

# placeholder for pose  
pose = []
x = 0
y = 0 
theta = 0
pose.append(np.array([x,y,theta]))

ts_e_prev = 0
ts_fog_prev = 0
idx_e_prev = 0
idx_fog_prev = 0

for idx_e,ts_e in enumerate(TS_e[1:,]):
    # Get nearest TS for FOG sensor reading
    idx_fog = (ts_e - TS_FOG[0])/f_FOG
    idx_fog = int(np.rint(idx_fog))
    ts_fog = TS_FOG[idx_fog]
    
    # delta t
    dT_e = (ts_e - ts_e_prev)*1e-9
    dT_FOG = (ts_fog - ts_fog_prev)*1e-9
    tau = (dT_FOG + dT_e)/2
     
    # Feed into motion model to get new pose.   
    v = (encoder_data[idx_e,0] - encoder_data[idx_e_prev,0]) * np.pi * WHEEL_DIAMETER/(4096*tau)
    w = (np.sum(FOG_data[idx_fog_prev:idx_fog,2]))/tau
    
    # New pose
    x += v*tau*np.cos(theta)
    y += v*tau*np.sin(theta)
    theta += w*tau
    
    # Append pose    
    pose.append(np.array([x,y,theta]))    
    
    ts_e_prev = ts_e
    ts_fog_prev = ts_fog
    idx_e_prev = idx_e
    idx_fog_prev = idx_fog
     
pose = np.array(pose)

plt.plot(pose[:,0], pose[:,1])
plt.title('Car Position')
plt.xlabel('x co-ordinate')
plt.ylabel('y co-ordinate')
plt.show()