#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 22:36:45 2022

@author: parth
a) Implement IMU localization based on SE(3)\
    kinematics using the linear and angular 
    velocity measurements.
"""
import numpy as np
from pr3_utils import *
from scipy.linalg import expm

data = load_data('data/03.npz')
pose = np.eye(4)

pose_toplot = pose

ang_vel, lin_vel, _, TS, _, _, _ = parse_data(data)

prev_t = TS[0,0]

for i in range(1,lin_vel.shape[1]):
       
    w = ang_vel[:,i]
    v = lin_vel[:,i]
    
    tm = np.array([[0,-w[2],w[1],v[0]],
                   [w[2],0,-w[0],v[1]],
                   [-w[1],w[0],0,v[2]],
                   [0,0,0,0]])
    
    pose = pose @ expm(tm * (TS[0,i]-prev_t))
    prev_t = TS[0,i]
    pose_toplot = np.dstack((pose_toplot, pose))
    
visualize_trajectory_2d(pose_toplot)
