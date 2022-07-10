#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 22:40:03 2022

@author: parth

Utility functions for camera
"""
import numpy as np
from pr3_utils import Homogenize, Dehomogenize, hat

def translate_to_world(features, w_T_imu, imu_T_cam, K, b):
    '''
    Utility function translates features in image coordinate to 
    the world frame.
    '''
    z = K[0,0]*b/(features[0,:]-features[2,:])
    y = z*(features[1,:]-K[1,2])/K[1,1]
    x = z*(features[0,:]-K[0,2])/K[0,0]
    
    xcam_h = Homogenize(np.vstack((x,y,z)))
    x_imu = imu_T_cam @ xcam_h
    x_w = Dehomogenize(w_T_imu @ x_imu)
    return x_w

def translate_to_cam(x_w, w_T_imu, imu_T_cam, Ks):
    '''
    Translates the landmark coordinates in world frame to the
    left image pixel coordinates
    '''
    assert x_w.shape[0] == 3 , 'Expect 3xN world points array'
    x_cam = np.linalg.inv(w_T_imu @ imu_T_cam) @ Homogenize(x_w)
    
    x_img = Ks @ x_cam/x_cam[2,:]
    return x_img

def dpi_dq(q):
    '''
    Compute the derivative of the projection function
    at q
    '''
    q1,q2,q3,q4 = q[:]
    
    der = np.array([[1,0,-q1/q3,0],
                       [0,1,-q2/q3,0],
                       [0,0,0,0],
                       [0,1,-q4/q3,1]])/q3
    
    return der

def dot_op(s):
    '''
    Utility function to compute the dot operator for jacobian of  
    observation model wrt map co-ord
    '''
    assert s.shape[0] == 4 , 'Expext 3D homogeneous coordinates'
    s = s/s[-1]
    
    return np.vstack((np.hstack((np.eye(3), -hat(s[0:3]))), np.zeros((1,6))))
    

if __name__ == "__main__":
    print('MAIN')
    print(dot_op(np.array([1,2,3,1])))