#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 22:38:13 2022

@author: parth
"""

import numpy as np
from pr3_utils import *
from cam_utils import *
from scipy.linalg import expm
from tqdm import tqdm


def compute_Jacobian(Ks, w_T_imu, imu_T_cam, x_w, idx_list):
    '''
    Helper function to compute the Jacobian of the observation 
    model wrt to landmark points
    '''
    N = len(idx_list)
    M = x_w.shape[1]
    cam_T_w = np.linalg.inv(w_T_imu @ imu_T_cam)
    x_cam = cam_T_w @ Homogenize(x_w)
    
    H = np.zeros((4*N,3*M))
    P = np.hstack((np.eye(3), np.zeros((3,1))))
    for i in range(N):
        idx = idx_list[i]
        H[4*i:4*i+4,3*idx:3*idx+3] = Ks @ dpi_dq(x_cam[:,idx]) @ cam_T_w @ P.T
    
    return H
    
data = load_data('data/03.npz')

# Read Data
ang_vel, lin_vel, features, TS, imu_T_cam, K, b = parse_data(data)

# Stereo camera matrix
Ks = np.vstack((K[:2,:],K[:2,:]))
Ks = np.hstack((Ks, np.zeros((4,1))))
Ks[2,-1] = - Ks[0,0]*b

# init pose 
pose = np.eye(4)

# Cache last time stamp
prev_TS = TS[0,0]

# Placeholder to store pose of the robot
pose_toplot = pose

# Read features
curr_features,detected_LM_idx = get_features_current_frame(features, 0, 20)

# Translate image features to world coordinates
x_w_landmark = translate_to_world(curr_features, pose, imu_T_cam, K, b)

# Number of landmarks
M = curr_features.shape[1]

# Init mean and cov
mean = np.ones((3,M))*np.nan
cov = np.zeros((3*M,3*M))

for i in range(len(detected_LM_idx)):
    idx = detected_LM_idx[i]
    mean[:,idx] = x_w_landmark[:,idx]
    cov[3*idx:3*idx+3,3*idx:3*idx+3] = np.eye(3)
    
# Initalize noises
# Measurement noise
V = np.eye(4)*0.5

###### Loop over time #######
for fIdx in tqdm(range(1,lin_vel.shape[1])):
    
    # Manage time
    tau = TS[0,fIdx]-prev_TS
    prev_TS = TS[0,fIdx]
    
    w = ang_vel[:,fIdx]
    v = lin_vel[:,fIdx]
    
    # Get twist matix for motion model
    tm = twist_matrix(np.hstack((v,w)))
    
    # Compute new pose
    pose = pose @ expm(tm*tau)    
    
    # Save pose 
    pose_toplot = np.dstack((pose_toplot, pose))
    
    # Featch image features
    curr_features,valid_landmarks = get_features_current_frame(features, fIdx, 20)
    
    # Get feature idx to update and add
    idx_toUpdate, idx_toAdd = get_idx(valid_landmarks, detected_LM_idx)
    
    # Update
    if len(idx_toUpdate) > 0:
        # Compute Jacobian
        H = compute_Jacobian(Ks, pose, imu_T_cam, mean, idx_toUpdate)
        
        # Compute Kalman Gain
        KG = cov @ H.T @ np.linalg.inv(H @ cov @ H.T + np.kron(np.eye(len(idx_toUpdate)),V))
        
        # Mean update
        res = curr_features[:, idx_toUpdate] - translate_to_cam(mean[:,idx_toUpdate], pose, imu_T_cam, Ks)
        delta_mean = KG @  res.ravel('F')
        mean[:,idx_toUpdate] = mean[:,idx_toUpdate] + delta_mean.reshape(3,-1, order = 'F')[:,idx_toUpdate]
        
        # Covariance Update
        cov = cov - KG @ H @ cov
        
    if len(idx_toAdd) > 0:
        x_w_landmark = translate_to_world(curr_features, pose, imu_T_cam, K, b)
        
        for j in range(len(idx_toAdd)):
            idx = idx_toAdd[j]
            mean[:,idx] = x_w_landmark[:,idx]
            cov[3*idx:3*idx+3,3*idx:3*idx+3] = np.eye(3)
            detected_LM_idx.append(idx)

visualize_trajectory_2d(pose_toplot, mean)