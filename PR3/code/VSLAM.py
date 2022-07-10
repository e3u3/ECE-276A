#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 23:48:18 2022

@author: parth
VSLAM
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
    cam_T_imu = np.linalg.inv(imu_T_cam)
    imu_T_w = np.linalg.inv(w_T_imu)
    x_cam = cam_T_w @ Homogenize(x_w)
    x_imu = imu_T_w @ Homogenize(x_w)
    
    H = np.zeros((4*N,3*M+6))
    P = np.hstack((np.eye(3), np.zeros((3,1))))
    for i in range(N):
        idx = idx_list[i]
        H[4*i:4*i+4,3*idx:3*idx+3] = Ks @ dpi_dq(x_cam[:,idx]) @ cam_T_w @ P.T
        H[4*i:4*i+4,-6:] = -Ks @ dpi_dq(x_cam[:,idx]) @ cam_T_imu  @ dot_op(x_imu[:,idx])
    return H
    
data = load_data('data/03.npz')
feature_skip = 15

# Read Data
ang_vel, lin_vel, features, TS, imu_T_cam, K, b = parse_data(data)

# Stereo camera matrix
Ks = np.vstack((K[:2,:],K[:2,:]))
Ks = np.hstack((Ks, np.zeros((4,1))))
Ks[2,-1] = - Ks[0,0]*b

# init pose 
mean_pose = expm(twist_matrix(np.array([0,0,0,np.pi,0,0]))) #np.eye(4)

# Cache last time stamp
prev_TS = TS[0,0]

# Placeholder to store pose of the robot
pose_toplot = mean_pose

# Read features
curr_features,detected_LM_idx = get_features_current_frame(features, 0, feature_skip)

# Translate image features to world coordinates
x_w_landmark = translate_to_world(curr_features, mean_pose, imu_T_cam, K, b)

# Number of landmarks
M = curr_features.shape[1]

# Init mean and cov
mean_L = np.ones((3,M))*np.nan
cov = np.zeros((3*M+6,3*M+6))

mean_L[:,detected_LM_idx] = x_w_landmark[:,detected_LM_idx]
cov[-6:,-6:] = np.eye(6)
for i in range(len(detected_LM_idx)):
    idx = detected_LM_idx[i]
    cov[3*idx:3*idx+3,3*idx:3*idx+3] = np.eye(3)
    
# Initalize noises
# Measurement noise
V = np.eye(4)*10

# Motion noise
W = np.eye(6)
W[0:3,0:3] = W[0:3,0:3]*1#0.3
W[3:6,3:6] = W[3:6,3:6]*0.3#0.05

###### Loop over time #######
for fIdx in tqdm(range(1,lin_vel.shape[1])):
    
    # Manage time
    tau = TS[0,fIdx]-prev_TS
    prev_TS = TS[0,fIdx]
    
    w = ang_vel[:,fIdx]
    w[-1] = -w[-1]
    v = lin_vel[:,fIdx]
     
    # Get twist matix for motion model
    tm = twist_matrix(np.hstack((v,w)))
    
    # Mean pose predict
    mean_pose = mean_pose @ expm(tm*tau)    
    
    # Cov pose predict
    F = expm(-tau*hat_alt(np.hstack((v,w))))
    cov[-6:,-6:] = F @ cov[-6:,-6:] @ F.T + W
        
    # Featch image features
    curr_features,valid_landmarks = get_features_current_frame(features, fIdx, feature_skip)
    
    # Get feature idx to update and add
    idx_toUpdate, idx_toAdd = get_idx(valid_landmarks, detected_LM_idx)
    
    # Update
    if len(idx_toUpdate) > 0:
        # Compute Jacobian
        H = compute_Jacobian(Ks, mean_pose, imu_T_cam, mean_L, idx_toUpdate)

        # Compute Kalman Gain
        KG = cov @ H.T @ np.linalg.inv(H @ cov @ H.T + np.kron(np.eye(len(idx_toUpdate)),V))
        
        # Mean update
        res = curr_features[:, idx_toUpdate] - translate_to_cam(mean_L[:,idx_toUpdate], mean_pose, imu_T_cam, Ks)
        delta_mean = KG @  res.ravel('F')
        
        # landmark pos update
        delta_mean_L = delta_mean[:-6].reshape(3,-1, order = 'F')
        mean_L[:,idx_toUpdate] = mean_L[:,idx_toUpdate] + delta_mean_L[:,idx_toUpdate]
        
        # mean pose update
        delta_mean_pose = delta_mean[-6:]
        mean_pose = mean_pose @ expm(twist_matrix(delta_mean_pose))
        
        # Covariance Update
        t1 = (np.eye(3*M + 6) - KG @ H)
        cov = t1 @ cov @ t1.T + KG @ np.kron(np.eye(len(idx_toUpdate)),V) @ KG.T
        #cov = cov - KG @ H @ cov
        
    if len(idx_toAdd) > 0:
        x_w_landmark = translate_to_world(curr_features, mean_pose, imu_T_cam, K, b)
        mean_L[:,idx_toAdd] = x_w_landmark[:,idx_toAdd]
        for j in range(len(idx_toAdd)):
            idx = idx_toAdd[j]
            cov[3*idx:3*idx+3,3*idx:3*idx+3] = np.eye(3)
            detected_LM_idx.append(idx)
            
    # Save pose 
    pose_toplot = np.dstack((pose_toplot, mean_pose))
    
visualize_trajectory_2d(pose_toplot, mean_L)

