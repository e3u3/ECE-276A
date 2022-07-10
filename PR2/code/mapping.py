#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 13:26:03 2022

@author: parth
"""
import numpy as np
from pr2_utils import read_data_from_csv, bresenham2D
from matplotlib import pyplot as plt



def get_lidar_points(range_dist, dist_threshold):
    '''
    Helper function to get 2D lidar points
    from the range points
    '''
    
    angles = np.linspace(-5, 185, 286) / 180 * np.pi
    ranges = range_dist
    
    angles = angles[np.logical_and(ranges > dist_threshold[0], ranges < dist_threshold[1])]
    ranges = ranges[np.logical_and(ranges > dist_threshold[0], ranges < dist_threshold[1])] 
        
    return np.vstack((np.multiply(ranges,np.cos(angles)),np.multiply(ranges,np.sin(angles))))
    
    
def Homogenize(x):
    '''
    Helper function to homogenize points
    '''
    return np.vstack((x, np.zeros((1,x.shape[1]))))
    

def Dehomogenize(x):
    '''
    Helper function to dehomogenize points
    '''
    return x[:-1]/x[-1]

def plot_lidar_points(points):
    '''
    Helper function to plot the xy lidar scan
    points
    '''
    plt.scatter(points[0,:], points[1,:])
    plt.show()

def physical_to_map(x,y,MAP):
    '''
    Utility function to convert physical coordinates
    to map coordinates
    '''
    
    if (x < MAP['xmin'] or y < MAP['ymin']):
        print('hi')
    
    #x_map = max(np.floor((x - MAP['xmin'])/MAP['res'])-1,0)
    #y_map = max(np.floor((y - MAP['ymin'])/MAP['res'])-1,0)
    
    x_map = np.ceil((x - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    y_map = np.ceil((y - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    
    return int(x_map), int(y_map)
    

## MAIN ##


# Read data
TS_lidar, lidar_data = read_data_from_csv('data/sensor_data/lidar.csv')

# init MAP
MAP = {}
MAP['res']   = 5 #meters
MAP['xmin']  = -10  #meters
MAP['ymin']  = -1050
MAP['xmax']  =  1250
MAP['ymax']  =  10
MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int) #DATA TYPE: char or int8


lam = 1 #np.log(4)
lam_min = -np.inf #0.05
lam_max =  np.inf #0.7


# Extrinsic parameters (lidar)
R_Lidar_V = np.array([[0.00130201, 0.796097, 0.605167], 
                    [0.999999, -0.000419027, -0.00160026],
                    [-0.00102038, 0.605169, -0.796097]])

t_lidar_V = np.array([0.8349, -0.0126869, 1.76416])

# Car pose
car_pose = np.array([0,0,0])
sx = car_pose[0]
sy = car_pose[1]

for idx, ts_lidar in enumerate(TS_lidar):
    
    # Extract 2D Lidar points
    lidar_points = get_lidar_points(lidar_data[idx], [0.5, 70])
    plot_lidar_points(lidar_points)
   
    # Transform from lidar to world coordinate
    lidar_points_V = Dehomogenize(R_Lidar_V @ Homogenize(lidar_points) + t_lidar_V.reshape(-1,1))
    
    # Transform from vehicle to world
    theta = car_pose[2]
    t_v_w = car_pose[:2]
    R_V_W = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
    
    lidar_points_W = R_V_W @ lidar_points_V + t_v_w.reshape(-1,1)
    
    for i in range(lidar_points_W.shape[1]):
        ex = lidar_points_W[0,i]
        ey = lidar_points_W[1,i]
        
        # Translate physical points to map co-ordinates
        ex_m, ey_m = physical_to_map(ex,ey,MAP)
        sx_m, sy_m = physical_to_map(sx,sy,MAP)
                
        # Get free and occupied points from bresenham's 
        # 2D algorithm
        occ_points = bresenham2D(sx_m, sy_m, ex_m, ey_m)
        occ_points = occ_points.astype(np.int)
        

        #MAP['map'][occ_points[0,:-1], occ_points[1,:-1]] -= lam
        MAP['map'][ex_m,ey_m] = 1
    
    #MAP['map'][MAP['map'] < lam_min] = lam_min
    #MAP['map'][MAP['map'] > lam_max] = lam_max
    plt.imshow(MAP['map'],cmap="hot");
    plt.title("Occupancy grid map")
    
    
    break