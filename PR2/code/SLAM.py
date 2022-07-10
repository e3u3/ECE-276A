#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 15:29:52 2022

@author: parth
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 18:46:35 2022

@author: parth
"""
import numpy as np
from pr2_utils import read_data_from_csv, bresenham2D, mapCorrelation
import time
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import sys
import cv2

def tic():
  return time.time()

def toc(tstart, name="Operation"):
  print('%s took: %s sec.\n' % (name,(time.time() - tstart)))

class SLAM:
    def __init__(self, N = 1):
        '''
        Constructor
        '''
        print('Initalizing ......')
        
        # Number of particles
        self.N = N
        # Current state
        self.Particles = np.zeros((N,3))
        self.Weights = np.ones((N,1))/N

        # Load data
        self.TS_lidar, self.lidar_data = read_data_from_csv('data/sensor_data/lidar.csv')
        self.TS_encoder, self.encoder_data = read_data_from_csv('data/sensor_data/encoder.csv')
        self.TS_FOG, self.FOG_data = read_data_from_csv('data/sensor_data/fog.csv')
        self.TS_img = self.parse_img_TS()
        
        self.DIST_THRESHOLD = [1, 60]
        
        
        # Calculate frequency
        # Calculate freq
        self.f_FOG = (max(self.TS_FOG) - min(self.TS_FOG))/self.TS_FOG.shape[0]
        self.f_lidar = (max(self.TS_lidar) - min(self.TS_lidar))/self.TS_lidar.shape[0]
        self.f_encoder = (max(self.TS_encoder) - min(self.TS_encoder))/self.TS_encoder.shape[0]
        self.f_img = (max(self.TS_img) - min(self.TS_img))/self.TS_img.shape[0]
        
        # Extrinsic parameters (lidar)
        self.R_Lidar_V = np.array([[0.00130201, 0.796097, 0.605167], 
                            [0.999999, -0.000419027, -0.00160026],
                            [-0.00102038, 0.605169, -0.796097]])

        self.T_lidar_V = np.array([0.8349, -0.0126869, 1.76416]).reshape(-1,1)
        
        self.lam = np.log(4)
        self.lam_min = -128
        self.lam_max = 128
        
        # Initialize map
        self.init_MAP()
        self.PATH = []
        self.PATH.append([0,0])
        
        
        # Placeholder to store the previous idx 
        # for predict step
        self.PrevIdx = [0,0]
        
        # CONSTANTS
        self.RESOLUTION = 4096
        self.WHEEL_DIAMETER = 0.62
        self.COV_V = 0.1
        self.COV_W = 0.001
        
        self.xs = np.arange(-0.4,0.4+0.1,0.1)
        self.ys = self.xs
        
        # Camera intrinsics for texture mapping
        self.fsu = 8.1690378992770002e+02
        self.fsv = 8.1156803828490001e+02
        self.cu = 5.0510166700000003e-01
        self.cv = 6.0850726281690004e+02
        self.b = 475.14
        self.RO = np.array([[0,-1,0],
                            [0,0,-1],
                            [1,0,0]])
        self.R_Cam_V = np.array([[-0.00680499, -0.0153215, 0.99985],
                                 [-0.999977, 0.000334627, -0.00680066],
                                 [-0.000230383, -0.999883, -0.0153234]])
        
        self.T_Cam_V = np.array([1.64239,0.247401, 1.58411]).reshape(-1,1)
    
    def parse_img_TS(self):
        '''
        Utility function to extract timestamps from stereo img dir
        '''
        right_img_dir = sorted(os.listdir('data/sensor_data/stereo_right/'))
        left_img_dir = sorted(os.listdir('data/sensor_data/stereo_left/'))
        TS = []
        for file in right_img_dir:
            ts = int(file[:-4])
            TS.append(ts)
        TS = np.array(TS)
        self.Left_IMG_dir = left_img_dir
        self.Right_IMG_dir = right_img_dir
        return TS
            
        
    def init_MAP(self):
        '''
        Initalize the map with lidar scan
        '''
        print('Initalizing Map ......')
        
        # self.MAP = self.lidar_data[0]
        
        # init MAP
        MAP = {}
        MAP['res']   = 10 #0.5 #meters
        MAP['xmin']  = -80  #meters
        MAP['ymin']  = -1080
        MAP['xmax']  =  1280
        MAP['ymax']  =  80
        MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
        MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
        MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.float32) #DATA TYPE: char or int8
        MAP['texturemap'] = np.dstack([np.ones((MAP['sizex'],MAP['sizey']),dtype=np.float32)]*3)

        self.MAP = MAP
        self.X_IM = np.arange(self.MAP['xmin'],self.MAP['xmax']+self.MAP['res'],self.MAP['res']) #x-positions of each pixel of the map
        self.Y_IM = np.arange(self.MAP['ymin'],self.MAP['ymax']+self.MAP['res'],self.MAP['res']) #y-positions of each pixel of the map
        
        lidar = self.lidar_data[0]
        # Lidar to vehicle frame
        lidar_w = self.Dehomogenize(self.R_Lidar_V @ self.Homogenize(self.range_to_lidar(lidar)).T + self.T_lidar_V)
        
        sx = 0
        sy = 0
        
        for j in range(lidar_w.shape[1]):
            ex = lidar_w[0,j]
            ey = lidar_w[1,j]
            
            # Translate physical points to map co-ordinates
            ex_m, ey_m = self.physical_to_map(ex,ey)
            sx_m, sy_m = self.physical_to_map(sx,sy)
            
            # bresenham's algorithm                
            occ_points = bresenham2D(sx_m, sy_m, ex_m, ey_m)
            occ_points = occ_points.astype(np.int)
            self.MAP['map'][occ_points[0,:-1], occ_points[1,:-1]] -= self.lam
            self.MAP['map'][ex_m,ey_m] += self.lam
    
        self.MAP['map'][self.MAP['map'] < self.lam_min] = self.lam_min
        self.MAP['map'][self.MAP['map'] > self.lam_max] = self.lam_max
        
        
    def range_to_lidar(self, range_dist):
        '''
        Utility function to convert range data
        to lidar data in lidar frame
        '''
        angles = np.linspace(-5, 185, 286) / 180 * np.pi
        ranges = range_dist
        
        angles = angles[np.logical_and(ranges < self.DIST_THRESHOLD[1],ranges > self.DIST_THRESHOLD[0])]
        ranges = ranges[np.logical_and(ranges < self.DIST_THRESHOLD[1],ranges > self.DIST_THRESHOLD[0])]
            
        return np.vstack((np.multiply(ranges,np.cos(angles)),np.multiply(ranges,np.sin(angles)))).T
      
    @staticmethod
    def Homogenize(x):
        '''
        Helper function to homogenize points
        '''
        return np.hstack((x, np.ones((x.shape[0],1))))# np.hstack((x, np.ones((1,x.shape[1]))))
        
    @staticmethod
    def Dehomogenize(x):
        '''
        Helper function to dehomogenize points
        '''
        return x[:-1]
    
    def get_indices_from_TS(self, ts, idx):
        '''
        Utility function to get indices for each
        sensor corresponding to the passed timestamp ts.
        '''
        #idx_lidar = idx
            
        idx_fog = (ts - self.TS_FOG[0])/self.f_FOG
        idx_fog = np.floor(idx_fog).astype(np.int32)
        
        idx_lidar = (ts - self.TS_lidar[0])/self.f_lidar
        idx_lidar = int(np.rint(idx_lidar))
        
        idx_encoder = (ts - self.TS_encoder[0])/self.f_encoder
        idx_encoder = np.floor(idx_encoder).astype(np.int32)
        
        idx_img = (ts - self.TS_img[0])/self.f_img
        idx_img = np.floor(idx_img).astype(np.int32)
        
        return idx_fog, idx_lidar, idx_encoder, idx_img

    def physical_to_map(self, x, y):
        '''
        Utility function to convert physical coordinates
        to map coordinates
        '''
        
        x_map = min(np.ceil((x - self.MAP['xmin']) / self.MAP['res'] ).astype(np.int16)-1, self.MAP['map'].shape[0]-1)
        y_map = min(np.ceil((y - self.MAP['ymin']) / self.MAP['res'] ).astype(np.int16)-1, self.MAP['map'].shape[1]-1)
        
        
        return int(x_map), int(y_map)
    
    @staticmethod
    def get_2D_rotation_matrix(theta):
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]]) 
        return R

    def map_texture(self, idx_img, R, t):
        '''
        Helper function to calulate the disparity of 
        image pair for given idx_img
        '''
        image_l = cv2.imread(os.path.join('data/sensor_data/stereo_left/',self.Left_IMG_dir[idx_img]),0)
        image_r = cv2.imread(os.path.join('data/sensor_data/stereo_right/',self.Right_IMG_dir[idx_img]),0)
        
        image_l = cv2.cvtColor(image_l, cv2.COLOR_BAYER_BG2BGR)
        image_r = cv2.cvtColor(image_r, cv2.COLOR_BAYER_BG2BGR)

        image_l_gray = cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY)
        image_r_gray = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)

        # You may need to fine-tune the variables `numDisparities` and `blockSize` based on the desired accuracy
        stereo = cv2.StereoBM_create(numDisparities=32, blockSize=9) 
        disparity = stereo.compute(image_l_gray, image_r_gray)
        
        # Extract RGB , Depth values
        RGBD = np.dstack((image_l, disparity))
        # Computations to calculate equivlanet world coordinate
        U = np.ones_like(disparity) * np.arange(disparity.shape[0]).reshape(-1,1)
        V = np.ones_like(disparity) * np.arange(disparity.shape[1]).reshape(1,-1)

        RGBDUV = np.dstack((RGBD,U,V))
        RGBDUV = RGBDUV.reshape(-1,6)

        RGBDUV = RGBDUV[RGBDUV[:,3] > 0]
        Z = self.fsu * self.b / RGBDUV[:,3]
        X = (RGBDUV[:,-2] - self.cu)*self.b/RGBDUV[:,3]
        Y = (RGBDUV[:,-1] - self.cv)*self.b/RGBDUV[:,3]*self.fsu/self.fsv

        XYZ = np.column_stack((X,Y,Z))
        # Vehicle
        XYZ = np.linalg.inv(self.RO.T @ self.R_Cam_V) @ XYZ.T + self.T_Cam_V

        XYZRGB = np.column_stack((XYZ.T,RGBDUV[:,:3]))/1000
        
        # Filter particles based on Z height
        XYZRGB = XYZRGB[XYZRGB[:,2] < 10]
        
        XY_W = R @ XYZRGB[:,:2].T + t
        XY_W = XY_W - np.array([self.MAP['xmin'],self.MAP['ymin']]).reshape(-1,1)
        XY_W = np.ceil(XY_W/self.MAP['res']).astype(np.int16)-1
        XY_W = XY_W.clip(max = np.array([self.MAP['map'].shape[0]-1,self.MAP['map'].shape[1]-1]).reshape(-1,1))
        
        self.MAP['texturemap'][XY_W[0,:],XY_W[1,:],:] = XYZRGB[:,3:]
        
        return
        

    def update(self, idx_lidar, idx_img):
        '''
        Update step
        '''            
        # Read lidar data
        lidar = self.lidar_data[idx_lidar]
        
        # Transfer to vehicle frame
        lidar_v = self.R_Lidar_V @ self.Homogenize(self.range_to_lidar(lidar)).T + self.T_lidar_V
        lidar_v = lidar_v[:,lidar_v[-1,:] < 5]
        lidar_v = self.Dehomogenize(lidar_v)
        
        if lidar_v.shape[0] == 0:
            # NO POINTS
            return
        
        # TODO - Vectorize here!        
        map_cr = []
        
        # For each particle transfer lidar to world coordinate
        for i in range(self.N):
            theta = self.Particles[i,2]
            R = self.get_2D_rotation_matrix(theta)
               
            # Transfer to world frame
            lidar_w = R @ lidar_v + self.Particles[i,:2].reshape(-1,1)
            
            # Consider maximum correlation value
            map_cr.append(np.max(mapCorrelation(self.MAP['map'], self.X_IM, self.Y_IM, lidar_w, self.xs, self.ys)))
            
        map_cr = np.array(map_cr)
        map_cr = np.exp(map_cr)
        map_cr = map_cr/np.sum(map_cr)
        weights = np.multiply(self.Weights, map_cr.reshape(-1,1))
        
        if np.sum(weights) != 0:
            self.Weights = weights
            self.Weights = self.Weights/np.sum(self.Weights)
            
            idx = np.argmax(self.Weights)
            self.PATH.append([self.Particles[idx,0],self.Particles[idx,1]])
            
            # MAP UPDATE based on best particle
            theta = self.Particles[idx,2]
            R = self.get_2D_rotation_matrix(theta)         
            
            # Transfer to world frame
            lidar_w = R @ lidar_v + self.Particles[idx,:2].reshape(-1,1)
            
            # MAP texture
            self.map_texture(idx_img, R, self.Particles[idx,:2].reshape(-1,1))
            
            sx = self.Particles[idx,0]
            sy = self.Particles[idx,1]
            
            for j in range(lidar_w.shape[1]):
                ex = lidar_w[0,j]
                ey = lidar_w[1,j]
                
                # Translate physical points to map co-ordinates
                ex_m, ey_m = self.physical_to_map(ex,ey)
                sx_m, sy_m = self.physical_to_map(sx,sy)
                
                # bresenham's algorithm                
                occ_points = bresenham2D(sx_m, sy_m, ex_m, ey_m)
                occ_points = occ_points.astype(np.int)
                
                self.MAP['map'][occ_points[0,:-1], occ_points[1,:-1]] -= self.lam
                self.MAP['map'][ex_m,ey_m] += self.lam

            self.MAP['map'][self.MAP['map'] < self.lam_min] = self.lam_min
            self.MAP['map'][self.MAP['map'] > self.lam_max] = self.lam_max
            
            # Resample
            if 1/np.sum(np.square(self.Weights)) < self.N*0.6:
                particles = []
                for i in range(self.N):
                    j = np.random.choice(self.N, p = self.Weights.squeeze())
                    particles.append(self.Particles[j])
                self.Weights = np.ones((self.N,1))/self.N
                    
        
    def predict(self, idx_fog, idx_encoder):
        '''
        Predict step
        '''

        dT_e = self.TS_encoder[idx_encoder] - self.PrevIdx[0]
        dT_FOG = self.TS_FOG[idx_fog] - self.PrevIdx[1]
        tau = (dT_e + dT_FOG)/2
        
        # Feed into motion model to get new pose.
        v = (self.encoder_data[idx_encoder,0] - self.encoder_data[self.PrevIdx[0],0]) * np.pi * self.WHEEL_DIAMETER/(self.RESOLUTION*tau)
        w = (np.sum(self.FOG_data[self.PrevIdx[1]:idx_fog,2]))/tau
        
        # delta pose 
        delta_x = v*tau*np.cos(self.Particles[:,2]).reshape(-1,1) + np.random.normal(0, self.COV_V, self.N).reshape(-1,1)
        delta_y = v*tau*np.sin(self.Particles[:,2]).reshape(-1,1) + np.random.normal(0, self.COV_V, self.N).reshape(-1,1)
        delta_theta = w*tau*np.ones_like(delta_x) + np.random.normal(0, self.COV_W, self.N).reshape(-1,1)
        
        self.Particles += np.hstack((delta_x, delta_y, delta_theta))
        self.PrevIdx = [idx_encoder, idx_fog]
        
        return
        
    
    def run(self):
        '''
        Peform SLAM on loaded data
        '''
        for idx, ts in enumerate(tqdm(self.TS_encoder[0:-1:2])):
            
            # Get the indices corresponding to each sensor to access each sensor data    
            idx_fog, idx_lidar, idx_encoder, idx_img = self.get_indices_from_TS( ts, idx)
            
            # Predit step for each time stamp
            self.predict(idx_fog, idx_encoder)
            
            # Update step for every 5th lidar scan
            if (idx_lidar % 10 == 0 and idx_lidar < self.lidar_data.shape[0]):
                self.update(idx_lidar, idx_img)
    

if __name__ == "__main__":
    N = 1#int(sys.argv[1])
    ts = tic()
    SLAMOBJ = SLAM(N)
    SLAMOBJ.run()
    toc(ts, 'SLAM')  
    
    plt.imshow(SLAMOBJ.MAP['map'],cmap="gray");
    plt.title("Occupancy grid map")
    
    np.save('path.npy', SLAMOBJ.PATH)
    np.save('map.npy', SLAMOBJ.MAP['map'])

