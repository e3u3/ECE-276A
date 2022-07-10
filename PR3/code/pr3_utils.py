import numpy as np
import os
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler

def load_data(file_name):
    '''
    function to read visual features, IMU measurements and calibration parameters
    Input:
        file_name: the input data file. Should look like "XX.npz"
    Output:
        t: time stamp
            with shape 1*t
        features: visual feature point coordinates in stereo images, 
            with shape 4*n*t, where n is number of features
        linear_velocity: velocity measurements in IMU frame
            with shape 3*t
        angular_velocity: angular velocity measurements in IMU frame
            with shape 3*t
        K: (left)camera intrinsic matrix
            with shape 3*3
        b: stereo camera baseline
            with shape 1
        imu_T_cam: extrinsic matrix from (left)camera to imu, in SE(3).
            with shape 4*4
    '''
    with np.load(file_name) as data:
    
        t = data["time_stamps"] # time_stamps
        features = data["features"] # 4 x num_features : pixel coordinates of features
        linear_velocity = data["linear_velocity"] # linear velocity measured in the body frame
        angular_velocity = data["angular_velocity"] # angular velocity measured in the body frame
        K = data["K"] # intrindic calibration matrix
        b = data["b"] # baseline
        imu_T_cam = data["imu_T_cam"] # Transformation from left camera to imu frame 
    
    return t,features,linear_velocity,angular_velocity,K,b,imu_T_cam

def parse_data(data):
    '''
    Utility function to parse the npy file
    '''
    ang_vel = data[3]
    lin_vel = data[2]
    features = data[1]
    TS = data[0]
    imu_T_cam = data[-1]
    K = data[-3]
    b = data[-2]
    return ang_vel, lin_vel, features, TS, imu_T_cam, K, b

def load_data_from_dir(dir_name):
    '''
    function to read visual features, IMU measurements and calibration parameters
    Input:
        file_name: the input data file. Should look like "XX.npz"
    Output:
        t: time stamp
            with shape 1*t
        features: visual feature point coordinates in stereo images, 
            with shape 4*n*t, where n is number of features
        linear_velocity: velocity measurements in IMU frame
            with shape 3*t
        angular_velocity: angular velocity measurements in IMU frame
            with shape 3*t
        K: (left)camera intrinsic matrix
            with shape 3*3
        b: stereo camera baseline
            with shape 1
        imu_T_cam: extrinsic matrix from (left)camera to imu, in SE(3).
            with shape 4*4
    '''
    
    data = {}
    
    file_list = os.listdir(dir_name)
    for file in file_list:
        print(os.path.join(dir_name, file))
        data[file[:-4]] = np.load(os.path.join(dir_name, file))
    
    return data


def visualize_trajectory_2d(pose, mean = None, path_name="Unknown",show_ori=False):
    '''
    function to visualize the trajectory in 2D
    Input:
        pose:   4*4*N matrix representing the camera pose, 
                where N is the number of poses, and each
                4*4 matrix is in SE(3)
    '''
    fig,ax = plt.subplots(figsize=(5,5))
    n_pose = pose.shape[2]
    ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name)
    ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
    ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
    if mean is not None:
        ax.scatter(mean[0,:],mean[1,:],s = 2,label="landmark")
  
    if show_ori:
        select_ori_index = list(range(0,n_pose,max(int(n_pose/50), 1)))
        yaw_list = []
        
        for i in select_ori_index:
            _,_,yaw = mat2euler(pose[:3,:3,i])
            yaw_list.append(yaw)
    
        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
        ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
            color="b",units="xy",width=1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(False)
    ax.legend()
    plt.savefig('test.jpeg')
    plt.show(block=True)

    return fig, ax

def Homogenize(x):
    '''
    Converts points from inhomogeneous to homogeneous coordinates
    '''
    return np.vstack((x,np.ones((1,x.shape[1]))))

def Dehomogenize(x):
    '''
    Converts points from homogeneous to inhomogeneous coordinates
    '''
    return x[:-1]/x[-1]

def get_features_current_frame(features, idx, K = 10):
    '''
    Get features in current frame corresponding to 
    time index idx.
    
    Returns a 3xM array with invalid points marked as NaN
    '''
    f = features[:,:,idx]
    
    # Filter out features not presnet i.e. (-1)
    #f = f[:,f[0,:] > 0]
    
    # Set invalid points as NAN
    f[:,f[0,:] < 0] = np.nan
    # Sample every kth point
    f = f[:,0:-1:K]
    valid_landmarks = np.where(~np.isnan(f[0,:]))
    
    return f,valid_landmarks[0].tolist()

def get_idx(listA, list_global):
    '''
    Utility function to parse through listA and get indices that are present in 
    list_global and indices that appear for the first time
    '''
    list_new = []
    list_old = []
    for i in listA:
        if i in list_global:
            list_old.append(i)
        else:
            list_new.append(i)
    
    return list_old, list_new

def hat(x):
    '''
    Utility function to create skew symmetric matrix 
    given a 3D vector
    '''
    assert x.shape[0] == 3, 'Expect a 3D vector for hat operator'
    
    x_hat = np.array([[0, -x[2],x[1]],
                      [x[2],0,-x[0]],
                      [-x[1],x[0],0]])
    
    return x_hat

def hat_alt(x):
    '''
    Utility function to create special hat operator
    given a 6D vector
    '''
    assert x.shape[0] == 6, 'Expect a 3D vector for alt_hat operator'
    
    v_hat = hat(x[0:3])
    w_hat = hat(x[3:6])
    
    op1 = np.hstack((w_hat, v_hat))
    op2 = np.hstack((np.zeros((3,3)), w_hat))
    op = np.vstack((op1, op2))    
    return op

def twist_matrix(x):
    '''
    Utility function to create a twist matrix given 
    a 6D vector
    '''
    assert x.shape[0] == 6, 'Expect a 6d vector in twist matrix'
    tm = np.vstack((np.column_stack((hat(x[3:6]), x[0:3])),np.zeros((1,4))))
    return tm


if __name__ == "__main__":
    print('MAIN')
