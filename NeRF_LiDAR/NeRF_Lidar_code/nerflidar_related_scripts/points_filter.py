from turtle import home
import numpy as np
import os
from logging.handlers import RotatingFileHandler
import torch

import numpy as np
import random
import functools
from os import path

# from utils.render_utils import generate_renderpath

from tqdm import tqdm
# from vis import visualize_depth
from scipy.spatial.transform import Rotation as R
import imageio
from matplotlib import cm

def normalize(x):
        return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = viewmatrix(vec2, up, center)
    c2w=c2w.astype(np.float32)
    return c2w

def recenter_poses(poses):
    
    poses_ = poses.copy()
    bottom = np.reshape([0,0,0,1.], [1,4])
    
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2).astype(np.float32)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)
    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = np.concatenate([poses_[:,:3,:4], bottom], -2).astype(np.float32)
    return poses, c2w
def weigted_median_filter(all_points,point):
    related_points= all_points[np.linalg.norm(all_points-point,axis = 1)<0.1]
    new_point = np.median(related_points,axis=0)
    return new_point

def weigted_median_filter_polar(point,points_polar,degree):
    related_points= points_polar[((points_polar[:,2]-point[2])<degree)&((points_polar[:,2]-point[2])>-degree)]
    new_point = np.median(related_points,axis=0)
    
    return new_point
def transfer_back(temp_point):
    point_new = temp_point[0]*np.array([np.sin(temp_point[2]*np.pi/180)*np.cos(temp_point[1]*np.pi/180),np.sin(temp_point[1]*np.pi/180),np.cos(temp_point[2]*np.pi/180)*np.cos(temp_point[1]*np.pi/180)])
    return point_new
def filter_func(points,lidar_origin,degree):
    points_new =[]
    for i in range(32):
        # translate points into polar coordinates, then apply weigted_median_filter
        points_temp = points[i]- lidar_origin
        
        # horizontal_vec = np.array([0,-1,0])
        points_polar = []
        for i in range(len(points_temp)):
            vec1 = points_temp[i]
            vec1 = vec1 / np.linalg.norm(vec1)
            temp_point = np.array([np.linalg.norm(points_temp[i]), 90-np.arccos(points_temp[i][1]/np.linalg.norm(points_temp[i]))*180/np.pi, np.arctan2(vec1[0],vec1[2])*180/np.pi])
            points_polar.append(temp_point)
        points_polar = np.stack(points_polar, axis=0)
        new_points=[]
        for point in points_polar:
            new_point = weigted_median_filter_polar(point,points_polar,degree)
            
            new_points.append(transfer_back(new_point))

        points_new.append(new_points)
        
    filtered_points = np.concatenate(points_new,axis=0) + lidar_origin
    return filtered_points
    

if __name__ == '__main__':
    datapath = '/SSD_DISK/users/zhangjunge/scene7_revision_lidar_render'
    points = np.load(os.path.join(datapath,'points.npy'))
    camera_parameters = np.load('/SSD_DISK/users/zhangjunge/scene7_camera_parameters/c2w.npy')
    lidar_points_path = '/SSD_DISK/users/zhangjunge/lidar_points_scene7'
    points_center = np.load(os.path.join(lidar_points_path,'points{:03d}.npy'.format(15)))[:,-1]

    
    points_1 = points_center.copy();points_1[2] = 0.
    camera_parameters_inv = np.linalg.inv(camera_parameters)


    points_ = points_center[:3]@ camera_parameters_inv[:3,:3].T + camera_parameters_inv[:3,3]
    points_1 = points_1[:3] @ camera_parameters_inv[:3,:3].T + camera_parameters_inv[:3,3]
    
    # points_center_next = points_center_next @ camera_parameters[:3,:3]
    # velocity = (points_center_next - points_center)/0.5
    poses = np.load(os.path.join('/SSD_DISK/users/zhangjunge/6cam_scene7_revision','poses_bounds.npy'))[:,:-4].reshape(-1,3,5)[:,:3,:4]
    poses = np.concatenate(
        [poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], 2)
    # poses : right down backwards
    poses, c2w = recenter_poses(poses)
    c2w_inv = np.linalg.inv(c2w)
    # origin

    lidar_origin = points_ @ c2w[:3,:3] + c2w_inv[:3,3]
    points = points.reshape(32,-1,3)
    degree=4

    
    filtered_points = filter_func(points,lidar_origin,degree)
    np.save('filtered_points_{:02d}.npy'.format(degree),filtered_points)