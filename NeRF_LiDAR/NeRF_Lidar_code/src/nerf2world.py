import numpy as np
import os
from utils import recenter_poses

def nerf_to_lidar_ksc(nerf_points,lidar2global,datadir):
    # nerf points: N*3 already centered at the origin
    # lidar2global : ego2global @ lidar2ego(fixed)

    camera_parameters = np.load(os.path.join(datadir,'c2w.npy'))
    # c2w_front = ego2global @cam2ego
    # lidar2cam = inv(cam2ego) @ lidar2ego
    try:
        c2w_inv = np.load(os.path.join(datadir,'c2w_recenter_transform.npy'))
        c2w = np.linalg.inv(c2w_inv)
    except:
        c2w = np.load(os.path.join(datadir,'c2w_recenter.npy'))
        c2w_inv = np.linalg.inv(c2w)
    # points @ c2w
    nerf_points = np.concatenate([nerf_points,np.ones((nerf_points.shape[0],1))],axis=1)
    world_points = (nerf_points @ c2w.T) @ (camera_parameters).T @ np.linalg.inv(lidar2global).T
    return world_points[:,:3] , (c2w.T) @ (camera_parameters).T @ np.linalg.inv(lidar2global).T
def nerf_to_lidar(nerf_points,lidar2global,datadir):
    # nerf points: N*3 already centered at the origin
    # lidar2global : ego2global @ lidar2ego(fixed)

    camera_parameters = np.load(os.path.join(datadir,'c2w.npy'))
    # c2w_front = ego2global @cam2ego
    # lidar2cam = inv(cam2ego) @ lidar2ego
    try:
        c2w_inv = np.load(os.path.join(datadir,'c2w_recenter_transform.npy'))
        c2w = np.linalg.inv(c2w_inv)
    except:
        c2w = np.load(os.path.join(datadir,'c2w_recenter.npy'))
        c2w_inv = np.linalg.inv(c2w)
    # c2w_inv = np.linalg.inv(c2w)
    # points @ c2w
    nerf_points = np.concatenate([nerf_points,np.ones((nerf_points.shape[0],1))],axis=1)
    world_points = (nerf_points @ c2w.T) @ (camera_parameters).T @ np.linalg.inv(lidar2global).T
    return world_points[:,:3]
def nerf_to_world(nerf_points,datadir= ''):
    # nerf points: N*3 already centered at the origin
    # lidar2global : ego2global @ lidar2ego(fixed)
    camera_parameters = np.load(os.path.join(datadir,'c2w.npy'))
    # c2w_front = ego2global @cam2ego
    # lidar2cam = inv(cam2ego) @ lidar2ego
    # poses : right down backwards
    try:
        c2w_inv = np.load(os.path.join(datadir,'c2w_recenter_transform.npy'))
        c2w = np.linalg.inv(c2w_inv)
    except:
        c2w = np.load(os.path.join(datadir,'c2w_recenter.npy'))
        c2w_inv = np.linalg.inv(c2w)
    # points @ c2w
    nerf_points = np.concatenate([nerf_points,np.ones((nerf_points.shape[0],1))],axis=1)
    world_points = (nerf_points @ c2w.T) @ (camera_parameters).T
    return world_points[:,:3]

def world_to_nerf(lidarpoints,lidar2global=None,datadir= ''):
    camera_parameters = np.load(os.path.join(datadir,'c2w.npy'))
    # c2w_front = ego2global @cam2ego
    # lidar2cam = inv(cam2ego) @ lidar2ego
    c2w = np.load(os.path.join(datadir,'c2w_recenter.npy'))

    c2w_inv = np.linalg.inv(c2w)
    # points @ c2w
    if lidar2global is None:
        # this time the points are in lidar coordinates
        lidarpoints = np.concatenate([lidarpoints,np.ones((lidarpoints.shape[0],1))],axis=1)
        # world_points = (nerf_points @ c2w.T) @ (camera_parameters).T @ np.linalg.inv(lidar2global).T
        nerf_points = lidarpoints @ np.linalg.inv(camera_parameters).T @ c2w_inv.T
    return nerf_points[:,:3]

def get_lidar2global(simulation_path,datadir):
    ego_trace = os.path.join(simulation_path, 'ego_trace.npy')
    ego_trace = np.load(ego_trace)
    cam2global = np.load(os.path.join(datadir,'c2w.npy')) #cam2global
   
    lidar2cam =np.load(os.path.join(datadir,'lidar2cam.npy')).astype(np.float32)
    lidar2global = cam2global @ lidar2cam 
    global_origins = nerf_to_world(ego_trace,datadir)[:-1,:]
    lidar2globals = np.broadcast_to(lidar2global[None,...],(global_origins.shape[0],4,4))

    global_origins = np.concatenate([global_origins,np.ones((global_origins.shape[0],1))],axis=-1)[:,:,None]
    lidar2globals = np.concatenate([lidar2globals[:,:,:3],global_origins],axis=-1)
    return lidar2globals
if __name__ == '__main__':
    lidar2global = np.load('lidar_points/lidar2global.npy')
    lidar_points =  np.load('lidar_points/points{:03d}.npy'.format(5)).transpose()[:,:3]
    import pdb;pdb.set_trace()
    nerf_poitns = world_to_nerf(lidar_points,lidar2global=None)
    with open('mask_vis/mask_lidar.obj', 'w') as f:
        for v in lidar_points:
            f.write("v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} \n"
                .format(v=v))