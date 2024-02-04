import numpy as np
import os
import argparse
def generate_mask(nerf_points,nerf_semantic):
    dist = np.linalg.norm(nerf_points, axis=-1)
    sky_mask = nerf_semantic!=10
    ## ROAD:20 BUILDING:19 in mseg
    ## ROAD:0 BUILDING:2 in cityscape
    mask0 =(dist <70)
    mask1 = (dist<60) | ((nerf_semantic==0) | (nerf_semantic==2))
    mask2 = (dist<50) | (nerf_semantic!=8)
    # mask3 = nerf_semantic!=0
    mask = mask1 &  sky_mask &mask2 &mask0
    return mask
def vis(datapath,expname,idx):
    result_path = os.path.join('lidar_points_compare',expname)
    os.makedirs(result_path,exist_ok=True)
    try:
        lidar_origin = np.load(os.path.join(datapath, 'lidar_origin_{:03d}.npy'.format(idx)))
    except FileNotFoundError:
        lidar_origin = np.load(os.path.join(datapath, 'lidar_origins.npy'))[idx]

    nerf_points = np.load(os.path.join(datapath, 'points_{:03d}.npy'.format(idx)))
    nerf_semantic = np.load(os.path.join(datapath, 'points_semantic_{:03d}.npy'.format(idx)))
  
    mask = generate_mask(nerf_points-lidar_origin,nerf_semantic)
    nerf_points = nerf_points[mask,:]
    
    with open(os.path.join(result_path, 'nerf_points_{:03d}.obj'.format(idx)),'w') as f:
        for v in nerf_points:
            f.write("v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} \n"
                .format(v=v))
    return
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expname', type=str,default='scene7_revision_v5_lidar_render')
    args = parser.parse_args()
    idx =0
    expname = args.expname
    datapath = os.path.join('lidar_rendering/',expname)
    vis(datapath,expname,idx)