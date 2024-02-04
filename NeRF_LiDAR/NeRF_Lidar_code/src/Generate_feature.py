import numpy as np
import torch
import os
import sys
from model.ray_drop_train import ray_drop_learning
import argparse
from lidar_utils import point_cloud_to_range_image, real_to_var
from lidar_rendering_vis import generate_mask
from nerf2world import nerf_to_lidar
from depth_filter import depth_filter
import torch.nn.functional as F

# points to images
def pcs2img(points,semantic=None,rgb=None,H=32,W = 1024,log=True,return_proj =False,moving_mask_name =None,return_scan = False):
    # the points transported into the function should be centered at the LIDAR sensor
    proj_rgb = None
    proj_semantics = None

    if return_scan:
        real,proj_semantics,proj_mask,proj_rgb,scan = point_cloud_to_range_image(points, isMatrix = True,H=H, W=W,semantic=semantic,rgb =rgb,
            return_semantic= True,  return_remission = False,return_mask=True,return_scan=True)
        return  real,proj_semantics,proj_mask,proj_rgb,scan
    if isinstance(points, np.ndarray):
        ## apply on simulated data
        if return_proj:
            real,proj_semantics,proj_mask,proj_rgb,proj_xyz = point_cloud_to_range_image(points, isMatrix = True,H=H, W=W,semantic=semantic,rgb =rgb,
            return_semantic= True,  return_remission = False,return_mask=True,return_points=True)
        else:
            real,proj_semantics,proj_mask,proj_rgb = point_cloud_to_range_image(points, isMatrix = True,H=H, W=W,semantic=semantic,rgb =rgb,
            return_semantic= True,  return_remission = False,return_mask=True)
    else:
        ## apply on GT data
        if return_proj:
            real,proj_mask,proj_xyz = point_cloud_to_range_image(points, isMatrix = False,H=H, W=W,semantic=semantic,rgb =None,
            return_semantic= False,  return_remission = False,return_mask=True,return_points=True,moving_mask_name=moving_mask_name)
        elif semantic is not None:
            real,proj_semantics,proj_mask = point_cloud_to_range_image(points, isMatrix = False,H=H, W=W,semantic=semantic,rgb =None,
            return_semantic= True,  return_remission = False,return_mask=True,moving_mask_name=moving_mask_name)
        else:
            real,proj_mask = point_cloud_to_range_image(points, isMatrix = False,H=H, W=W,semantic=semantic,rgb =None,
            return_semantic= False,  return_remission = False,return_mask=True,moving_mask_name=moving_mask_name)
    #Make negatives 0
    if log:
        real = np.where(real<0, 0, real) + 0.0001
        #Apply log
        real = ((np.log2(real+1)) / 6.5)
        #Make negatives 0
        real = np.clip(real, 0, 1)
    else:
        real = real/100.
        real = np.clip(real, 0, 1)
    if return_proj:
        return real,proj_semantics,proj_mask,proj_rgb,proj_xyz
    else:
        return real,proj_semantics,proj_mask,proj_rgb

# generate GT feature
def generate_gt_data(datadir,exp_dir,lidarrender_num,check_idx=0,W=1024,log=True,return_proj = False,moving_mask = False):
    gt_ranges=[];gt_masks=[];lidar_files=[];Proj_points = []
    for i in range(lidarrender_num):
        filename = os.path.join(datadir,'lidar_points','{:06d}.bin'.format(i))
        if moving_mask:
            moving_mask_name = os.path.join(datadir,'lidar_mask','{:04d}.txt'.format(i))
        else:
            moving_mask_name = None
        # if i== check_idx:
        #     np.save('lidar_point_coords.npy',np.fromfile(filename,dtype=np.float32).reshape(-1,5)[:,:3])
        if not return_proj:
            real,proj_semantics,proj_mask,proj_rgb = pcs2img(filename,semantic=None,W=W,log=log,return_proj=return_proj,moving_mask_name = moving_mask_name)
        else:
            real,proj_semantics,proj_mask,proj_rgb,proj_xyz = pcs2img(filename,semantic=None,W=W,log=log,return_proj=return_proj,moving_mask_name = moving_mask_name)
            Proj_points.append(proj_xyz)
        gt_ranges.append(real)
        gt_masks.append(proj_mask)
        lidar_files.append(filename)
    gt_ranges = np.stack(gt_ranges, axis = 0);gt_masks = np.stack(gt_masks, axis = 0)
    if not return_proj:
        return gt_ranges,gt_masks,lidar_files
    else:
        Proj_points=np.stack(Proj_points,axis=0)
        return gt_ranges,gt_masks,lidar_files,Proj_points
    
# Generate Simulation feature
def generate_simulation_data(lidar2global,lidarrender_num,lidarrender_path,args,check_idx=0,W=1024,log=True,datadir='',lidar_origins=None,\
                             simulation = False,return_proj = False,return_depends = False,pre_filter = True,verbose=False):
    import time;
    t0 = time.time()
    points_ranges=[];points_rgbs=[];points_semantics=[];points_masks=[];Points = [];Semantics = [];Var = [];Proj_points=[];Depth = [];Scans = []
    for i in range(lidarrender_num):
        if simulation:
            points = np.load(os.path.join(lidarrender_path,'points_{:04d}.npy').format(i))
            try:
                points_rgb = np.load(os.path.join(lidarrender_path,'points_rgb_{:04d}.npy').format(i))
            except:
                points_rgb = np.zeros((points.shape[0],3))
            points_semantic = np.load(os.path.join(lidarrender_path,'points_semantic_{:04d}.npy').format(i))
        else:
            # points = np.load(os.path.join(lidarrender_path,'points_{:03d}.npy').format(i))
            # try:
            #     points_rgb = np.load(os.path.join(lidarrender_path,'points_rgb_{:03d}.npy').format(i))
            # except:
            #     points_rgb = np.zeros((points.shape[0],3))
            # points_semantic = np.load(os.path.join(lidarrender_path,'points_semantic_{:03d}.npy').format(i))
            points = np.load(os.path.join(lidarrender_path,'points_{:04d}.npy').format(i))
            try:
                points_rgb = np.load(os.path.join(lidarrender_path,'points_rgb_{:04d}.npy').format(i))
            except:
                points_rgb = np.zeros((points.shape[0],3))
            points_semantic = np.load(os.path.join(lidarrender_path,'points_semantic_{:04d}.npy').format(i))     
        # deal data from lidarsim
        if os.path.exists(os.path.join(lidarrender_path,'depth_{:04d}.npy').format(i)) \
            or os.path.exists(os.path.join(lidarrender_path,'depth_{:03d}.npy').format(i)): 
            try:
                depth = np.load(os.path.join(lidarrender_path,'depth_{:04d}.npy').format(i))
            except:
                depth = np.load(os.path.join(lidarrender_path,'depth_{:03d}.npy').format(i))
            Depth.append(depth)
        ## nerf --- world --- lidar
        points_ = nerf_to_lidar(points,lidar2global[i],datadir=datadir)
        
        Points.append(points_);Semantics.append(points_semantic)
        if args.pre_mask:
            mask = generate_mask(points_,points_semantic)
            points_ = points_[mask]
            points_rgb = points_rgb[mask]
            points_semantic = points_semantic[mask]
        if args.depth_filter == 1 and pre_filter:
            if args.semantic_align:
                depth_filter_mask = depth_filter(points_,points_semantic,return_mask = True,width = 1,threshold=args.filter_thre)
            else:
                depth_filter_mask = depth_filter(points_,return_mask = True,width= 5)
            points_= points_[depth_filter_mask]
            points_semantic = points_semantic[depth_filter_mask]
            points_rgb = points_rgb[depth_filter_mask]
        if return_depends:
            real,proj_semantics,proj_mask,proj_rgb,scan  = pcs2img(points_,semantic=points_semantic,rgb=points_rgb,W=W,log=log,return_scan= True)
            Scans.append(scan)
        elif not return_proj:
            real,proj_semantics,proj_mask,proj_rgb = pcs2img(points_,semantic=points_semantic,rgb=points_rgb,W=W,log=log)
        else:
            real,proj_semantics,proj_mask,proj_rgb,proj_points = pcs2img(points_,semantic=points_semantic,rgb=points_rgb,W=W,log=log,return_proj = True)
            Proj_points.append(proj_points)
        points_ranges.append(real);points_semantics.append(proj_semantics);points_masks.append(proj_mask);points_rgbs.append(proj_rgb);
        if args.var:
            Var.append(real_to_var(real,size = 2))
    
    # cat all the features
    if args.normalize:
        if len(Depth) > 0:
            points_features = [np.concatenate([points_ranges[i][...,None],points_semantics[i][...,None]/20., points_ranges[i][...,None]<69],axis=-1) for i in range(lidarrender_num)]
        else:
            points_features = [np.concatenate([points_ranges[i][...,None],points_semantics[i][...,None]/20.],axis=-1) for i in range(lidarrender_num)]

    elif args.onehot_encoding:
        mask_surface = (points_semantics[i][...,None]==1) | (points_ranges[i][...,None]==0) 
        mask_manmade = (points_semantics[i][...,None]>1) & (points_ranges[i][...,None]<8) 
        mask_vegetation = (points_semantics[i][...,None]==8) | (points_semantics[i][...,None]==9)
        # points_features = [np.concatenate([points_ranges[i][...,None],mask_vegetation,mask_surface,mask_manmade,points_masks[i][...,None]],axis=-1) for i in range(lidarrender_num)]
        points_features = [np.concatenate([points_ranges[i][...,None],mask_vegetation,mask_surface,mask_manmade,],axis=-1) for i in range(lidarrender_num)]

    else:
        if not args.var:
        # points_features = [np.concatenate([points_ranges[i][...,None],points_semantics[i][...,None],points_masks[i][...,None],points_rgbs[i]],axis=-1) for i in range(lidarrender_num)]
            points_features = [np.concatenate([points_ranges[i][...,None],points_semantics[i][...,None],points_rgbs[i]],axis=-1) for i in range(lidarrender_num)]
        else:
            points_features = [np.concatenate([points_ranges[i][...,None],points_semantics[i][...,None],points_rgbs[i],Var[i][...,None]],axis=-1) for i in range(lidarrender_num)]
    points_features = np.stack(points_features, axis=0)
    Points = np.stack(Points, axis = 0); Semantics = np.stack(Semantics, axis = 0)
    t1 = time.time()
    if verbose:
        print(f'time consuming: {t1-t0}')
    if return_depends:
        return points_features, [Points,Semantics,Scans]
    if not return_proj:
        return points_features,Points
    else:
        Proj_points = np.stack(Proj_points,axis=0)
        return points_features,Points,Proj_points