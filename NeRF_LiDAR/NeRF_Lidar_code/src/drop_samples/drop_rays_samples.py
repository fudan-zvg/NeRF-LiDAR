import argparse
from email import parser
from pyexpat import model
import re
import numpy as np
from lidar_utils import point_cloud_to_range_image
import os
from nerf2world import nerf_to_lidar
from lidar_utils import *
import torch
from torch.utils.data import DataLoader, random_split
from model.ray_drop_train import ray_drop_learning
import argparse
def generate_mask_vis(points,index,model_args,W=1024,log=True,max_dist=100.,datadir='',lidarrender_path='',total_num = 50,imgs=None,lidar_files=None):
    pred_mask,gt_mask,pred_range = runner.predict(index)
    gt_mask = gt_mask.squeeze(0)
    if model_args['regression']:
        pred_range = pred_range[0][0]   
    # get points
    # index = get_idx_in_val(index,total_num=total_num)


    index = 80* 6
    points_ = points[index]
    laser_scan = LaserScan(H=32,W=W,fov_up = 10.67, fov_down= -30.67)
    laser_scan.set_points(points_,remissions=None,semantic=None,rgb=None)
    laser_scan.do_range_projection()

    remain_points = points_
    # raw mask
    os.makedirs('mask_vis',exist_ok=True)
    with open('mask_vis/mask0_{}.obj'.format(index), 'w') as f:
        for v in remain_points:
            f.write("v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} \n"
                .format(v=v))

    remain_points = points_[laser_scan.proj_mask[laser_scan.proj_y,laser_scan.proj_x].astype(np.uint8)==1]
    pred_mask = np.argmax(pred_mask, axis=0)
    # semantic = imgs[i][...,1]
    # pred_mask[semantic == 10]==0
    acc = ((pred_mask[laser_scan.proj_mask==1]) == (gt_mask[laser_scan.proj_mask==1])).sum()/(gt_mask.shape[0]*gt_mask.shape[1])
    print('Acc rate:',acc)
    mask = (laser_scan.proj_mask==1) & (gt_mask==1)
    remain_points = points_[mask[laser_scan.proj_y,laser_scan.proj_x].astype(np.uint8)==1]
    # true_mask

    # prediction mask
    remain_points = laser_scan.proj_xyz[(pred_mask==1)&(laser_scan.proj_mask==1)]

    mask = (laser_scan.proj_mask==1) & (pred_mask==1)
    remain_points = points_[mask[laser_scan.proj_y,laser_scan.proj_x]==1]

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--total_num',type=int, default=50)
    parser.add_argument('--ckpt', type=int, default=0)
    parser.add_argument('--expname', type=str, default = 'ray_drop')
    args = parser.parse_args()
    expdir = os.path.join('exp',args.expname)
    index = args.index
    with open(os.path.join(expdir,"args.txt"), 'r') as exp_file:
        exp_setting = {}
        for line in exp_file:
            k, v = line.strip().split('=')
            exp_setting[k.strip()] = v.strip()
    imgs=np.load(os.path.join(expdir,'points_features.npy'))
    gt_masks= np.load(os.path.join(expdir,'gt_masks.npy'))
    gt_ranges= np.load(os.path.join(expdir,'gt_ranges.npy'))
    Points= np.load(os.path.join(expdir,'Points.npy'))
    datadirs=[];lidarrender_paths=[]
    scenes = []


    import json
    with open(os.path.join(expdir,'model_args.json'),'r') as f:
        model_args = json.load(f)
    lidar_files = []
    with open(os.path.join(expdir,'lidar_files.txt'),'r') as f:
        for line in f:
            lidar_files.append(line.strip('\n'))

    data_depends = [imgs,gt_masks,gt_ranges]
    runner= ray_drop_learning(data_depends,n_channels=imgs.shape[-1],**model_args)
    ckpt = [f for f in os.listdir(expdir) if f.endswith('.pth')]
   
    ckpt =sorted(ckpt)
    if args.ckpt:
        runner.load_ckpt(os.path.join(expdir,'{:05d}.pth'.format(args.ckpt)))
        print('load specific ckpt')
    else:
        runner.load_ckpt(os.path.join(expdir,ckpt[-1]))
        print('load ckpt {}'.format(os.path.join(expdir,ckpt[-1])))
    for raydata in ['ray_drop_v2', 'ray_drop_v3']:
            lidarrender_path = '../{}'.format(raydata) 
            temp_lidarrender_paths = os.listdir(lidarrender_path)
            datadirs += [os.path.join('../6cam_{}_revision'.format(f.split('_')[0])) for f in temp_lidarrender_paths]
            lidarrender_paths += [os.path.join(lidarrender_path,f) for f in temp_lidarrender_paths]
    i =0;scene_idx =23;j =0
    index = 80*6
    datadir = '../6cam_scene{}_revision'.format(scene_idx)
    simulation_path = 'ray_drop_v2/scene{}_finetune_v2_lidar_render_1100'.format(scene_idx)
    lidar2global = np.load(os.path.join(datadir,'lidar_points','lidar2global.npy'))
    nerf_points = np.load(os.path.join(simulation_path,'points_{:03d}.npy'.format(i)))
    points_semantic = np.load(os.path.join(simulation_path,'points_semantic_{:03d}.npy'.format(i)))
    points_rgb = np.load(os.path.join(simulation_path,'points_rgb_{:03d}.npy'.format(i)))
    points_ = nerf_to_lidar(nerf_points,lidar2global[i],datadir)
    points_features = imgs[index]
    pred_mask,pred_range = runner.test(points_features[None,...])
    pred_mask = np.argmax(pred_mask, axis=0)
    laser_scan = LaserScan(H=32,fov_up = 10.67, fov_down= -30.67)
    laser_scan.set_points(points_,remissions=None,semantic=points_semantic,rgb=None)
    laser_scan.do_range_projection()

    proj_mask = laser_scan.proj_mask
    remain_points = laser_scan.proj_xyz[(pred_mask==1)&(proj_mask==1)]
    remain_labels = laser_scan.proj_semantic[(pred_mask==1)&(proj_mask==1)]
    
