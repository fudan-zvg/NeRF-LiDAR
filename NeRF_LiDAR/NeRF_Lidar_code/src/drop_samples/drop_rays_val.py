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
    index = get_idx_in_val(index,total_num=total_num)
  
    points_ = points[index]
    laser_scan = LaserScan(H=32,W=W,fov_up = 10.67, fov_down= -30.67)
    laser_scan.set_points(points_,remissions=None,semantic=None,rgb=None)
    laser_scan.do_range_projection()

    if model_args['delta']:
        if log:
            real = np.where(laser_scan.proj_range<0, 0, laser_scan.proj_range) + 0.0001
            #Apply log
            real = ((np.log2(real+1)) / 6.5)
            #Make negatives 0
            real = np.clip(real, 0, 1)
            
        else:
            real=laser_scan.proj_range/max_dist
            real = np.clip(real, 0, 1)
        pred_range =np.clip(2*pred_range-1+ real,0,1)
    print('index is %d' % index)

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
    # remain_points = laser_scan.proj_xyz[mask,:]
    remain_points = points_[mask[laser_scan.proj_y,laser_scan.proj_x].astype(np.uint8)==1]
    # true_mask
    with open('mask_vis/mask2_{}.obj'.format(index), 'w') as f:
        for v in remain_points:
            f.write("v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} \n"
                .format(v=v))
    # prediction mask
    remain_points = laser_scan.proj_xyz[(pred_mask==1)&(laser_scan.proj_mask==1)]
    with open('mask_vis/mask3_{}.obj'.format(index), 'w') as f:
        for v in remain_points:
            f.write("v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} \n"
                .format(v=v))
    mask = (laser_scan.proj_mask==1) & (pred_mask==1)
    remain_points = points_[mask[laser_scan.proj_y,laser_scan.proj_x]==1]
    with open('mask_vis/mask4_{}.obj'.format(index), 'w') as f:
        for v in remain_points:
            f.write("v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} \n"
                .format(v=v))


    if model_args['regression']:
        gt = runner.return_gt(index,val = False)
        gt_range = gt['range'].detach().cpu().numpy()
        gt_mask = gt['mask'].detach().cpu().numpy()
        # with pred_mask
        remain_points = laser_scan.proj_xyz[(pred_mask==1)&(laser_scan.proj_mask==1)]
        refine_dist = pred_range[(pred_mask==1)&(laser_scan.proj_mask==1)]
        # from [0,1] to real distance
        def dist_backtoreal(refine_dist,log=True,max_dist=100.):
            if log:
                refine_dist = 2**(refine_dist*6.5)-1
            else:
                refine_dist = max_dist *refine_dist
            return refine_dist
        refine_dist = dist_backtoreal(refine_dist,log=log,max_dist=max_dist)
        refine_points =  refine_dist[...,None] *(remain_points/np.linalg.norm(remain_points,axis=-1,keepdims=True) )
        with open('mask_vis/mask_refine_{}.obj'.format(index), 'w') as f:
            for v in refine_points:
                f.write("v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} \n"
                    .format(v=v))
        # with gt_mask
        remain_points = laser_scan.proj_xyz[(gt_mask==1)&(laser_scan.proj_mask==1)]
        refine_dist = pred_range[(gt_mask==1)&(laser_scan.proj_mask==1)]
        # from [0,1] to real distance
        refine_dist = dist_backtoreal(refine_dist,log=log,max_dist=max_dist)

        refine_points =  refine_dist[...,None] *(remain_points/np.linalg.norm(remain_points,axis=-1,keepdims=True) )
        with open('mask_vis/mask_refine_gtmask_{}.obj'.format(index), 'w') as f:
            for v in refine_points:
                f.write("v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} \n"
                    .format(v=v))
        # with no mask
        remain_points = laser_scan.proj_xyz[(laser_scan.proj_mask==1)]
        refine_dist = pred_range[(laser_scan.proj_mask==1)]
        refine_dist = dist_backtoreal(refine_dist,log=log,max_dist=max_dist)
        refine_points =  refine_dist[...,None] *(remain_points/np.linalg.norm(remain_points,axis=-1,keepdims=True) )
        with open('mask_vis/mask_refine_nomask_{}.obj'.format(index), 'w') as f:
            for v in refine_points:
                f.write("v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} \n"
                    .format(v=v))
    # GT lidar
    if lidar_files is not None:
        lidar_points =  np.fromfile(lidar_files[index],dtype=np.float32).reshape(-1,5)[:,:3]
        with open('mask_vis/lidar_{}.obj'.format(index), 'w') as f:
            for v in lidar_points:
                f.write("v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} \n"
                    .format(v=v))
    return acc
    # import pdb;pdb.set_trace()

    # remain_points = laser_scan.proj_xyz[(pred_mask==1)&(laser_scan.proj_mask==1)]
    # refine_dist = pred_range[(pred_mask==1)&(laser_scan.proj_mask==1)]
    # # from [0,1] to real distance
    # refine_dist = 2**(refine_dist*6.5)-1
    # refine_points =  refine_dist[...,None] *(remain_points/np.linalg.norm(remain_points,axis=-1,keepdims=True) )
    # with open('mask_vis/mask5_{}.obj'.format(index), 'w') as f:
    #     for v in refine_points:
    #         f.write("v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} \n"
    #             .format(v=v))

    # # remain_points = laser_scan.proj_xyz[(gt_mask==1)&(laser_scan.proj_mask==1)]
    # # refine_dist = gt_range[(gt_mask==1)&(laser_scan.proj_mask==1)]
    # remain_points = laser_scan.proj_xyz[(laser_scan.proj_mask==1)]
    # refine_dist = gt_range[(laser_scan.proj_mask==1)]
    # # from [0,1] to real distance
    # refine_dist = 2**(refine_dist*6.5)-1
    # refine_points =  refine_dist[...,None] *(remain_points/np.linalg.norm(remain_points,axis=-1,keepdims=True) )
    # with open('mask_vis/mask5_{}.obj'.format(index), 'w') as f:
    #     for v in refine_points:
    #         f.write("v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} \n"
    #             .format(v=v))
    # gt_mask = gt['mask']

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

    # model_args={'bilinear':True,'regression':True,'val_percent':args.val_percent,'transform':True}
    import json
    with open(os.path.join(expdir,'model_args.json'),'r') as f:
        model_args = json.load(f)
    lidar_files = []
    with open(os.path.join(expdir,'lidar_files.txt'),'r') as f:
        for line in f:
            lidar_files.append(line.strip('\n'))
    runner= ray_drop_learning(imgs,gt_masks,gt_ranges,n_channels=imgs.shape[-1],**model_args)
    ckpt = [f for f in os.listdir(expdir) if f.endswith('.pth')]
    def get_idx_in_val(idx,total_num = 50):
        dataset=range(total_num)
        n_val = int(total_num * model_args['val_percent'])
        n_train = total_num - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
        index = val_set[idx]

        return index
    ckpt =sorted(ckpt)

    if args.ckpt:
        runner.load_ckpt(os.path.join(expdir,'{:05d}.pth'.format(args.ckpt)))
        print('load specific ckpt')
    else:
        runner.load_ckpt(os.path.join(expdir,ckpt[-1]))
        print('load ckpt {}'.format(os.path.join(expdir,ckpt[-1])))
    if os.path.exists('./mask_vis'):
        os.system('rm -rf ./mask_vis')
    Acc= []
    for i in range(int(model_args['val_percent']*imgs.shape[0])):
        acc= generate_mask_vis(Points,i,model_args,W = int(exp_setting['W']),log=int(exp_setting['log']),
        datadir = exp_setting['datadir'],
        lidarrender_path = exp_setting['lidardatapath'],total_num = imgs.shape[0],imgs=imgs,lidar_files=lidar_files)
        Acc.append(acc)
    print('Avg Acc is: {:04f}' .format(np.mean(np.array(Acc))))