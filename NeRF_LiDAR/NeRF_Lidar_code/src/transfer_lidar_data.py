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

from Generate_feature import generate_gt_data,generate_simulation_data

def load_data(args,datadirs,lidarrender_paths,exp_dir):
    GT_ranges=[];GT_masks=[];Points_features=[];Points=[];Lidar_files=[];Proj_points=[];GT_Proj_points=[]
    for i in range(len(lidarrender_paths)):
        if os.path.exists(os.path.join(lidarrender_paths[i],'lidar_origins.npy')):
            lidar_origins = np.load(os.path.join(lidarrender_paths[i],'lidar_origins.npy'))
        else:
            lidar_origins = None
        if 'points_rgb_000.npy' in os.listdir(lidarrender_paths[i]) or 'points_rgb_0000.npy' in os.listdir(lidarrender_paths[i]): # i.e. GOT rgb feature of points.
            lidarrender_num = len([f for f in os.listdir(lidarrender_paths[i]) if f.startswith('points_') and f.endswith('.npy')])//3
        else:
            lidarrender_num = len([f for f in os.listdir(lidarrender_paths[i]) if f.startswith('points_') and f.endswith('.npy')])//2
        print('Rendered lidar frames: %d' % lidarrender_num)
        lidar2global = np.load(os.path.join(datadirs[i],'lidar_points','lidar2global.npy'))
        # 1. Generate GT rangeview images and masks
        if not args.return_proj:
            gt_ranges,gt_masks,lidar_files = \
            generate_gt_data(datadirs[i],exp_dir,lidarrender_num,check_idx=check_idx,W=args.W,log=args.log,return_proj=False,moving_mask=args.moving_mask)
        else:
            gt_ranges,gt_masks,lidar_files,gt_proj_points = \
            generate_gt_data(datadirs[i],exp_dir,lidarrender_num,check_idx=check_idx,W=args.W,log=args.log,return_proj = True,moving_mask=args.moving_mask)
            GT_Proj_points.append(gt_proj_points)
        GT_ranges.append(gt_ranges);GT_masks.append(gt_masks)
        # 2. Generate the training data
        # Note the saved format is different between the render data and the simulation data(for the purposes of not confusing them)
        if not args.return_proj:
            points_features,points = generate_simulation_data(lidar2global,lidarrender_num,lidarrender_paths[i],args,
                                check_idx=check_idx,W=args.W,log=args.log,datadir=datadirs[i],lidar_origins = lidar_origins,return_proj = False)
        else:
            points_features,points,proj_points = generate_simulation_data(lidar2global,lidarrender_num,lidarrender_paths[i],args,
                            check_idx=check_idx,W=args.W,log=args.log,datadir=datadirs[i],lidar_origins = lidar_origins,return_proj = True)
            Proj_points.append(proj_points)
        Points_features.append(points_features)
        Points.append(points)
        Lidar_files += lidar_files
    GT_ranges = np.concatenate(GT_ranges,axis =0 )
    GT_masks = np.concatenate(GT_masks,axis =0)
    Points_features=np.concatenate(Points_features,axis = 0)
    Points = np.concatenate(Points,axis = 0)  # in lidar coordinates
    with open(os.path.join(exp_dir,'lidar_files.txt'),'w') as f:
        for line in Lidar_files:
            f.write(line+'\n')

    np.save(os.path.join(exp_dir,'Points.npy'),Points) # points in lidar coordinates
    np.save(os.path.join(exp_dir,'points_features.npy'),Points_features) # points features
    np.save(os.path.join(exp_dir,'gt_masks.npy'),GT_masks)
    np.save(os.path.join(exp_dir,'gt_ranges.npy'),GT_ranges)
    if args.return_proj:
        Proj_points = np.concatenate(Proj_points,axis=0)
        GT_Proj_points = np.concatenate(GT_Proj_points,axis = 0)
        np.save(os.path.join(exp_dir,'Proj_points.npy'),Proj_points)
        np.save(os.path.join(exp_dir,'gt_Proj_points.npy'),GT_Proj_points)
    return 
if __name__ == '__main__': 
    ################################################################
    # Note: x y coordinates--- horizontal  ; z coordinates: vertical 
    # gt: righ forward up
    # nerf: right up back
    #0. parameters settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--expname', type=str, default='ray_drop')
    parser.add_argument('--lidarrender_num',type=int, default=50,nargs='+')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--pre_mask', action='store_true')
    parser.add_argument('--onehot_encoding', action='store_true')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--mask_loss', type=int, default=1)
    parser.add_argument('--W',type=int,default=1024)
    parser.add_argument('--delta', type=int, default=0)
    parser.add_argument('--val_percent',type=float, default=0.2)
    parser.add_argument('--log',type=int, default=1)
    parser.add_argument('--regression', type=int, default=0)
    parser.add_argument('--var',type = int, default= 1)
    parser.add_argument('--datadir',type=str, default = './', nargs='+')
    parser.add_argument('--lidardatapath', type=str, default = '',nargs='+')
    parser.add_argument('--mix_train',action='store_true')
    parser.add_argument('--ray_drop',type=str, default = '',nargs='+')
    parser.add_argument('--load_data',action='store_true')
    parser.add_argument('--depth_filter',action='store_true')
    parser.add_argument('--filter_thre',type=int,default= 0 )

    parser.add_argument('--semantic_align',action='store_true')
    parser.add_argument('--vgg',action= 'store_true')
    parser.add_argument('--return_proj',action='store_true')
    parser.add_argument('--feature_loss',action='store_true')
    parser.add_argument('--batch_size',type=int,default=1)
    parser.add_argument('--feature_loss_weights',type = float,default = 0.2)
    parser.add_argument('--vgg_weights',type=float,default = 0.5)
    parser.add_argument('--moving_mask',action = 'store_true')
    parser.add_argument('--roll',action= 'store_true')

    parser.add_argument('--simulation_path',type=str,)
    parser.add_argument('--dataset_path',type=str,)


    args = parser.parse_args()
    if args.feature_loss:
        args.return_proj = True

    exp_dir = os.path.join('./exp/',args.expname)
    os.makedirs(exp_dir,exist_ok=True)

    # experiments setting
    import json
    model_args={'bilinear':True,'regression':args.regression,'val_percent':args.val_percent,'transform':False,
        'mask_loss':args.mask_loss,'early_stop':args.early_stop,'delta' : args.delta,'vgg':args.vgg,
        'proj_xyz':args.return_proj,'feature_loss':args.feature_loss,'batch_size':args.batch_size,
        'feature_loss_weights':args.feature_loss_weights,'vgg_weights':args.vgg_weights,'roll':args.roll}
    jsObj = json.dumps(model_args)
    fileObject = open(os.path.join(exp_dir,'model_args.json'), 'w')
    fileObject.write(jsObj)
    fileObject.close()
    f = os.path.join(exp_dir, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    check_idx = 30
    if not args.mix_train:
        lidarrender_paths = [os.path.join('../Snerf/exp/lidar_rendering/',f) for f in args.lidardatapath]
        datadirs = ['../6cam_scene{}_revision'.format(f) for f in args.datadir]
    else:
        datadirs=[]
        lidarrender_paths =[]
        for raydata in args.ray_drop:
            lidarrender_path = '{}/{}'.format(args.simulation_path,raydata) 
            temp_lidarrender_paths = sorted(os.listdir(lidarrender_path))
            datadirs += [os.path.join('{}/{:04d}'.format(args.dataset_path,int(f))) for f in temp_lidarrender_paths]
            lidarrender_paths += [os.path.join(lidarrender_path,f) for f in temp_lidarrender_paths]

    print('lidarrender_paths include:{}'.format(lidarrender_paths))
    print('datadirs include: {}'.format(datadirs))

    if not args.load_data:
        load_data(args,datadirs,lidarrender_paths,exp_dir)
    ###### Load Data
    Points = np.load(os.path.join(exp_dir,'Points.npy'))
    Points_features = np.load(os.path.join(exp_dir,'points_features.npy'))
    GT_masks = np.load(os.path.join(exp_dir,'gt_masks.npy'))
    GT_ranges= np.load(os.path.join(exp_dir,'gt_ranges.npy'))
    if args.return_proj:
        Proj_points =  np.load(os.path.join(exp_dir,'Proj_points.npy'))
        GT_Proj_points =  np.load(os.path.join(exp_dir,'gt_Proj_points.npy'))

    data_depends = [Points_features,GT_masks,GT_ranges]
    if args.return_proj:
        data_depends += [Proj_points,GT_Proj_points]
    # 3. Ray Drop training
    print('start trainig ray_drop')
    print('Dataset size: %d' % Points_features.shape[0])
    device = torch.cuda.current_device()
    runner = ray_drop_learning(data_depends,n_channels=Points_features.shape[-1],device=device,**model_args)
    runner.train(savepath=exp_dir)