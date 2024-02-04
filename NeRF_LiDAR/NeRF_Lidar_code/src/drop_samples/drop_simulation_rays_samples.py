from tkinter.tix import Tree
import numpy as np
import os
from nerf2world import nerf_to_world,nerf_to_lidar
from transfer_lidar_data import pcs2img
from model.ray_drop_train import ray_drop_learning
from transfer_lidar_data import generate_simulation_data
from lidar_utils import *
from tqdm import tqdm
import argparse
from depth_filter import depth_filter

def write_pointsandlabels(j,savepath,points,labels):
    remain_points = points
    with open(os.path.join(savepath,'velodyne',"{:06d}.bin".format(j)), "wb") as binary_file:
        for v in remain_points:
            v.astype(np.float32).tofile(binary_file)
    remain_points = labels
    with open(os.path.join(savepath,'labels',"{:06d}.label".format(j)), "wb") as binary_file:
        for v in remain_points:
                v.astype(np.uint32).tofile(binary_file)
def drop_simulation(simulation_path,savepath,lidar2globals,runner,model_args,exp_setting,save_near=False,depth_refine=False,max_dist=100.,datadir='',
                nodrop=False,random_drop = False,mask_thre = 0.5):
   
    lidarrender_num = len([f for f in os.listdir(simulation_path) if f.startswith('points')])//3
    Remain_points =[];Remain_labels=[]

    if nodrop:
        for i in range(lidarrender_num):
            nerf_points = np.load(os.path.join(simulation_path,'points_{:04d}.npy'.format(i)))
            points_ = nerf_to_lidar(nerf_points,lidar2globals[i],datadir)
            points_semantic = np.load(os.path.join(simulation_path,'points_semantic_{:04d}.npy'.format(i)))
            Remain_points.append(points_)
            Remain_labels.append(points_semantic)
    elif random_drop:
        for i in range(lidarrender_num):
            nerf_points = np.load(os.path.join(simulation_path,'points_{:04d}.npy'.format(i)))
            points_ = nerf_to_lidar(nerf_points,lidar2globals[i],datadir)
            points_semantic = np.load(os.path.join(simulation_path,'points_semantic_{:04d}.npy'.format(i)))
            random_coords = np.random.randint(len(points_),size=28000)
            points_ = points_[random_coords]  
            points_semantic = points_semantic[random_coords]
            Remain_points.append(points_)
            Remain_labels.append(points_semantic)
    else:
        Points_features,Points = generate_simulation_data(lidar2globals,lidarrender_num,simulation_path,args,W=1024,log=True,datadir=datadir,simulation = True)
        # pred_mask,pred_range = runner.test(cat_img[None,...])
        print('Ray Drop: datadir{} simulation_path{}'.format(datadir,simulation_path))
        for i in tqdm(range(Points_features.shape[0])):
            points_features = Points_features[i]
            nerf_points = np.load(os.path.join(simulation_path,'points_{:04d}.npy'.format(i)))
            points_semantic = np.load(os.path.join(simulation_path,'points_semantic_{:04d}.npy'.format(i)))
            points_rgb = np.load(os.path.join(simulation_path,'points_rgb_{:04d}.npy'.format(i)))
            points_ = nerf_to_lidar(nerf_points,lidar2globals[i],datadir)
            if args.depth_filter:
                depth_filter_mask = depth_filter(points_,return_mask = True)[...,0]
                # points_features = points_features[depth_filter_mask.reshape(32,-1)]
                points_= points_[depth_filter_mask]
                points_semantic = points_semantic[depth_filter_mask]
            pred_mask,pred_range = runner.test(points_features[None,...])
            laser_scan = LaserScan(H=32,fov_up = 10.67, fov_down= -30.67)
            laser_scan.set_points(points_,remissions=None,semantic=points_semantic,rgb=None)
            laser_scan.do_range_projection()
            if not mask_thre:
                pred_mask = np.argmax(pred_mask, axis=0)
            else:
                pred_mask = pred_mask[1]>args.mask_thre
            if not save_near:
                mask = (pred_mask ==1)  & (laser_scan.proj_mask ==1)
                remain_points = points_[mask[laser_scan.proj_y,laser_scan.proj_x]==1]
                remain_labels = points_semantic[mask[laser_scan.proj_y,laser_scan.proj_x]==1]
            else:
                proj_mask = laser_scan.proj_mask
                remain_points = laser_scan.proj_xyz[(pred_mask==1)&(proj_mask==1)]
                remain_labels = laser_scan.proj_semantic[(pred_mask==1)&(proj_mask==1)]

                if model_args['regression'] and depth_refine:
                    pred_range = pred_range[0][0]
                    if model_args['delta']:
                        if int(exp_setting['log']):
                            real = np.where(laser_scan.proj_range<0, 0, laser_scan.proj_range) + 0.0001
                            #Apply log
                            real = ((np.log2(real+1)) / 6.5)
                            #Make negatives 0
                            real = np.clip(real, 0, 1)
                            
                        else:
                            real=laser_scan.proj_range/max_dist
                            real = np.clip(real, 0, 1)
                        pred_range =np.clip(2*pred_range-1+ real,0,1)
                    # remain_points = laser_scan.proj_xyz[(pred_mask==1)&(laser_scan.proj_mask==1)]
                    refine_dist = pred_range[(pred_mask==1)&(laser_scan.proj_mask==1)]
                    # from [0,1] to real distance
                    def dist_backtoreal(refine_dist,log=True,max_dist=max_dist):
                        if log:
                            refine_dist = 2**(refine_dist*6.5)-1
                        else:
                            refine_dist = max_dist *refine_dist
                        return refine_dist
                    refine_dist = dist_backtoreal(refine_dist,log=exp_setting['log'],max_dist=100.)
                    remain_points =  refine_dist[...,None] *(remain_points/np.linalg.norm(remain_points,axis=-1,keepdims=True) )
    
            Remain_points.append(remain_points)
            Remain_labels.append(remain_labels)
    return Remain_points,Remain_labels
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int,default=99)
    parser.add_argument('--expname', type=str,default='ray_drop')
    parser.add_argument('--depth_refine',action='store_true')
    parser.add_argument('--datadir',type=str,default='')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--pre_mask', action='store_true')
    parser.add_argument('--onehot_encoding', action='store_true')
    parser.add_argument('--ckpt', type=int, default=0)
    parser.add_argument('--start',type=int, default=99)
    parser.add_argument('--depth_filter',action='store_true')
    parser.add_argument('--simulation_path',type=str,default = '')
    parser.add_argument('--var',type = int, default= 1)
    parser.add_argument('--save_near',type=int, default=1)
    parser.add_argument('--nodrop',action='store_true')
    parser.add_argument('--random_drop',action='store_true')
    parser.add_argument('--mask_thre',type=float,default = 0.5)

    args = parser.parse_args()

    idx=args.idx
    expdir = os.path.join('exp',args.expname)
    with open(os.path.join(expdir,"args.txt"), 'r') as exp_file:
        exp_setting = {}
        for line in exp_file:
            k, v = line.strip().split('=')
            exp_setting[k.strip()] = v.strip()
    imgs=np.load(os.path.join(expdir,'points_features.npy'))
    gt_masks= np.load(os.path.join(expdir,'gt_masks.npy'))
    gt_ranges= np.load(os.path.join(expdir,'gt_ranges.npy'))
    import json
    with open(os.path.join(expdir,'model_args.json'),'r') as f:
        model_args = json.load(f)
    ckpt = [f for f in os.listdir(expdir) if f.endswith('.pth')]
    # if model_args['proj
    data_depends = [imgs,gt_masks,gt_ranges]
    model_args['proj_xyz'] = False
    runner= ray_drop_learning(data_depends,n_channels=imgs.shape[-1],**model_args)
    # runner.load_ckpt(os.path.join(expdir,ckpt[-1]))
    ckpt =sorted(ckpt)
    if args.ckpt:
        runner.load_ckpt(os.path.join(expdir,'{:05d}.pth'.format(args.ckpt)))
        print('load specific ckpt')
    else:
        runner.load_ckpt(os.path.join(expdir,ckpt[-1]))
        
    Points=[];Labels=[]
    samples = ['sample_labels']
    
    for i in range(len(samples)):
        savepath = 'sequences'
        idx = args.start+i
        if args.start<100:
            savepath = os.path.join(savepath,str(idx).zfill(2))
        else:
            savepath = os.path.join(savepath,str(idx).zfill(3))
        print(savepath)
        os.makedirs(savepath,exist_ok=True)
        os.makedirs(os.path.join(savepath,'velodyne'),exist_ok=True)
        os.makedirs(os.path.join(savepath,'labels'),exist_ok=True)

        lidar2globals = np.load(os.path.join(samples))
        root_simu_dir ='../6cam_scene23_revision';simulation_path=samples[i]
        temp_points,temp_labels =  drop_simulation(os.path.join(root_simu_dir,simulation_path),
        savepath,lidar2globals,runner,model_args,exp_setting,save_near=args.save_near,depth_refine=args.depth_refine,datadir=root_simu_dir,
        nodrop=args.nodrop,random_drop=args.random_drop,mask_thre= args.mask_thre)
        Points.append(temp_points)
        Labels.append(temp_labels)
        
    import multiprocessing
    print('Start write .bin and .label files')
    for i in range(len(samples)):
        test=multiprocessing.Pool(processes=32)
        savepath = 'sequences'
        idx = args.start+i
        if args.start<100:
            savepath = os.path.join(savepath,str(idx).zfill(2))
        else:
            savepath = os.path.join(savepath,str(idx).zfill(3))
        print(savepath)
        Remain_points =Points[i]
        Remain_labels = Labels[i]
        # import pdb;pdb.set_trace()
        for j in range(len(Remain_points)):
            test.apply_async(write_pointsandlabels,args=(j,savepath,Remain_points[j],Remain_labels[j]))
            # test.apply_async(write_labels,args=(j,savepath,))
        
        test.close()
        test.join()
