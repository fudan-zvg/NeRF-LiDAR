from tkinter.tix import Tree
import numpy as np
import os
from nerf2world import nerf_to_world,nerf_to_lidar
from Generate_feature import pcs2img
from model.ray_drop_train import ray_drop_learning
from Generate_feature import generate_simulation_data
from lidar_utils import *
from tqdm import tqdm
import argparse
from depth_filter import depth_filter
from scipy.special import softmax
import glob
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
                nodrop=False,random_drop = False,mask_thre = 0.5,post_process = False):
    
    if 'points_rgb_{:04d}.npy'.format(0) in os.listdir(simulation_path): # i.e. GOT rgb feature of points
        lidarrender_num = len([f for f in os.listdir(simulation_path) if f.startswith('points') and f.endswith('npy')])//3
    else:
        lidarrender_num = len([f for f in os.listdir(simulation_path) if f.startswith('points') and f.endswith('npy')])//2

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

        Points_features, depends = generate_simulation_data(lidar2globals,lidarrender_num,simulation_path,args,W=1024,log=True,\
                                                          datadir=datadir,simulation = True,return_depends =True,pre_filter = False,\
                                                            verbose=args.verbose)
        print('Ray Drop: datadir{} simulation_path{}'.format(datadir,simulation_path))
        Points, Semantics, Scans = depends
        # import pdb;pdb.set_trace()
        for i in tqdm(range(Points_features.shape[0])):
                # direct load points
                points_features = Points_features[i]
                points_ = Points[i]
                laser_scan = Scans[i]

                points_semantic = Semantics[i]
                # nerf_points = np.load(os.path.join(simulation_path,'points_{:04d}.npy'.format(i)))
                # points_semantic = np.load(os.path.join(simulation_path,'points_semantic_{:04d}.npy'.format(i)))
                # points_ = nerf_to_lidar(nerf_points,lidar2globals[i],datadir)
                depth_filter_mask = None
                if args.depth_filter == 1:
                    if points_.shape == laser_scan.points.shape: # nofilter before
                        # depth_filter_mask = depth_filter(points_,return_mask = True)[...,0]
                        if args.semantic_align:
                            depth_filter_mask = depth_filter(points_,points_semantic,return_mask = True,width = 1,threshold=args.filter_thre)
                        else:
                            depth_filter_mask = depth_filter(points_,return_mask = True,width= 5)
                        if args.dist_thre:
                            depth_filter_mask = depth_filter_mask | (np.linalg.norm(points_,axis = -1) < args.dist_thre)
                    else:
                        points_ = laser_scan.points # filter before
                        points_semantic = laser_scan.semantic
                        depth_filter_mask = np.ones_like(points_semantic) # remain all points
                    # import pdb;pdb.set_trace()
                    # points_features = points_features[depth_filter_mask.reshape(32,-1)]
                    # points_= points_[depth_filter_mask]
                    # points_semantic = points_semantic[depth_filter_mask]
                pred_mask,pred_range = runner.test(points_features[None,...])
                # laser_scan = LaserScan(H=32,fov_up = 10.67, fov_down= -30.67)
                # laser_scan.set_points(points_,remissions=None,semantic=points_semantic,rgb=None)
                # laser_scan.do_range_projection()

                # direct load laser scan
                pred_mask = softmax(pred_mask,axis = 0)[1] # probability
                
                if not mask_thre:
                    pred_mask = np.argmax(pred_mask, axis=0)
                    import pdb;pdb.set_trace()
                else:
                    if args.place_car:
                        # import pdb;pdb.set_trace()
                        car_mask = laser_scan.proj_semantic ==13 # get car mask
                        if car_mask.sum() > 0:
                            car_thre = np.percentile(pred_mask[car_mask],50)
                            pred_mask[car_mask] = pred_mask[car_mask] > car_thre # deal car first
                        pred_mask = (pred_mask>args.mask_thre)
                    else:
                        pred_mask = pred_mask>args.mask_thre
                mask = (pred_mask ==1)  & (laser_scan.proj_mask ==1)
                
                if not save_near:
                    # mask = (pred_mask ==1)  & (laser_scan.proj_mask ==1)
                    mask = (mask[laser_scan.proj_y,laser_scan.proj_x]==1) & (depth_filter_mask==1)
                    remain_points = points_[mask]
                    remain_labels = points_semantic[mask]
                else:
                    # import pdb;pdb.set_trace()
                    # proj_mask = laser_scan.proj_mask
                    # mask = (pred_mask==1)&(proj_mask==1)
                    if depth_filter_mask is None:
                        remain_points = laser_scan.proj_xyz[mask]
                        remain_labels = laser_scan.proj_semantic[mask]
                    else:
                        # import pdb;pdb.set_trace()
                        idx = laser_scan.proj_idx.reshape(-1)
                        # get depth_filter idx 
                        filter_mask = depth_filter_mask[idx].reshape(32,-1)
                        mask = mask & (filter_mask==1)
                        remain_points = laser_scan.proj_xyz[mask]
                        remain_labels = laser_scan.proj_semantic[mask]

                    # if model_args['regression'] and depth_refine:
                    #     pred_range = pred_range[0][0]
                    #     if model_args['delta']:
                    #         if int(exp_setting['log']):
                    #             real = np.where(laser_scan.proj_range<0, 0, laser_scan.proj_range) + 0.0001
                    #             #Apply log
                    #             real = ((np.log2(real+1)) / 6.5)
                    #             #Make negatives 0
                    #             real = np.clip(real, 0, 1)
                                
                    #         else:
                    #             real=laser_scan.proj_range/max_dist
                    #             real = np.clip(real, 0, 1)
                    #         pred_range =np.clip(2*pred_range-1+ real,0,1)
                    #     # remain_points = laser_scan.proj_xyz[(pred_mask==1)&(laser_scan.proj_mask==1)]
                    #     refine_dist = pred_range[(pred_mask==1)&(laser_scan.proj_mask==1)]
                    #     # from [0,1] to real distance
                    #     def dist_backtoreal(refine_dist,log=True,max_dist=max_dist):
                    #         if log:
                    #             refine_dist = 2**(refine_dist*6.5)-1
                    #         else:
                    #             refine_dist = max_dist *refine_dist
                    #         return refine_dist
                    #     refine_dist = dist_backtoreal(refine_dist,log=exp_setting['log'],max_dist=100.)
                    #     remain_points =  refine_dist[...,None] *(remain_points/np.linalg.norm(remain_points,axis=-1,keepdims=True) )
                ## drop sky
                sky_mask = remain_labels == 10
                remain_points = remain_points[~sky_mask]
                remain_labels = remain_labels[~sky_mask]
                ## drop road_outlier
                road_outlier = (remain_labels == 0) & (remain_points[:,2]<-3)
                remain_points = remain_points[~road_outlier]
                remain_labels = remain_labels[~road_outlier]
                Remain_points.append(remain_points)
                Remain_labels.append(remain_labels)

    
    
    return Remain_points,Remain_labels
def get_lidar2global(simulation_path,datadir):
    if os.path.exists(os.path.join(simulation_path, 'lidar2globals.npy')):
        lidar2globals = np.load(os.path.join(simulation_path, 'lidar2globals.npy'))
    else:
        assert  os.path.exists(os.path.join(simulation_path, 'ego_trace.npy')) or os.path.exists(os.path.join(simulation_path, 'lidar_origins.npy'))#for simulation data
        try:
            ego_trace = os.path.join(simulation_path, 'ego_trace.npy')
            ego_trace = np.load(ego_trace)
        except:
            ego_trace = os.path.join(simulation_path, 'lidar_origins.npy')
            ego_trace = np.load(ego_trace)
            ego_trace = ego_trace[:,0,:]
        num = len(glob.glob(simulation_path+'/points*.npy'))//3
        cam2global = np.load(os.path.join(datadir,'c2w.npy')) #cam2global
        # lidar2cam = np.linalg.inv(cam2ego) @ lidar2ego
        lidar2cam =np.load(os.path.join(datadir,'lidar2cam.npy')).astype(np.float32)
        lidar2global = cam2global @ lidar2cam 
        global_origins = nerf_to_world(ego_trace,datadir)
        if len(global_origins) > num:
            global_origins = global_origins[:-1,:]

        lidar2globals = np.broadcast_to(lidar2global[None,...],(global_origins.shape[0],4,4))

        global_origins = np.concatenate([global_origins,np.ones((global_origins.shape[0],1))],axis=-1)[:,:,None]
        lidar2globals = np.concatenate([lidar2globals[:,:,:3],global_origins],axis=-1)

        
    
    return lidar2globals

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int,default=99)
    parser.add_argument('--expname', type=str,default='')
    parser.add_argument('--depth_refine',action='store_true')
    parser.add_argument('--datadir',type=str,default='')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--pre_mask', action='store_true')
    parser.add_argument('--onehot_encoding', action='store_true')
    parser.add_argument('--ckpt', type=int, default=0)
    parser.add_argument('--start',type=int, default=99)
    parser.add_argument('--depth_filter',action='store_true')
    parser.add_argument('--filter_thre',type=int,default= 0 )
    parser.add_argument('--simulation_path',type=str,default = '')
    parser.add_argument('--var',type = int, default= 1)
    parser.add_argument('--save_near',type=int, default=1)
    parser.add_argument('--nodrop',action='store_true')
    parser.add_argument('--random_drop',action='store_true')
    parser.add_argument('--mask_thre',type=float,default = 0.5)
    parser.add_argument('--dvgo', action= 'store_true')
    parser.add_argument('--place_car',action='store_true')
    parser.add_argument('--dist_thre',type = int, default= 0,help='the distance threshold for filter')
    parser.add_argument('--semantic_align',action='store_true')
    parser.add_argument('--verbose',action= 'store_true')
    parser.add_argument('--save_sensor_data',type=int,default=1)
    parser.add_argument('--savepath',type=str)
    ### Load setting
    args = parser.parse_args()

    idx=args.idx
    expdir = os.path.join('./exp',args.expname)
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

    data_depends = [imgs,gt_masks,gt_ranges]
    model_args['proj_xyz'] = False
    ### Load model

    runner= ray_drop_learning(data_depends,n_channels=imgs.shape[-1],**model_args)
    ckpt =sorted(ckpt)
    if args.ckpt:
        runner.load_ckpt(os.path.join(expdir,'{:05d}.pth'.format(args.ckpt)))
        print('load specific ckpt')
    else:
        runner.load_ckpt(os.path.join(expdir,ckpt[-1]))
    if '/HDD' in args.simulation_path:
        root_simu_dir = '{}'.format(args.simulation_path)
    else:
    	root_simu_dir = '../{}'.format(args.simulation_path)
    simulation_paths = os.listdir(root_simu_dir)
    simulation_paths = sorted(simulation_paths)
    Points=[];Labels=[]
    args.depth_filter = exp_setting['depth_filter']=='True' or args.depth_filter
    # assert args.depth_filter == (args.dist_thre > 0), 'Wrong match'
    if args.depth_filter:
        print('apply filter')
    if args.depth_refine:
        print("Apply Depth Refinement")
    for i in range(len(simulation_paths)):
        savepath = args.savepath
        idx = args.start+i
        if args.start<100:
            savepath = os.path.join(savepath,str(idx).zfill(2))
        else:
            savepath = os.path.join(savepath,str(idx).zfill(3))
        print(savepath)
        os.makedirs(savepath,exist_ok=True)
        os.makedirs(os.path.join(savepath,'velodyne'),exist_ok=True)
        os.makedirs(os.path.join(savepath,'labels'),exist_ok=True)
        simulation_path = simulation_paths[i]
        # generate lidar2global for all simulated data
        if args.dvgo:
            datadir = os.path.join('../nuScenes_scenes/{}'.format(simulation_paths[i]))
        else:
            datadir = os.path.join('../Scenes/6cam_{}_revision'.format(simulation_paths[i].split('_')[0]))
        # datadir = args.datadir
        lidar2globals = get_lidar2global(os.path.join(root_simu_dir,simulation_path),datadir)
        if args.save_sensor_data:
            np.save(os.path.join(savepath,'lidar2globals.npy'),lidar2globals)
            lidar2ego = np.load(os.path.join(datadir,'lidar2ego.npy'))
            
            # ego2global = lidar2global @ ego2lidar
            ego2globals = [lidar2globals[i] @ np.linalg.inv(lidar2ego) for i in range(len(lidar2globals))]
            ego2globals = np.stack(ego2globals)

            lidar2egos = np.expand_dims(lidar2ego,0).repeat(len(lidar2globals),axis = 0)
            np.save(os.path.join(savepath,'lidar2egos.npy'),lidar2egos)
            np.save(os.path.join(savepath,'ego2globals.npy'),ego2globals)
            try:
                os.system('cp -r {} {}'.format(os.path.join(datadir,'first_sample_token.txt'),os.path.join(savepath,'sample_token.txt')))
            except:
                print('No sample token.txt')
                pass

        temp_points,temp_labels =  drop_simulation(os.path.join(root_simu_dir,simulation_path),
        savepath,lidar2globals,runner,model_args,exp_setting,save_near=args.save_near,depth_refine=args.depth_refine,datadir=datadir,
        nodrop=args.nodrop,random_drop=args.random_drop,mask_thre= args.mask_thre)
        Points.append(temp_points)
        Labels.append(temp_labels)
        
    import multiprocessing
    print('Start write .bin and .label files')
    for i in range(len(simulation_paths)):
        test=multiprocessing.Pool(processes=8)
        savepath = args.savapath
        idx = args.start+i
        # savepath = simulation_paths[i]
        if args.start<100:
            savepath_ = os.path.join(savepath,str(idx).zfill(2))
        elif args.start<1000:
            savepath_ = os.path.join(savepath,str(idx).zfill(3))
        else:
            savepath_ = os.path.join(savepath,str(idx).zfill(4))
        print(savepath_)
        Remain_points =Points[i]
        Remain_labels = Labels[i]
        for j in range(len(Remain_points)):
            test.apply_async(write_pointsandlabels,args=(j,savepath_,Remain_points[j],Remain_labels[j]))
        
        test.close()
        test.join()
