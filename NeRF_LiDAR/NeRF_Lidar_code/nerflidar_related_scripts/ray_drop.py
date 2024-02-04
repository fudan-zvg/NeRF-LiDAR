### 
# image grid
"""
1. rgb
2. intensity
3. semantic label

4. distance 
5. occupancy mask
"""
from socket import if_indextoname
from tkinter import Canvas
from unicodedata import name
import numpy as np
from PIL import Image
import cv2
import os
from points_filter import filter_func
from Snerf.src.dataloader.load_nuscenes import recenter_poses
from tqdm import tqdm
from ray_drop_learning import ray_drop_learning

def vis_points(points,img,i):
    Canvas = img*255.
    # Image.fromarray(
    #     (img * 255.).astype(np.uint8)).save(
    #         'test.png')
    Canvas[points[:,1],points[:,0]]=np.array([255.,0,0])
    cv2.imwrite('test_{:02d}.png'.format(i),Canvas)

def ray_drop_preprocessing(gt_points,rendered_points,rgbs,semantics,num_per_laser =2000,window_len=2048):
    # index =15 # the data we test
    interval = 30 # the number of images of each camera

    # N*6*3
    # N: number of points; 6 cams; 2 coords
    # canvas = []
    # for i in range(camera_index):
    #     canva = np.zeros((lidar_points.shape[0],1))
    #     rgb  = rgbs[i*interval+index]
    #     semantic = semantics[i*interval+index]
    #     c2w = extrinsics[i*interval+index]
    #     # print('cameara_index:{:01d}'.format(i),c2w[:3,3])
    #     K = intrinsics[i*interval+index]
    #     c2w_inv = np.linalg.inv(c2w)
    #     camera_coords = lidar_points @ c2w_inv[:3,:3].T + c2w_inv[:3,3].reshape(1,3)
    #     mask = camera_coords[:,-1]>1
    #     pixel_coords = camera_coords @ K.T 
        
       
    #     pixel_coords = pixel_coords[mask,:]
       
    #     pixel_coords = pixel_coords/pixel_coords[:,-1:]
    #     pixel_coords =  pixel_coords.round().astype(np.int16)[:,:2]
        
    #     x_mask=(pixel_coords[:,0]>=0)&(pixel_coords[:,0]<W)
    #     y_mask = (pixel_coords[:,1]>=0)&(pixel_coords[:,1]<H)
    #     pixel_coords =pixel_coords[x_mask & y_mask]
    #     vis_points(pixel_coords,img,i)
    #     canva = np.zeros((lidar_points.shape[0],2)).astype(np.int16)
    #     temp =canva[mask]
    #     temp[x_mask & y_mask] =pixel_coords
    #     canva[mask]=temp
    #     canvas.append(canva)
    # canvas = np.stack(canvas,axis=1)
    # vote for points in all cam
    ####################################################################the code below should be optimized!
    # vote = np.linalg.norm(canvas,axis=-1)
    # mask = vote.sum(axis=-1)!=0 # points withrgb value
    
    # rgb_value =np.zeros((lidar_points.shape[0],3)) # N * 3
    # for i in range(lidar_points.shape[0]):
    #     if vote[i,:].sum()!=0:
    #         nonzero = np.nonzero(vote[i])[0]
    #         img_index = nonzero[np.argmin(vote[i][nonzero])]
    #         rgb_value[i] = imgs[img_index][canvas[i][img_index][1],canvas[i][img_index][0],:]
    
    lidar_points = gt_points[:,:-1].reshape(4,-1,32)[:3,:,:].transpose()
    lidar_origin = gt_points[:3,-1]
    rendered_points = rendered_points.reshape(32,-1,3)
    # xycoords = (rendered_points - lidar_origin)[:,:,:2]
    angles = np.arctan2((rendered_points - lidar_origin)[:,:,:2][:,:,0],(rendered_points - lidar_origin)[:,:,:2][:,:,1])*180/np.pi
    angles_gt = np.arctan2((lidar_points - lidar_origin)[:,:,:2][:,:,0],(lidar_points - lidar_origin)[:,:,:2][:,:,1])*180/np.pi
    distance_mask  = np.zeros((32,window_len))
    occupancy_mask = np.zeros((32,window_len))
    rgb_mask = np.zeros((32,window_len,3))
    semantic_mask = np.zeros((32,window_len))
    interval = np.linspace(-180,180,window_len+1)
    count = np.zeros((32,window_len))
    gt_mask = np.zeros((32,window_len))
    for i in range(lidar_points.shape[0]):
        for j in range(lidar_points.shape[1]):
            angle_index = np.where(interval>angles_gt[i][j])[0].min()
            gt_mask[i,angle_index-1]+=1
    gt_mask = gt_mask!=0
    for i in range(32):
        for j in range(angles.shape[1]):
            angle_index = np.where(interval>angles[i][j])[0].min()
            distance_mask[i,angle_index-1] +=  np.linalg.norm(rendered_points[i][j]-lidar_origin)
            
            occupancy_mask[i,angle_index-1]+=1
            rgb_mask[i,angle_index-1]+=rgbs[i*num_per_laser+j]
            semantic_mask[i,angle_index-1]+=semantics[i*num_per_laser+j]
            count [i,angle_index-1] +=1
    occupancy_mask = occupancy_mask!=0
    
    rgb_mask[count!=0]/=count[count!=0][...,None]
    semantic_mask[count!=0]/=count[count!=0]
    distance_mask[count!=0]/=count[count!=0]
    return rgb_mask,semantic_mask,distance_mask,occupancy_mask,gt_mask
    

if __name__ == '__main__':
    # camera_parameters = np.load('/SSD_DISK/users/zhangjunge/scene7_camera_parameters/c2w.npy')
    # poses = np.load('/SSD_DISK/users/zhangjunge/6cam_scene7/poses_bounds.npy')[:,:-4].reshape(-1,3,5)
    camera_parameters = np.load('/SSD_DISK/users/zhangjunge/scene7_camera_parameters/c2w.npy')
    camera_parameters_inv = np.linalg.inv(camera_parameters)

    poses = np.load('/SSD_DISK/users/zhangjunge/6cam_scene7_revision/poses_bounds.npy')[:,:-4].reshape(-1,3,5)
    # index = 15
    frame_num =60
    window_len =1024
    lidar_data_pth = '/SSD_DISK/users/zhangjunge/gt_lidar/lidar_points_scene7_80'
    lidar_points=[]
    for i in range(len(os.listdir(lidar_data_pth))):
        temp_points =np.load(os.path.join(lidar_data_pth,'points{:03d}.npy'.format(i)))
        lidar_points.append(temp_points)
    # lidar_points=np.stack(lidar_points, axis=0)
    # ###################### ######## ######## ######## ##### camera parameters ######## #####################
    # intrinsics
    raw_cam_K=poses[:,:,4].copy().astype(np.float32).transpose([1,0]); 
    cx=raw_cam_K[0,:]
    cy=raw_cam_K[1,:]
    focal=raw_cam_K[2,:]
    K=[np.array([[focal[i],0,cx[i]],[0,focal[i],cy[i]],[0,0,1]]) for i in range(poses.shape[0])]
    K = np.stack(K, 0)
    # extrinsics
    # poses = np.concatenate([poses[:,:, 1:2],poses[:, :,0:1], -poses[:,:, 2:3], poses[:,:, 3:4], poses[:,:, 4:5]], 2)
    poses = np.concatenate(
    [poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], 2)
    poses, c2w = recenter_poses(poses)
    c2w_inv = np.linalg.inv(c2w)

    # c2w = np.stack([np.concatenate([c2w[i],np.array([0,0,0,1]).reshape(1,4)],axis=0) for i in range(poses.shape[0])],axis=0)
    ################################################################ NeRF based Rendering
    render_datapath = '/SSD_DISK/users/zhangjunge/scene7_revision_lidar_render'


    degree =2; 
    print('begin preprocessing')
    rgb_masks,semantic_masks,distance_masks,occupancy_masks,gt_masks=[],[],[],[],[]
    for index in tqdm(range(0,frame_num)):
        rgbs=np.load(os.path.join(render_datapath,'points_rgb_{:03d}.npy'.format(index)))#(32*N,3)
        num_per_laser = rgbs.shape[0]//32
        semantics = np.load(os.path.join(render_datapath,'points_semantic_{:03d}.npy'.format(index)))#(32*N,)
        rendered_points = np.load(os.path.join(render_datapath,'points_{:03d}.npy'.format(index))).reshape(32,-1,3) # (32,N,3)

        # get lidar_origin
        lidar_origin = lidar_points[index][:3,-1]
        lidar_origin =lidar_origin @ camera_parameters_inv[:3,:3].T + camera_parameters_inv[:3,3]
        lidar_origin = lidar_origin @ c2w[:3,:3] + c2w_inv[:3,3]
        filtered_points = filter_func(rendered_points,lidar_origin,degree)
        rgb_mask,semantic_mask,distance_mask,occupancy_mask,gt_mask = ray_drop_preprocessing(lidar_points[index],filtered_points,rgbs,semantics,num_per_laser,window_len=window_len)
        rgb_masks.append(rgb_mask);semantic_masks.append(semantic_mask);distance_masks.append(distance_mask)
        occupancy_masks.append(occupancy_mask);gt_masks.append(gt_mask)
    rgb_masks = np.stack(rgb_masks);semantic_masks= np.stack(semantic_masks);distance_masks= np.stack(distance_masks);
    occupancy_masks= np.stack(occupancy_masks);gt_masks= np.stack(gt_masks)
    import pdb;pdb.set_trace()
    for mask in [rgb_masks,semantic_masks,distance_masks,occupancy_masks,gt_masks]:
        mask = np.stack(mask,axis=0)
        np.save('{}.npy'.format(str(mask)),mask)
    ray_drop_learning(rgb_masks,semantic_masks,distance_masks,occupancy_masks,gt_masks)