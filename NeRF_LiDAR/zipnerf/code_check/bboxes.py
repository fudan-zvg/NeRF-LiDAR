import os
import numpy as np
import json
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from PIL import Image
import io
import time
import imageio
import torch
def find_interval(timestamps, t):
    # 查找给定时间t之前的最后一个时间戳的索引
    idx = np.searchsorted(timestamps, t, side='right') - 1

    # 检查索引是否在有效范围内
    if idx < 0:
        return 0 ,1  # t 位于第一个时间戳之前
    elif idx >= len(timestamps) - 1:
        return -2,-1  # t 位于最后一个时间戳之后
    else:
        return (idx, idx + 1)
def load_time(root_dir):
    times = np.loadtxt(os.path.join(root_dir,'timestamps.txt'))
    time_min, time_max = times.min() , times.max()
    # 1e6 is 1s
    time_unit = 1e6
    time_scale = (time_min,time_unit)
    times = (times - time_min) / time_unit
    return times,time_scale
def vis_pose(pose,ego_idx=0,out_name = 'output.mp4'):
    # pose N_obj,N_time,9
    
    ego = pose[ego_idx,...]
    obj_idx = [i for i in range(len(pose))];obj_idx.remove(ego_idx)
    obj_pose = pose[obj_idx,...]
    t0=time.time()
    vis_imgs = []
    arrived_points = []
    for time_t in range(obj_pose.shape[1]):
        fig, ax = plt.subplots()
        curr_pos = obj_pose[:,time_t,:3]
        valid = np.linalg.norm(obj_pose[:,time_t,7:10],axis=-1,) > 1
        if arrived_points!=[]:
            plt.scatter(np.concatenate(arrived_points,axis=0)[:,0],np.concatenate(arrived_points,axis=0)[:,1],marker='s',c = 'b')
        point = curr_pos[valid][:,:2] #(x, y)
        arrived_points.append(point)
        plt.scatter(point[:,0],point[:,1],marker='s',c = 'y')
        plt.scatter(ego[time_t,0],ego[time_t,1],marker='s',c = 'r')
        plt.xlim(-100, 100)
        plt.ylim(-20, 20)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        # img_arr = np.array(Image.open(buf))
        vis_imgs.append(Image.open(buf))
        plt.close()

    print(f'time consumes {time.time()-t0}')
    # gif
    # vis_imgs[0].save('output.gif', save_all=True, append_images=vis_imgs[1:], loop=0, duration=200)

    # mp4
    images = [np.array(img).astype(np.uint8) for img in vis_imgs]
    # writer = imageio.get_writer('output.mp4', fps=10)
    writer = imageio.get_writer(out_name,format='FFMPEG', fps=10)
    for img in images:
        writer.append_data(img)
    writer.close()
    return
def pose_interpolation(timestamps,track):
    # timestamps : N
    # track : n * 9
    recored_time = track[:,-2]
    recored_center = track[:,:3]
    orientation = track[:,3:7]
    wlh = track[0,7:10]
    track_id = track[0,-1]
    interpolated_pose = []
    for index,t in enumerate(timestamps):
        if t < recored_time.min():
            orient = Quaternion(track[0,3:7])
            theta_z = np.array([orient.yaw_pitch_roll[0]])
            pose_array = np.concatenate([track[0,:3],theta_z,track[0,7:]])
            # invalid bbox
            pose_array[7:10]*=1e-5
        elif t> recored_time.max():
            orient = Quaternion(track[-1,3:7])
            theta_z = np.array([orient.yaw_pitch_roll[0]])
            pose_array = np.concatenate([track[-1,:3],theta_z,track[-1,7:]])
            # invalid bbox
            pose_array[7:10]*=1e-5
        else:
            # apply interpolation
            idx0, idx1 = find_interval(recored_time,t)
            t0 ,t1 = recored_time[idx0], recored_time[idx1]
            c0 ,c1 = recored_center[idx0], recored_center[idx1]
            q0, q1 = Quaternion(orientation[idx0]), Quaternion(orientation[idx1])
    
            center = np.array([np.interp(t, [t0, t1], [c0_, c1_] ) for c0_,c1_ in zip(c0, c1)])

            # Interpolate orientation.
            rotation = Quaternion.slerp(q0, q1,amount=(t - t0) / (t1 - t0))
            theta_z = np.array([rotation.yaw_pitch_roll[0]])

            pose_array = np.concatenate([center.flatten(),theta_z,wlh.flatten(),np.array([t]),track_id.reshape(-1)])
        interpolated_pose.append(pose_array) # 9,

    
    return np.stack(interpolated_pose)

if __name__ == '__main__':
    root_dir = '/data1/junge/nuscenes/0237_front'
    # time_min = 0.
    timestamps,time_scale = load_time(root_dir)
    time_min,time_unit = time_scale
    time_unit = 1e6

    with open(os.path.join(root_dir,'bboxes.json'),'r') as f:
        bboxes = json.load(f)

    track_id = 0
    camera_parameters = np.load(os.path.join(root_dir,'c2w.npy'))
    camera_parameters_inv = np.linalg.inv(camera_parameters)
    # scale_factor = np.load(os.path.join(root_dir,'scene_scale.npy'))
    scale_factor = 1
    transform = np.load(os.path.join(root_dir,'c2w_recenter_transform.npy'))
    c2w = np.linalg.inv(transform)
    c2w_inv = transform
    tracks_np = []
    for instance,annotations in bboxes.items():
        obj_pose = []

        if instance == 'ego':
            class_type = 'ego_car'
        else:
            class_type = annotations[0][11]
        if 'human' in class_type:
            continue
        for idx,ann in enumerate(annotations):
            class_type = ann[11] # human, vehicle, etc.
            
            # ann in global annotations 
            center = (np.array(ann[:3])@ camera_parameters_inv[:3,:3].T + camera_parameters_inv[:3,3]) 
            center = center @ c2w[:3,:3] + c2w_inv[:3,3]
            center *= scale_factor
            # l, w, h for x, y, z
            wlh = np.array(ann[3:6]) * scale_factor *1.2 # for shading
            wlh[1] , wlh [0] =wlh[0] , wlh [1] # make l ,w, h aligned with x y z
            orient = np.array(ann[6:10])
            q , r = np.linalg.qr(c2w_inv[:3,:3] @ camera_parameters_inv[:3,:3] ) # apply transform inverse as c2w
            orth_transform = q @ r.round()
            orth_trans = Quaternion(matrix=orth_transform)
            orient = orth_trans * orient
            orient = np.array([orient[0],orient[1],orient[2],orient[3]])
            theta_y = np.array([Quaternion(orient).yaw_pitch_roll[0]])
            timestamp = np.array([ann[10]])
            timestamp = (timestamp - time_min)/time_unit

            obj_pose.append(np.concatenate([center.flatten(),orient.flatten(),wlh.flatten(),timestamp,np.array([track_id])],axis=-1)) # [12,]
        obj_pose = np.stack(obj_pose)
        obj_pose_indice = np.argsort(obj_pose[:,-2]) # sort with time
        obj_pose = obj_pose[obj_pose_indice]
            
        interpolated_pose = pose_interpolation(timestamps,obj_pose)
        tracks_np.append(interpolated_pose)
        print(f'done {track_id}th instance')
        if instance == 'ego':
            ego_track = interpolated_pose
            print(f'done ego {track_id}th')
        track_id +=1

    tracks_np = np.stack(tracks_np)

    vis_pose(tracks_np)


    ckpt_latent = torch.load('tracknet_ckpt_40000.ckpt')
    delta_dist =  ckpt_latent['state_dict']['opt_t'].cpu().numpy()
    scale_factor = np.load(os.path.join(root_dir,'scene_scale.npy'))
    delta_dist = delta_dist / scale_factor

    tracks_np_delta = tracks_np
    tracks_np_delta[1:,:,:3] += delta_dist
    
    vis_pose(tracks_np_delta,out_name= 'output_delta.mp4')