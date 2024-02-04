import os
import numpy as np
import torch
import glob
from tqdm import tqdm
from scipy.spatial import Delaunay
import scipy
def cast_lidar_ray_batch(lidar_origins,
                        lidar_directions,pixels):
    origins = lidar_origins
    directions = lidar_directions
    viewdirs = lidar_directions / np.linalg.norm(lidar_directions)
    # radii = np.zeros(lidar_origins.shape[0]).reshape(-1,1)
    radii = np.ones(lidar_origins.shape[0]).reshape(-1,1) * 0.0005
    imageplane = np.zeros_like(lidar_origins)[:,:2] # not used
    # keep same as lidar directions
    base_x = lidar_directions
    base_y = lidar_directions
    return dict(
        origins=origins,
        directions=directions,
        viewdirs=viewdirs,
        radii=radii,
        imageplane=imageplane,
        lossmult=pixels.get('lossmult'),
        near=pixels.get('near'),
        far=pixels.get('far'),
        cam_idx=pixels.get('cam_idx'),
        exposure_idx=pixels.get('exposure_idx'),
        exposure_values=pixels.get('exposure_values'),
        base_x=base_x, ## 2 img plane bases in global world cd
        base_y=base_y
    )
def get_gt_info(datadir,recenter_param = None,testsavedir='',frames_num = 80):
    rays_num= 32
    lidar_angles = [-30.67,-9.33, -29.33, -8.00, -28.00, -6.67,-26.67, -5.33, -25.33, -4.00, -24.00,-2.67, -22.67, -1.33,-21.33,0.00,
    -20.00,1.33,-18.67,2.67,-17.33,4.00,-16.00,5.33,-14.67,6.67,-13.33,8.00,-12.00,9.33,-10.67,10.67]
    lidar_angles = sorted(lidar_angles)
    
    ################################# Parameters setting and Loading data
    points_per_angle = 1100
    start_frame = 0 
    
    # args,render_fn,train_params = load_data(argss)
    # c2w: the transformation matrix, c2w_recenter: the recenter transformation matrix
    camera_parameters = np.load(os.path.join(datadir,'c2w.npy'))
    camera_parameters_inv = np.linalg.inv(camera_parameters)
    lidar_points_path = os.path.join(datadir,'lidar_points')
    if recenter_param is None:
        c2w = np.load(os.path.join(datadir,'c2w_recenter_transform.npy'))
        c2w_inv = np.linalg.inv(c2w)
        scale_factor = np.load(os.path.join(datadir,'scene_scale.npy'))
    else:
        transform, scale_factor = recenter_param
        R = transform[:3,:3]
        assert ((R @ R.T - np.eye(3))**2 ).sum() < 1e-3
        c2w = np.linalg.inv(transform)
        c2w_inv = transform
    ################################################################ directions
    horizontal_angles = np.linspace(270,-90,points_per_angle)/180*np.pi

    directions = [];lidar_origins=[]
  
    ################################################################ Lidar data
    lidar2global = np.load(os.path.join(datadir,'lidar_points','lidar2global.npy')).astype(np.float32)
    for frame_idx in tqdm(range(start_frame,frames_num)):
        points_center = np.load(os.path.join(lidar_points_path,'points{:03d}.npy'.format(frame_idx)))[:,-1]
        if os.path.exists(os.path.join(lidar_points_path,'points{:03d}.npy'.format(frame_idx+1))):
            points_center_next = np.load(os.path.join(lidar_points_path,'points{:03d}.npy'.format(frame_idx+1)))[:,-1]
        else:
            points_center_next = np.load(os.path.join(lidar_points_path,'points{:03d}.npy'.format(frame_idx)))[:,-1]
        points_1 = points_center.copy();points_1[2] = 0.
        
        points_ = points_center[:3] @ camera_parameters_inv[:3,:3].T + camera_parameters_inv[:3,3]
        points_1 = points_1[:3]@ camera_parameters_inv[:3,:3].T + camera_parameters_inv[:3,3]
        points_center_next = points_center_next[:3]@ camera_parameters_inv[:3,:3].T + camera_parameters_inv[:3,3]
    
        # origin
        time_interval = np.linspace(0,0.05,points_per_angle) # frequency 20Hz
        lidar_origin = points_ @ c2w[:3,:3] + c2w_inv[:3,3]
        lidar_origin_next = points_center_next @ c2w[:3,:3] + c2w_inv[:3,3]
       
        # the frequency of the lidar is 20 Hz
        lidar_origin_all = -(time_interval.reshape(-1,1))@((lidar_origin_next-lidar_origin))[None,:]/(0.5/10) +lidar_origin[None,:]
        lidar_origin_all =np.concatenate([lidar_origin_all for i in range(len(lidar_angles))],axis=0)
        lidar_origin_all = lidar_origin_all * scale_factor
        # lidar_origin_1 = points_1 @ c2w[:3,:3] + c2w_inv[:3,3]

        ########## save all the lidar points
        # lidar_points = lidar_data[frame_idx]
        
        directions_ = get_directions(lidar_angles,horizontal_angles) # right down backwards
        directions_ = directions_ @ lidar2global[frame_idx][:3,:3].T @ camera_parameters_inv[:3,:3].T

        directions_ = directions_ @ c2w[:3,:3] # tranform for the avg poses
        directions.append(directions_)

        lidar_origins.append(lidar_origin_all)
    os.makedirs(testsavedir,exist_ok=True)
    np.save(os.path.join(testsavedir,'lidar_origins.npy'),np.stack(lidar_origins)/scale_factor)
    return (lidar_origins , directions)

def get_simu_info(datadir,recenter_param = None,complicated = False,testsavedir = '',render_nums = 100,speed = False):
    os.makedirs(testsavedir,exist_ok=True)
    # if testsavedir == '':
    #     testsavedir = datadir
    if recenter_param is None:
        c2w = np.load(os.path.join(datadir,'c2w_recenter_transform.npy'))
        c2w_inv = np.linalg.inv(c2w)
        scale_factor = np.load(os.path.join(datadir,'scene_scale.npy'))
    else:
        transform, scale_factor = recenter_param
        R = transform[:3,:3]
        assert ((R @ R.T - np.eye(3))**2 ).sum() < 1e-3
        c2w = np.linalg.inv(transform)
        c2w_inv = transform

    
    camera_parameters = np.load(os.path.join(datadir,'c2w.npy'))
    camera_parameters_inv = np.linalg.inv(camera_parameters)

    lidar_angles = [-30.67,-9.33, -29.33, -8.00, -28.00, -6.67,-26.67, -5.33, -25.33, -4.00, -24.00,-2.67, -22.67, -1.33,-21.33,0.00,
    -20.00,1.33,-18.67,2.67,-17.33,4.00,-16.00,5.33,-14.67,6.67,-13.33,8.00,-12.00,9.33,-10.67,10.67]
    lidar_angles = sorted(lidar_angles)
    ################################# Parameters setting and Loading data
    points_per_angle = 1100
    start_idx = 0 
    end_idx = 80
    # c2w: the transformation matrix, c2w_recenter: the recenter transformation matrix
    
    lidar_points_path = os.path.join(datadir,'lidar_points')
    ################################################################ directions
    horizontal_angles = np.linspace(270,-90,points_per_angle)/180*np.pi
    directions = get_directions(lidar_angles,horizontal_angles)
    lidar2cam = np.load(os.path.join(datadir,'lidar2cam.npy')).astype(np.float32)

    # right down backwards
    directions = directions @ lidar2cam[:3,:3].T# tranfer to the front camera: right backwards down 
    P = np.eye(3);P[1,1] =-1;P[2,2] = -1
    # please keep right downwards forwards before applying the c2w avg poses.
    # directions = directions @ P # transfer the requeseted coordinate system i.e. right upwards backwards
    directions = directions @ c2w[:3,:3] # tranform for the avg poses

    points_center = np.load(os.path.join(lidar_points_path,'points{:03d}.npy'.format(start_idx)))[:,-1]
    points_center_final = np.load(os.path.join(lidar_points_path,'points{:03d}.npy'.format(end_idx)))[:,-1]

    points_ = points_center[:3] @ camera_parameters_inv[:3,:3].T + camera_parameters_inv[:3,3]
    points_center_final = points_center_final[:3]@ camera_parameters_inv[:3,:3].T + camera_parameters_inv[:3,3]
    lidar_origin = points_ @ c2w[:3,:3] + c2w_inv[:3,3]
    lidar_origin_final = points_center_final @ c2w[:3,:3] + c2w_inv[:3,3]

    lidar_origin *= scale_factor
    lidar_origin_final *= scale_factor
    # the frequency of the lidar is 20 Hz

    time_interval = np.linspace(0,0.05,points_per_angle) # frequency 20Hz
    lidar_origin_interval = np.linspace(0,1,render_nums+1).reshape(-1,1) *(lidar_origin_final - lidar_origin)
    
    if complicated:
        # y
        lidar_origin_interval[:,1] = lidar_origin_interval[:,1] + 0.1 * np.random.randn(len(lidar_origin_interval))
        # x z
        lidar_origin_interval[:,[0,2]] = lidar_origin_interval[:,[0,2]] + 2 * (np.random.rand(len(lidar_origin_interval),2)*2-1)

        # speed
        speed = True
        speed_ = np.random.rand(render_nums)*6
        # 0m/s - 10m/s
    # this is just a straight line for the ego vehicle
    np.save(os.path.join(testsavedir,'ego_trace.npy'),(lidar_origin_interval+lidar_origin)/scale_factor)
    if not speed:
        origins = (lidar_origin_interval+lidar_origin)[:-1]
    elif complicated:
        origins = (lidar_origin_interval+lidar_origin)[:-1].reshape(-1,1,3) + (time_interval.reshape(1,-1) * speed_.reshape(-1,1))[...,None]
        origins = np.expand_dims(origins,1).repeat(len(lidar_angles),axis = 1).reshape(render_nums,-1,3)

    # P = np.eye(4);P[1,1]=-1;P[2,2]=-1;
    # def cal_l2ws():
    #     lidar2global = []
    #     for idx in range(len(origins)):
    #         tmp = np.eye(4)
    #         tmp[:3,3] = origins[idx] / scale_factor # in world coords
    #         c2w_ = camera_parameters @ c2w @ tmp
    #         l2ws = c2w_ @ lidar2cam
    #         lidar2global.append(l2ws)
    #     lidar2global = np.stack(lidar2global)
    #     return lidar2global
    # lidar2global = cal_l2ws()
    # np.save(os.path.join(testsavedir,'lidar2globals.npy'),lidar2global)
    return (origins, directions)


def load_lidar(datadir,moving_mask = True, return_full = False,recenter_param = None):
    if recenter_param is None:
        c2w = np.load(os.path.join(datadir,'c2w_recenter_transform.npy'))
        c2w_inv = np.linalg.inv(c2w)
        scale_factor = 1.
    else:
        transform, scale_factor = recenter_param
        # transform is [R T; 0 1]
        R = transform[:3,:3]
        assert ((R @ R.T - np.eye(3))**2 ).sum() < 1e-3
        c2w = np.linalg.inv(transform)
        c2w_inv = transform
    camera_parameters = np.load(os.path.join(datadir,'c2w.npy'))
    camera_parameters_inv = np.linalg.inv(camera_parameters)

    lidar_path = os.path.join(datadir,'lidar_points')
    lidar_files = glob.glob(os.path.join(datadir,'lidar_points','*.bin'))
    i_train_lidar = np.array([i for i in range(len(lidar_files))])

    lidar_distances=[];lidar_origins=[];lidar_rays_directions = []
    lidar2globals = np.load(os.path.join(lidar_path,'lidar2global.npy'))
    Points = []
    intensitys = []
    # Aggregated_Points = []
    # aggregated = True if 'aggregated' in args.keys() else False
    aggregated = False
    for frame_idx in i_train_lidar:
        bbox = None
        if moving_mask:
            f = open(os.path.join(datadir,'lidar_mask','{:04d}.txt'.format(frame_idx)),'r')
            bbox = f.readlines() 
            bboxes = np.array([box.split()[1:] for box in bbox]).astype(np.float32).reshape(-1,8,3) # get corners
        else:
            bboxes = None

        distance,directions,mask,intensity = get_pointsfile(os.path.join(lidar_path,'{:06d}.bin'.format(frame_idx)),return_full = return_full,bboxes = bboxes)

        points_center = np.load(os.path.join(lidar_path,'points{:03d}.npy'.format(frame_idx)))[:,-1]
        points_ = points_center[:3] @ camera_parameters_inv[:3,:3].T + camera_parameters_inv[:3,3]

        # origin
        
        lidar_origin = points_ @ c2w[:3,:3] + c2w_inv[:3,3]        

        directions = directions @ lidar2globals[frame_idx][:3,:3].T @ camera_parameters_inv[:3,:3].T
        directions = directions @ c2w[:3,:3] # tranform for the avg poses
        
        # if aggregated:
        #     tmp_points = lidar_origin + directions * distance.reshape(-1,1)
        #     Aggregated_Points.append(tmp_points)

        lidar_rays_directions.append(directions)
        lidar_distances.append(distance.flatten())
        lidar_origins.append(np.expand_dims(lidar_origin,axis= 0).repeat(distance.shape[0],axis = 0))

        intensitys.append(intensity)
        if return_full:
            filename = os.path.join(lidar_path,'{:06d}.bin'.format(frame_idx))
            scan = np.fromfile(filename, dtype=np.float32)
            # scan = scan.reshape((-1, 4)) points in nuScenes: N*5
            scan = scan.reshape((-1, 5))
            Points.append(scan[:,:3])
    lidar_distances = np.concatenate(lidar_distances).reshape(-1,1) * scale_factor
    lidar_origins = np.concatenate(lidar_origins).reshape(-1,3) * scale_factor
    # lidar_origins = np.concatenate(lidar_origins).reshape(-1,3)
    lidar_rays_directions = np.concatenate(lidar_rays_directions).reshape(-1,3)
    intensitys = np.concatenate(intensitys) # N,1
    intensitys = intensitys / intensitys.max() # for scale 
    if not return_full and not aggregated:
        lidar_depends = [i_train_lidar, lidar_distances, lidar_origins,lidar_rays_directions,intensitys]
    # elif aggregated:
        # lidar_depends = [i_train_lidar, lidar_distances, lidar_origins,lidar2globals,lidar_rays_directions,Aggregated_Points]
    else:
        lidar_depends = [i_train_lidar, lidar_distances, lidar_origins,lidar_rays_directions,Points]
    return lidar_depends

def load_lidar_label(datadir,moving_mask = True, return_full = False,finetune_skip=1,test_iou = False,label_mapping = 'nuscenes_label.yaml'):
    lidar_labels = [];lidar_distances=[];lidar_label_origins=[];lidar_rays_directions = []
    lidar_label_path = os.path.join(datadir,'sample_labels')
    lidar_label_files = glob.glob(os.path.join(lidar_label_path,'velodyne','*.bin'))
    lidar2globals = np.load(os.path.join(lidar_label_path,'lidar2global.npy'))
    i_train_lidar_label= [i for i in range(0,len(lidar_label_files),finetune_skip)] if not test_iou \
                    else [i for i in range(len(lidar_label_files)) if i%5!=0 and i%2!=0] 
    Points= []
    masks = []
    try:
        f = open(os.path.join(datadir,'lidar_points','sample_index.txt'),'r')
        sample_indexes  = f.readlines()
        sample_indexes = [int(i.split('\n')[0]) for i in sample_indexes]
    except:
        pass
    for frame_idx in i_train_lidar_label:
        if return_full:
            filename = os.path.join(lidar_label_path,'velodyne','{:06d}.bin'.format(frame_idx))
            scan = np.fromfile(filename, dtype=np.float32)
            # scan = scan.reshape((-1, 4)) points in nuScenes: N*5
            scan = scan.reshape((-1, 5))
            Points.append(scan)
        
        if moving_mask:
            f = open(os.path.join(datadir,'lidar_mask','{:04d}.txt'.format(sample_indexes[frame_idx])),'r')
            bbox = f.readlines() 
            bboxes = np.array([box.split()[1:] for box in bbox]).astype(np.float32).reshape(-1,8,3) # get corners
        distance,directions,tmp_mask = get_pointsfile(os.path.join(lidar_label_path,'velodyne','{:06d}.bin'.format(frame_idx)),bboxes = bboxes,return_full = True,return_mask = True)
        c2w = np.load(os.path.join(datadir,'c2w_recenter.npy'))
        camera_parameters = np.load(os.path.join(datadir,'c2w.npy'))
        camera_parameters_inv = np.linalg.inv(camera_parameters)
     
        # origin
        c2w_inv = np.linalg.inv(c2w)
        directions = directions @ lidar2globals[frame_idx][:3,:3].T @ camera_parameters_inv[:3,:3].T
        directions = directions @ c2w[:3,:3] # tranform for the avg poses
        lidar_rays_directions.append(directions)
        lidar_distances.append(distance.reshape(-1,))
        temp_origin = lidar2globals[frame_idx,:3,3].reshape(1,3)@ camera_parameters_inv[:3,:3].T + camera_parameters_inv[:3,3]
        temp_origin = temp_origin @ c2w[:3,:3] + c2w_inv[:3,3]
        lidar_label_origins.append(temp_origin)
        points_label = np.fromfile(os.path.join(lidar_label_path,'labels','{:06d}.label'.format(frame_idx)), dtype=np.uint8).reshape([-1,])
        with open(os.path.join(datadir,label_mapping), 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        learning_map = semkittiyaml['learning_map']
        points_label = np.vectorize(learning_map.__getitem__)(points_label)
        lidar_labels.append(points_label)
        masks.append(tmp_mask)

    if not return_full:
        mask = [(lidar_labels[i][masks[i]==1]!=255) & (lidar_distances[i]>3) &(lidar_distances[i]<100) for i in range(len(lidar_distances))]
        lidar_rays_directions = [lidar_rays_directions[i][mask[i]] for i in range(len(lidar_distances))]
        lidar_distances=[lidar_distances[i][mask[i]] for i in range(len(lidar_distances))]
        lidar_labels = [lidar_labels[i][masks[i]==1][mask[i]] for i in range(len(lidar_distances))]
        lidar_depends = [lidar_distances,lidar_labels,lidar_label_origins,lidar_rays_directions]
        return lidar_depends
    else:
        lidar_depends = [lidar_distances,lidar_labels,lidar_label_origins,lidar_rays_directions,Points]
        return lidar_depends


def in_hull(p, hull):
    """
    :param p: (N, K) test points
    :param hull: (M, K) M corners of a box
    :return (N) bool
    """
    
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)

    return flag
def get_pointsfile(filename,return_full = False,bboxes = None,return_points = False):
    scan = np.fromfile(filename, dtype=np.float32)
    # scan = scan.reshape((-1, 4)) points in nuScenes: N*5
    scan = scan.reshape((-1, 5))
    points = scan[:,:3]
    intensity = scan[:,3:4]
  
    
    
    def with_3d_bbox_hull(points,bboxes):
        # get a coarse mask
        # all_masks = np.ones((points.shape[0],))
        flag = np.zeros(points.shape[0],)
        for bbox in bboxes:
            flag += in_hull(points,bbox)
        all_masks = flag == 0 # not in the box
        # for bbox in bboxes:
        #     center = bbox.mean(axis=0).reshape(1,3)
        #     all_masks[np.linalg.norm(points-center,axis=-1)<thre]=0.

        # mask[all_masks==0] = 0.
        # points = points[all_masks==1]
        # intensity = intensity[all_masks==1]
        # return points,mask,intensity
        return all_masks
    if bboxes is not None:
        # mask =  with_3d_bbox(points,bboxes,intensity, thre = 3)
        mask =  with_3d_bbox_hull(points,bboxes)
    else:
        mask = np.ones((points.shape[0],)).astype(np.int8)
    depth = np.linalg.norm(points, 2, axis=1)
    depth_mask = (depth>3) & (depth<100) 
    if not return_full:
        mask_ = mask & depth_mask
        mask[mask_] = 0.
        points = points [mask_,:]  
        intensity = intensity[mask_]
        depth =depth[mask_] 

    # scan_x = points[:, 0]
    # scan_y = points[:, 1]
    # scan_z = points[:, 2]
    # yaw = -np.arctan2(scan_y, scan_x)

    # pitch = np.arcsin(scan_z / depth) #  right,fowward, up
    directions = points / depth[...,None]
    if not return_points:
        return depth,directions,mask,intensity
    else:
        return depth,directions,mask,intensity,points
    




# render



def get_render_poses(data_dict,RENDER_N = 100):
    poses = data_dict['poses']

    # start_pose = poses[len(poses)//2]
    # rotation = [torch.tensor(R.from_euler('y',d,degrees=True).as_matrix()).float() for d in np.linspace(-45 , 315 , RENDER_N+1)][:-1]
    # rotation = torch.stack(rotation,axis=0)
    # poses_raw = poses[id]
    # poses_raw[1,3]-=0.3 # y axis 
    # render_poses=poses_raw.broadcast_to(RENDER_N,4,4).clone()
    # render_poses[:, :3,:3] =render_poses[:, :3,:3] @ rotation[:,:3,:3]
    Poses = []

    id = len(poses)//2
    # Part 1
    # import pdb;pdb.set_trace()
    render_poses=poses[id].broadcast_to(RENDER_N//2,4,4).clone()
    # render_poses[:,:3,3] = render_poses[:,:3,3] +torch.linspace(0,RENDER_N,RENDER_N)[:,None]*(poses[id,:3,3]-poses[id,:3,3])/RENDER_N
    rotation = [torch.tensor(R.from_euler('y',d,degrees=True).as_matrix()).float() for d in np.linspace(0 , 360 , RENDER_N//2)]
    rotation = torch.stack(rotation,axis=0).to(poses.device)
    render_poses[:, :3,:3] =render_poses[:, :3,:3] @ rotation[:,:3,:3]
    Poses.append(render_poses)
    # Part 2
    
    def generate_path(pose_num,start,end):
        from pyquaternion import Quaternion
        device = start.device
        start = start.numpy()
        end = end.numpy()
        Q , r = np.linalg.qr(start[:3,:3].astype(np.float64))

        # import pdb;pdb.set_trace()
        start_rot = Quaternion(matrix = Q@r.round())
        # import pdb;pdb.set_trace()
        Q , r = np.linalg.qr(end[:3,:3].astype(np.float64))
        # Q , r = np.linalg.qr(Q)
        end_rot = Quaternion(matrix = Q@r.round())
        render_poses = []
        translation_interval = np.linspace(0,1,pose_num)[:,None]*(end[:3,3]-start[:3,3])
        for idx,q in enumerate(Quaternion.intermediates(start_rot, end_rot, pose_num-2,include_endpoints=True)):
            A = np.eye(4)
            A[:3,:3]=q.rotation_matrix
            A[:3,3] = translation_interval[idx] + start[:3,3]
            render_poses.append(A)
        
        render_poses = np.stack(render_poses)
        return torch.tensor(render_poses).to(device).float()
    # simply apply one pose
    # render_poses=poses[id].broadcast_to(RENDER_N//2,4,4).clone()
    # # end: poses[id+45,:3,3]
    # # start: poses[id,:3,3]
    # render_poses[:,:3,3] = render_poses[:,:3,3] +torch.linspace(0,1,RENDER_N//2)[:,None].to(poses.device)*(poses[id+45,:3,3]-poses[id,:3,3])
    render_poses = generate_path(RENDER_N//2,poses[id],poses[id+45])
    # import pdb;pdb.set_trace()
    Poses.append(render_poses)
    render_poses = torch.cat(Poses,dim=0)
    return render_poses

def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, dump_images=False,
                      render_factor=0, render_video_flipy=False, render_video_rot90=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs = []
    segs = []
    depths = []
    bgmaps = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]
        c2w = torch.Tensor(c2w)
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth', 'alphainv_last', 'segmantation']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(1024, 0), rays_d.split(1024, 0), viewdirs.split(1024, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()
        bgmap = render_result['alphainv_last'].cpu().numpy()
        if 'segmentation' in render_result.keys():
            seg = render_result['segmentation'].cpu().numpy()
            segs.append(seg)

        rgbs.append(rgb)
        depths.append(depth)
        bgmaps.append(bgmap)
        if i==0:
            print('Testing', rgb.shape)

        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))

    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    if render_video_flipy:
        for i in range(len(rgbs)):
            rgbs[i] = np.flip(rgbs[i], axis=0)
            depths[i] = np.flip(depths[i], axis=0)
            bgmaps[i] = np.flip(bgmaps[i], axis=0)

    if render_video_rot90 != 0:
        for i in range(len(rgbs)):
            rgbs[i] = np.rot90(rgbs[i], k=render_video_rot90, axes=(0,1))
            depths[i] = np.rot90(depths[i], k=render_video_rot90, axes=(0,1))
            bgmaps[i] = np.rot90(bgmaps[i], k=render_video_rot90, axes=(0,1))

    if savedir is not None and dump_images:
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    bgmaps = np.array(bgmaps)

    return rgbs, depths, bgmaps


def get_directions(vertical_angles,horizontal_angles): #lidar: right forward upwards
    directions = []
    for vertical_angle in vertical_angles:
        theta = vertical_angle/180*np.pi
        for horizontal_angle in horizontal_angles:
            phi = horizontal_angle
            # direction = np.array([np.cos(theta)*np.sin(phi),np.sin(theta),np.cos(theta)*np.cos(phi)])
            direction = np.array([np.cos(theta)*np.sin(phi),np.cos(theta)*np.cos(phi),np.sin(theta)])
            directions.append(direction)
    return np.stack(directions,axis=0).astype(np.float32)


def get_lidar2globals(cfg,data_dict,testsavedir='',complicated = False):
    rays_num= 32
    lidar_angles = [-30.67,-9.33, -29.33, -8.00, -28.00, -6.67,-26.67, -5.33, -25.33, -4.00, -24.00,-2.67, -22.67, -1.33,-21.33,0.00,
    -20.00,1.33,-18.67,2.67,-17.33,4.00,-16.00,5.33,-14.67,6.67,-13.33,8.00,-12.00,9.33,-10.67,10.67]
    lidar_angles = sorted(lidar_angles)
    
    ################################# Parameters setting and Loading data
    points_per_angle = 1100
    start_idx= 0 
    end_idx =80
    render_nums = 1000
    # args,render_fn,train_params = load_data(argss)
    # c2w: the transformation matrix, c2w_recenter: the recenter transformation matrix
    camera_parameters = np.load(os.path.join(cfg.data.datadir,'c2w.npy'))
    camera_parameters_inv = np.linalg.inv(camera_parameters)
    lidar_points_path = os.path.join(cfg.data.datadir,'lidar_points')
    c2w= np.load(os.path.join(cfg.data.datadir,'c2w_recenter.npy'))
    c2w_inv = np.linalg.inv(c2w)
    ################################################################ directions
    horizontal_angles = np.linspace(270,-90,points_per_angle)/180*np.pi
   
    directions = get_directions(lidar_angles,horizontal_angles)
    # lidar2ego = np.load(os.path.join(args.datadir,'lidar2ego.npy')).astype(np.float32)
    # cam2ego = np.load(os.path.join(args.datadir,'cam2ego.npy')).astype(np.float32)
    # lidar2cam = np.linalg.inv(cam2ego) @ lidar2ego
    lidar2cam = np.load(os.path.join(cfg.data.datadir,'lidar2cam.npy')).astype(np.float32)

    # right down backwards
    directions = directions @ lidar2cam[:3,:3].T# tranfer to the front camera: right backwards down 

    P = np.eye(3);P[1,1] =-1;P[2,2] = -1
    # please keep right downwards forwards before applying the c2w avg poses.
    # directions = directions @ P # transfer the requeseted coordinate system i.e. right upwards backwards
    directions = directions @ c2w[:3,:3] # tranform for the avg poses
    ################################################################ Lidar data
    lidar_data = []
    for i in range(len([f for f in os.listdir(lidar_points_path) if f.endswith('npy') and f.startswith('lidar_')])):
        lidar_data_path = os.path.join(lidar_points_path,'points{:03d}.npy'.format(i))
        lidar_data_ = np.load(lidar_data_path)[:3,:-1].transpose([1, 0])
        lidar_data_ = lidar_data_ @ camera_parameters_inv[:3,:3].T +camera_parameters_inv[:3,3]
        lidar_data_ = lidar_data_ @ c2w[:3,:3] + c2w_inv[:3,3]
        lidar_data.append(lidar_data_)       
    frame_idx = 15

    points_center = np.load(os.path.join(lidar_points_path,'points{:03d}.npy'.format(start_idx)))[:,-1]
    points_center_final = np.load(os.path.join(lidar_points_path,'points{:03d}.npy'.format(end_idx)))[:,-1]
    
    points_ = points_center[:3] @ camera_parameters_inv[:3,:3].T + camera_parameters_inv[:3,3]
    points_center_final = points_center_final[:3]@ camera_parameters_inv[:3,:3].T + camera_parameters_inv[:3,3]


    lidar_origin = points_ @ c2w[:3,:3] + c2w_inv[:3,3]
    # exit()
    lidar_origin_final = points_center_final @ c2w[:3,:3] + c2w_inv[:3,3]
    # the frequency of the lidar is 20 Hz
    time_interval = np.linspace(0,0.05,points_per_angle) # frequency 20Hz
    # lidar_origin_final = lidar_origin
    lidar_origin_interval = np.linspace(0,1,render_nums+1).reshape(-1,1) *(lidar_origin_final - lidar_origin)
    speed = False
    if complicated:
        # y
        lidar_origin_interval[:,1] = lidar_origin_interval[:,1] + 0.1 * np.random.randn(len(lidar_origin_interval))
        # x z
        lidar_origin_interval[:,[0,2]] = lidar_origin_interval[:,[0,2]] + 2 * (np.random.rand(len(lidar_origin_interval),2)*2-1)
        # import pdb;pdb.set_trace()
        # speed
        speed = True
        speed_ = np.random.rand(render_nums)*6
        # 0m/s - 10m/s
    # this is just a straight line for the ego vehicle
    np.save(os.path.join(testsavedir,'ego_trace.npy'),lidar_origin_interval+lidar_origin)
    if not speed:
        return ((lidar_origin_interval+lidar_origin)[:-1], directions)
    elif complicated:
        origins = (lidar_origin_interval+lidar_origin)[:-1].reshape(-1,1,3) + (time_interval.reshape(1,-1) * speed_.reshape(-1,1))[...,None]
        origins = np.expand_dims(origins,1).repeat(len(lidar_angles),axis = 1).reshape(render_nums,-1,3)
        return (origins, directions)

def get_lidar2globals_nv(cfg,data_dict,testsavedir='',no_load = False,complicated = False,RENDER_N=200):
    render_nums  = RENDER_N
    rays_num= 32
    lidar_angles = [-30.67,-9.33, -29.33, -8.00, -28.00, -6.67,-26.67, -5.33, -25.33, -4.00, -24.00,-2.67, -22.67, -1.33,-21.33,0.00,
    -20.00,1.33,-18.67,2.67,-17.33,4.00,-16.00,5.33,-14.67,6.67,-13.33,8.00,-12.00,9.33,-10.67,10.67]
    lidar_angles = sorted(lidar_angles)
    
    ################################# Parameters setting and Loading data
    points_per_angle = 1100
    start_frame = 0 
    frames_num = 80
    # args,render_fn,train_params = load_data(argss)
    # c2w: the transformation matrix, c2w_recenter: the recenter transformation matrix
    camera_parameters = np.load(os.path.join(cfg.data.datadir,'c2w.npy'))
    camera_parameters_inv = np.linalg.inv(camera_parameters)
    lidar_points_path = os.path.join(cfg.data.datadir,'lidar_points')
    c2w= np.load(os.path.join(cfg.data.datadir,'c2w_recenter.npy'))
    c2w_inv = np.linalg.inv(c2w)
    ################################################################ directions
    horizontal_angles = np.linspace(270,-90,points_per_angle)/180*np.pi
   

   

    if not no_load:
        lidar2global = np.load(os.path.join(cfg.data.datadir,'lidar_points','lidar2global.npy')).astype(np.float32)
    else:
        lidar2cam = np.load(os.path.join(cfg.data.datadir,'lidar2cam.npy')).astype(np.float32)
        render_poses = get_render_poses(data_dict,RENDER_N = 200)
        if isinstance(render_poses,torch.Tensor):
            if render_poses.device != 'cpu':
                render_poses = render_poses.cpu()
            render_poses= render_poses.numpy()
        # nerf -- frontcam -- global 
        P = np.eye(4);P[1,1]=-1;P[2,2]=-1;
        c2ws = [camera_parameters @ P @ c2w @ render_poses[i] for i in range(render_poses.shape[0])]
        l2ws = [c2ws[i] @ lidar2cam for i in range(len(c2ws))]
        l2ws = np.stack(l2ws,axis=0)
        # import pdb;pdb.set_trace()
        lidar2global = l2ws
        np.save(os.path.join(testsavedir,'lidar2globals.npy'),lidar2global)
    directions = []
    for frame_idx in range(render_nums):
        directions_ = get_directions(lidar_angles,horizontal_angles) # right down backwards
        directions_ = directions_ @ lidar2global[frame_idx][:3,:3].T @ camera_parameters_inv[:3,:3].T

        directions_ = directions_ @ c2w[:3,:3] # tranform for the avg poses
        directions.append(directions_)
   
    origins = lidar2global[:,:3,3] # N,3 ,global

    origins = origins @ camera_parameters_inv[:3,:3].T + camera_parameters_inv[:3,3]
    origins = origins @ c2w[:3,:3] + c2w_inv[:3,3]
    return (origins,directions)

    
   
def get_lidar2globals_compare(cfg,data_dict,testsavedir=''):
    rays_num= 32
    lidar_angles = [-30.67,-9.33, -29.33, -8.00, -28.00, -6.67,-26.67, -5.33, -25.33, -4.00, -24.00,-2.67, -22.67, -1.33,-21.33,0.00,
    -20.00,1.33,-18.67,2.67,-17.33,4.00,-16.00,5.33,-14.67,6.67,-13.33,8.00,-12.00,9.33,-10.67,10.67]
    lidar_angles = sorted(lidar_angles)
    
    ################################# Parameters setting and Loading data
    points_per_angle = 1100
    start_frame = 0 
    frames_num = 80
    # args,render_fn,train_params = load_data(argss)
    # c2w: the transformation matrix, c2w_recenter: the recenter transformation matrix
    camera_parameters = np.load(os.path.join(cfg.data.datadir,'c2w.npy'))
    camera_parameters_inv = np.linalg.inv(camera_parameters)
    lidar_points_path = os.path.join(cfg.data.datadir,'lidar_points')
    c2w= np.load(os.path.join(cfg.data.datadir,'c2w_recenter.npy'))
    c2w_inv = np.linalg.inv(c2w)
    ################################################################ directions
    horizontal_angles = np.linspace(270,-90,points_per_angle)/180*np.pi
   

    directions = [];lidar_origins=[]
    # P = np.eye(3);P[1,1] =-1;P[2,2] = -1
    # # please keep right downwards forwards before applying the c2w avg poses.
    # # directions = directions @ P # transfer the requeseted coordinate system i.e. right upwards backwards
    # directions = directions @ c2w[:3,:3] # tranform for the avg poses
    ################################################################ Lidar data
    lidar2global = np.load(os.path.join(cfg.data.datadir,'lidar_points','lidar2global.npy')).astype(np.float32)
    
    for frame_idx in tqdm(range(start_frame,frames_num)):
        points_center = np.load(os.path.join(lidar_points_path,'points{:03d}.npy'.format(frame_idx)))[:,-1]
        # import pdb;pdb.set_trace()
        points_center_next = np.load(os.path.join(lidar_points_path,'points{:03d}.npy'.format(frame_idx+1)))[:,-1]
        points_1 = points_center.copy();points_1[2] = 0.
        
        points_ = points_center[:3] @ camera_parameters_inv[:3,:3].T + camera_parameters_inv[:3,3]
        points_1 = points_1[:3]@ camera_parameters_inv[:3,:3].T + camera_parameters_inv[:3,3]
        points_center_next = points_center_next[:3]@ camera_parameters_inv[:3,:3].T + camera_parameters_inv[:3,3]
    
        # origin
        time_interval = np.linspace(0,0.05,points_per_angle) # frequency 20Hz
        lidar_origin = points_ @ c2w[:3,:3] + c2w_inv[:3,3]
        lidar_origin_next = points_center_next @ c2w[:3,:3] + c2w_inv[:3,3]
       
        # the frequency of the lidar is 20 Hz
        lidar_origin_all = -(time_interval.reshape(-1,1))@((lidar_origin_next-lidar_origin))[None,:]/(0.5/10) +lidar_origin[None,:]
        lidar_origin_all =np.concatenate([lidar_origin_all for i in range(len(lidar_angles))],axis=0)
        # lidar_origin_1 = points_1 @ c2w[:3,:3] + c2w_inv[:3,3]

        ########## save all the lidar points
        # lidar_points = lidar_data[frame_idx]
        
        directions_ = get_directions(lidar_angles,horizontal_angles) # right down backwards
        directions_ = directions_ @ lidar2global[frame_idx][:3,:3].T @ camera_parameters_inv[:3,:3].T

        directions_ = directions_ @ c2w[:3,:3] # tranform for the avg poses
        directions.append(directions_)
        # import pdb;pdb.set_trace()
        lidar_origins.append(lidar_origin_all)
    np.save(os.path.join(testsavedir,'lidar_origins.npy'),np.stack(lidar_origins))
    return (lidar_origins, directions)

@torch.no_grad()
def render_simulation(model, lidar2globals,testsavedir, ndc,device, render_kwargs,
                     ):
    rgbs = []
    segs = []
    depths = []
    bgmaps = []
    origins,directions = lidar2globals
    with torch.no_grad():
        for i,origin in enumerate(tqdm(origins)):
            if len(origins) == len(directions):
                directions_ = directions[i]
            else:
                directions_ = directions
            rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_frame(origin,directions_,)
            keys = ['rgb_marched', 'depth', 'alphainv_last', 'segmentation']
            rays_o = rays_o.flatten(0,-2).to(device)
            rays_d = rays_d.flatten(0,-2).to(device)
            viewdirs = viewdirs.flatten(0,-2).to(device)
            render_result_chunks = [
                {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
                for ro, rd, vd in zip(rays_o.split(2048, 0), rays_d.split(2048, 0), viewdirs.split(2048, 0))
            ]
            render_result = {
                k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(directions_.shape[0],-1)
                for k in render_result_chunks[0].keys()
            }
            rgb = render_result['rgb_marched'].cpu().numpy()
            depth = render_result['depth'].cpu().numpy()
            bgmap = render_result['alphainv_last'].cpu().numpy()
            # if 'segmentation' in render_result.keys():
            seg = render_result['segmentation']
            if model.save_seg_prob:
                logits_2_label = lambda x: torch.argmax(x, dim=-1)
            else:
                logits_2_label = lambda x: torch.softmax(x, dim=-1)
            seg_ = logits_2_label(seg).cpu().numpy()
            segs.append(seg_)

            rgbs.append(rgb)
            depths.append(depth)
            bgmaps.append(bgmap)
            def get_points_corrds(rays,depth):
                
                origins =rays[0]
                directions =rays[1]
                points = origins + torch.tensor(depth)*directions#/directions[:,-2:-1]
                return points

            points = get_points_corrds((rays_o,rays_d),depth)

            np.save(os.path.join(testsavedir, 'points_{:04d}.npy'.format(i)),points.detach().cpu().numpy())
            np.save(os.path.join(testsavedir, 'points_semantic_{:04d}.npy'.format(i)),seg_)
            np.save(os.path.join(testsavedir, 'points_rgb_{:04d}.npy'.format(i)),rgb)


    rgbs = np.array(rgbs)
    depths = np.array(depths)
    bgmaps = np.array(bgmaps)

    return rgbs, depths, bgmaps



@torch.no_grad()
def render_lidar_compare(model, lidar2globals,testsavedir, ndc, render_kwargs,
                     ):
    rgbs = []
    segs = []
    depths = []
    bgmaps = []
    origins,directions = lidar2globals
    with torch.no_grad():
        for i in tqdm(range(len(origins))):
            origin = origins[i]
            directions_ = directions[i]
            # origin,directions = lidar2globals[i]
            rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_frame(origin,directions_)

            keys = ['rgb_marched', 'depth', 'alphainv_last', 'segmentation']
            rays_o = rays_o.flatten(0,-2)
            rays_d = rays_d.flatten(0,-2)
            viewdirs = viewdirs.flatten(0,-2)
            render_result_chunks = [
                {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
                for ro, rd, vd in zip(rays_o.split(1024, 0), rays_d.split(1024, 0), viewdirs.split(1024, 0))
            ]
            render_result = {
                k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(directions_.shape[0],-1)
                for k in render_result_chunks[0].keys()
            }
            rgb = render_result['rgb_marched'].cpu().numpy()
            depth = render_result['depth'].cpu().numpy()
            bgmap = render_result['alphainv_last'].cpu().numpy()
            # if 'segmentation' in render_result.keys():
            seg = render_result['segmentation']
            logits_2_label = lambda x: torch.argmax(x, dim=-1)
            seg_ = logits_2_label(seg).cpu().numpy()
            segs.append(seg_)

            rgbs.append(rgb)
            depths.append(depth)
            bgmaps.append(bgmap)
            def get_points_corrds(rays,depth):
                
                origins =rays[0]
                directions =rays[1]
                points = origins + torch.tensor(depth)*directions#/directions[:,-2:-1]
                return points

            points = get_points_corrds((rays_o,rays_d),depth)

            np.save(os.path.join(testsavedir, 'points_{:03d}.npy'.format(i)),points.detach().cpu().numpy())
            
            np.save(os.path.join(testsavedir, 'points_semantic_{:03d}.npy'.format(i)),seg_)
            np.save(os.path.join(testsavedir, 'points_rgb_{:03d}.npy'.format(i)),rgb)


    rgbs = np.array(rgbs)
    depths = np.array(depths)
    bgmaps = np.array(bgmaps)

    return rgbs, depths, bgmaps

@torch.no_grad()
def render_lidar_eval(model, lidar_depends,testsavedir, ndc, render_kwargs):
    lidar_dists, lidar_origins,lidar_rays_directions,lidar_labels = lidar_depends
    segs = []
    depths = []
    import os
    for i in tqdm(range(len(lidar_origins))):
        rays_d= torch.tensor(lidar_rays_directions[i]).float()
        rays_o = torch.broadcast_to(torch.tensor(lidar_origins[i]).reshape(1,3),
                            rays_d.shape).float() 
        viewdirs = rays_d/torch.linalg.norm(rays_d,dim=-1,keepdim=True)
        # directions= lidar_rays_directions[i]
        # origin,directions = lidar2globals[i]
        keys = ['rgb_marched', 'depth', 'alphainv_last', 'segmentation']
       
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(1024, 0), rays_d.split(1024, 0), viewdirs.split(1024, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(rays_d.shape[0],-1)
            for k in render_result_chunks[0].keys()
        }
        depth = render_result['depth'].cpu().numpy()
        # if 'segmentation' in render_result.keys():
        seg = render_result['segmentation']
        logits_2_label = lambda x: torch.argmax(x, dim=-1)
        seg_ = logits_2_label(seg).cpu().numpy()
        segs.append(seg_)

        depths.append(depth)
        def get_points_corrds(rays,depth):
            origins =rays[0]
            directions =rays[1]
            points = origins + torch.tensor(depth)*directions#/directions[:,-2:-1]
            return points

        points = get_points_corrds((rays_o,rays_d),depth)

        # np.save(os.path.join(testsavedir, 'points_{:03d}.npy'.format(i)),points.detach().cpu().numpy())
        if i <len(lidar_origins) - len(lidar_labels):
            np.save(os.path.join(testsavedir, 'depth_{:03d}.npy'.format(i)),depth)
            np.save(os.path.join(testsavedir, 'gt_depth_{:03d}.npy'.format(i)),lidar_dists[i])
        # only save key frame semantic labels
        if i >= len(lidar_origins) - len(lidar_labels):
            np.save(os.path.join(testsavedir, 'points_semantic_{:03d}.npy'.format(i)),seg_)
            np.save(os.path.join(testsavedir, 'gt_points_semantic_{:03d}.npy'.format(i)),lidar_labels[i-len(lidar_origins) + len(lidar_labels)])

    
    import os
    import yaml
    with open('/SSD_DISK/users/zhangjunge/Scenes/6cam_scene1_revision/NeRFlidar_label_v7.yaml', 'r') as stream:
        # to 8 classes    
        semkittiyaml = yaml.safe_load(stream)
    learning_map = semkittiyaml['learning_map']
    points_labels = []
    for idx, label in enumerate(lidar_labels):
        points_label = np.vectorize(learning_map.__getitem__)(segs[idx+len(lidar_origins) - len(lidar_labels)])
        points_labels.append(points_label)

    # points_labels = np.concatenate(points_labels)
    # lidar_labels = np.concatenate(lidar_labels)
    from miou_cal import eval_miou
    per_class_iou = eval_miou(points_labels,lidar_labels)
    with open(os.path.join(testsavedir,'iou.txt'),'w') as f:
        for idx in range(len(per_class_iou)):
            f.write(f'{per_class_iou[idx]}\n')
        f.write(f'miou is {per_class_iou.mean()}')

    return None


def load_time_lidar(root_dir,time_scale):
    times = np.loadtxt(os.path.join(root_dir,'timestamps.txt'))

    # time_min, time_max = times.min() , times.max()
    # 1e6 is 1s
    time_unit = 1e6
    time_min,time_unit = time_scale
    times = (times - time_min) / time_unit
    return times