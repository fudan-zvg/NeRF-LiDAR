import torch
from pyquaternion import Quaternion
import numpy as np
import os
def scale_frames(p, sc_factor, inverse=False):
    """Scales points given in N_frames in each dimension [xyz] for each frame or rescales for inverse==True

    Args:
        p: Points given in N_frames frames [N_points, N_frames, N_samples, 3]
        sc_factor: Scaling factor for new frame [N_points, N_frames, 3]
        inverse: Inverse scaling if true, bool

    Returns:
        p_scaled: Points given in N_frames rescaled frames [N_points, N_frames, N_samples, 3]
    """
    # Take 150% of bbox to include shadows etc.
    dim = torch.tensor([1., 1., 1.]).to(p.device) * sc_factor
    # dim = torch.tensor([0.1, 0.1, 0.1]) * sc_factor

    half_dim = dim / 2
    # scaling_factor = (1 / (half_dim + 1e-9)).unsqueeze(-2)
    scaling_factor = (1 / (half_dim + 1e-9))

    if not inverse:
        p_scaled = scaling_factor * p
    else:
        p_scaled = (1/scaling_factor) * p

    return p_scaled
def rotate_yaw(p, yaw):
    """Rotates p with yaw in the given coord frame with y being the relevant axis and pointing downwards

    Args:
        p: 3D points in a given frame [N_pts, N_frames, 3]/[N_pts, N_frames, N_samples, 3]
        yaw: Rotation angle

    Returns:
        p: Rotated points [N_pts, N_frames, N_samples, 3]
    """
    # p of size [batch_rays, n_obj, samples, xyz]
    # if len(p.shape) < 4:
    #     p = p.unsqueeze(-2)

    # c_y = torch.cos(yaw).unsqueeze(-1)
    # s_y = torch.sin(yaw).unsqueeze(-1)
    c_y = torch.cos(yaw)
    s_y = torch.sin(yaw)
    p_x = c_y * p[..., 0] - s_y * p[..., 2]
    p_y = p[..., 1]
    p_z = s_y * p[..., 0] + c_y * p[..., 2]

    return torch.stack([p_x, p_y, p_z], dim=-1)
def rotate_yaw_x(p, yaw):
    """Rotates p with yaw in the given coord frame with x being the relevant axis and pointing downwards

    Args:
        p: 3D points in a given frame [N_pts, N_frames, 3]/[N_pts, N_frames, N_samples, 3]
        yaw: Rotation angle

    Returns:
        p: Rotated points [N_pts, N_frames, N_samples, 3]
    """
    # p of size [batch_rays, n_obj, samples, xyz]
    # if len(p.shape) < 4:
    #     p = p.unsqueeze(-2)

    # c_y = torch.cos(yaw).unsqueeze(-1)
    # s_y = torch.sin(yaw).unsqueeze(-1)
    c_y = torch.cos(yaw)
    s_y = torch.sin(yaw)
    p_x = p[..., 0]
    p_y = c_y * p[..., 1] - s_y * p[..., 2]
    p_z = s_y * p[..., 1] + c_y * p[..., 2]

    return torch.stack([p_x, p_y, p_z], dim=-1)
def rotate_yaw_z(p, yaw,pitch = None):
    """Rotates p with yaw in the given coord frame with z being the relevant axis and pointing downwards

    Args:
        p: 3D points in a given frame [N_pts, N_frames, 3]/[N_pts, N_frames, N_samples, 3]
        yaw: Rotation angle

    Returns:
        p: Rotated points [N_pts, N_frames, N_samples, 3]
    """
    # p of size [batch_rays, n_obj, samples, xyz]
    # if len(p.shape) < 4:
    #     p = p.unsqueeze(-2)
    p_x = p[...,0]
    p_y = p[...,1]
    p_z = p[...,2]
    # c_y = torch.cos(yaw).unsqueeze(-1)
    # s_y = torch.sin(yaw).unsqueeze(-1)
    if pitch is not None:
        c_y = torch.cos(pitch)
        s_y = torch.sin(pitch)
        p_x = c_y * p_x - s_y * p_x
        p_y = p_y
        p_z = s_y * p_x + c_y * p_z

    c_y = torch.cos(yaw)
    s_y = torch.sin(yaw)

    # p_x = c_y * p[..., 0] - s_y * p[..., 2]
    # p_y = p[..., 1]
    # p_z = s_y * p[..., 0] + c_y * p[..., 2]
    p_x = c_y * p_x - s_y * p_y
    p_y = s_y * p_x + c_y * p_y
    p_z = p_z

    # raw at pitch
   

    return torch.stack([p_x, p_y, p_z], dim=-1)

def world2object(pts, dirs, pose, theta_z, dim=None, inverse=False):
    """Transform points given in world frame into N_obj object frames

    Object frames are scaled to [[-1.,1], [-1.,1], [-1.,1]] inside the 3D bounding box given by dim

    Args:
        pts: N_pts times 3D points given in world frame, [N_pts, 3]
        dirs: Corresponding 3D directions given in world frame, [N_pts, 3]
        pose: object position given in world frame, [N_pts, N_obj, 3]/if inverse: [N_pts, 3]
        # theta_y: Yaw of objects around world y axis, [N_pts, N_obj]/if inverse: [N_pts] for kitti format
        theta_z: Yaw of objects around world z axis, [N_pts, N_obj]/if inverse: [N_pts]
        dim: Object bounding box dimensions, [N_pts, N_obj, 3]/if inverse: [N_pts, 3]
        inverse: if true pts and dirs should be given in object frame and are transformed back into world frame, bool
            For inverse: pts, [N_pts, N_obj, 3]; dirs, [N_pts, N_obj, 3]

    Returns:
        pts_w: 3d points transformed into object frame (world frame for inverse task)
        dir_w: unit - 3d directions transformed into object frame (world frame for inverse task)
    """

    # Prepare args if just one sample per ray-object or world frame only
    # if len(pts.shape) == 3:
    # [batch_rays, n_obj, samples, xyz]
    if pts.dim() == 3:
        n_sample_per_ray = pts.shape[1]
        pose = torch.repeat_interleave(pose,n_sample_per_ray, dim = 0)
        theta_z = torch.repeat_interleave(theta_z,n_sample_per_ray, dim = 0)
        if dim is not None:
            dim = torch.repeat_interleave(dim,n_sample_per_ray, dim = 0)
        if len(dirs.shape) == 2:
            dirs = torch.repeat_interleave(dirs,n_sample_per_ray,dim = 0)

        pts = pts.reshape(-1, 3)

    # Shift the object reference point to the middle of the bbox (vkitti2 specific)
    # y_shift = (torch.tensor([0., -1., 0.]).unsqueeze(0) if inverse else
    #            torch.tensor([0., -1., 0.]).unsqueeze(0).unsqueeze(1)) * \
    #           (dim[..., 1] / 2).unsqueeze(-1)
    # pose_w = pose + y_shift
    pose_w = pose

    # Describes the origin of the world system w in the object system o


    t_w_o = rotate_yaw_z(-pose_w, theta_z)

    if not inverse:
        N_obj = theta_z.shape[1]
        pts_w = torch.repeat_interleave(pts.unsqueeze(1),N_obj,dim=1)
        dirs_w =torch.repeat_interleave(dirs.unsqueeze(1),N_obj,dim=1)

        # Rotate coordinate axis
        # TODO: Generalize for 3d roaations
        pts_o = rotate_yaw_z(pts_w, theta_z) + t_w_o
        dirs_o = rotate_yaw_z(dirs_w, theta_z)
        # theta_y = torch.ones_like(theta_z)
        # dirs_o = rotate_yaw(dirs_o, theta_y*float(np.deg2rad(-15)))
        # Scale rays_o_v and rays_d_v for box [[-1.,1], [-1.,1], [-1.,1]]
        if dim is not None:
            pts_o = scale_frames(pts_o, dim)
            dirs_o = scale_frames(dirs_o, dim)

        # Normalize direction
        # dirs_o = dirs_o / torch.norm(dirs_o, dim=3, keepdim=True)
        dirs_o = dirs_o / torch.norm(dirs_o, dim=-1, keepdim=True)
        return [pts_o, dirs_o]

    else:
        pts_o = pts.unsqueeze(0).unsqueeze(2)
        dirs_o = dirs
        if dim is not None:
            pts_o = scale_frames(pts_o, dim.unsqueeze(0), inverse=True)
            if dirs is not None:
                dirs_o = scale_frames(dirs_o, dim, inverse=True)

        pts_o = pts_o - t_w_o
        pts_w = rotate_yaw_z(pts_o, -theta_z)[0, :]

        if dirs is not None:
            dirs_w = rotate_yaw_z(dirs_o, -theta_z)
            # Normalize direction
            dirs_w = dirs_w / torch.norm(dirs_w, dim=-1, keepdim=True)
        else:
            dirs_w = None

        return [pts_w, dirs_w]

def box_pts(pts,viewdirs,obj_pose,transform = True,sym=False):
    # points: N_rays,N_samples,3
    # obj_pose: N_rays, N_obj, 9
    assert isinstance(obj_pose,torch.Tensor) 

    center = obj_pose[:,:,:3]; theta_z = obj_pose[:,:,3] ;wlh = obj_pose[:,:,4:7]
    if transform:
        pts_o, dirs_o = world2object(pts, viewdirs, center, theta_z, dim=wlh, inverse=False)
        intersection_map = (pts_o[...,0].abs() < 1) & (pts_o[...,1].abs() < 1) & (pts_o[...,2].abs() < 1)
        intersection_map = intersection_map.reshape(pts.shape[0],pts.shape[1],-1).detach()
        pts_o = pts_o.reshape(pts.shape[0],pts.shape[1],-1,3)
        dirs_o = dirs_o.reshape(pts.shape[0],pts.shape[1],-1,3)
    else:
        import pdb;pdb.set_trace()
        pts_o = scale_frames(pts_o, dim=wlh)
        dirs_o = scale_frames(dirs_o, dim=wlh)
        # pts_o, dirs_o = pts, viewdirs
        intersection_map = (pts_o[...,0].abs() < 1) & (pts_o[...,1].abs() < 1) & (pts_o[...,2].abs() < 1)
        intersection_map = intersection_map.reshape(pts.shape[0],pts.shape[1],-1).detach()

    # use symmetry information
    if sym:
        # no grad for sym
        with torch.no_grad():
            pts_o_sym,dirs_o_sym = symmetrize(pts_o,dirs_o,obj_pose = obj_pose) 
        # concatenate
        pts_o = torch.cat([pts_o,pts_o_sym],dim=0)
        dirs_o = torch.cat([dirs_o,dirs_o_sym],dim=0)
        intersection_map = torch.cat([intersection_map,intersection_map],dim=0)
        
    # N_rays, N_samples, N_obj,
    return pts_o, dirs_o,intersection_map

def symmetrize(pts_o,dirs_o,obj_pose):
    # device = pts_o.device
    # pts_shape = pts_o.shape[:-1]
    # N_rays, N_samples, N_obj = pts_shape
    # center = obj_pose[:,:,:3]; theta_z = obj_pose[:,:,3];
    # # pts: N_rays, N_samples, N_obj, 3
    # # obj_pose: N_rays, N_obj, 9
    # pts_o = pts_o.view(-1,N_obj,3) #  N_rays*N_samples, N_obj, 3
    # # padding to homogeneous
    # pts_o = torch.cat([pts_o,torch.ones_like(pts_o)[...,0:1]],dim=-1) #  N_rays*N_samples, N_obj, 4
    # dirs_o = dirs_o.view(-1,N_obj,3)
    # # pts_sym:  N_rays*N_samples, N_obj, 3
    # identity = torch.eye(4)[None,None,...] # 1,1,4,4
    # T = identity.repeat(N_rays,N_obj,1,1).to(device) #  N_rays, N_obj, 4,4
    # # center is translation
    # T[:,:,:3,3] = center
    # # theta_z is for z axis(Rotation)
    # c_y = torch.cos(theta_z) #  N_rays, N_obj
    # s_y = torch.sin(theta_z) 
    # T[:,:,0,0] = c_y
    # T[:,:,0,1] = -s_y
    # T[:,:,1,0] = s_y
    # T[:,:,1,1] = c_y
    # T = T.repeat(N_samples,1,1,1) # N_rays*N_samples, N_obj, 4,4
    # R = T[:,:,:3,:3] 
    # # symmetry at x axis
    # # sym_mtx = torch.eye(4);sym_mtx[0,0] = -1
    # sym_mtx = torch.eye(4);sym_mtx[1,1] = -1

    # sym_mtx = sym_mtx.unsqueeze(0).unsqueeze(0) # 1,1,4,4
    # sym_mtx = sym_mtx.repeat(N_rays*N_samples, N_obj,1,1).to(device) # N_rays*N_samples, N_obj, 4,4
    # # T = [[R t][0 1]] 
    # # T_inv = [[R^T -R^T@t][0 1]]
    # T_inv = T.clone() 
    # R_inv = R.transpose(-2,-1) # transpose #  N_rays*N_samples, N_obj, 3,3
    # T_inv[:,:,:3,:3] = R_inv
    # #  N_rays*N_samples, N_obj, 3,3 @  N_rays*N_samples, N_obj, 3
    # A = R_inv.reshape(-1,3,3)
    # B = T[:,:,:3,3].reshape(-1,3,1)
    # T_inv[:,:,:3,3] = -A.bmm(B).view(N_rays*N_samples, N_obj, 3) 
    
    # T = T.view(-1,4,4)
    # T_inv = T_inv.view(-1,4,4)
    # R = R.view(-1,3,3)
    # R_inv = R_inv.view(-1,3,3)
    # sym_mtx = sym_mtx.view(-1,4,4)
    # pts_o = pts_o.view(-1,4,1)
    # dirs_o = dirs_o.view(-1,3,1)
    # #  N_rays*N_samples, N_obj, 4,4
    # #  N_rays*N_samples, N_obj, 4,4 

    # # pts: N_rays, N_samples, N_obj, 3
    # # pts: N_rays, N_samples, N_obj, 3
    # pts_o_sym = T_inv.bmm(sym_mtx).bmm(T).bmm(pts_o)
    # pts_o_sym = pts_o_sym.view(N_rays, N_samples, N_obj, 4)[...,:3]
    # dirs_o_sym = R_inv.bmm(sym_mtx[:,:3,:3]).bmm(R).bmm(dirs_o)
    # dirs_o_sym = dirs_o_sym.view(N_rays, N_samples, N_obj, 3)

    # already in the objects frame
    # device = pts_o.device
    # pts_shape = pts_o.shape[:-1]
    # N_rays, N_samples, N_obj = pts_shape
    # sym_mtx = torch.eye(3);sym_mtx[1,1] = -1
    # sym_mtx = sym_mtx.unsqueeze(0).unsqueeze(0) # 1,1,3,3
    # sym_mtx = sym_mtx.repeat(N_rays*N_samples, N_obj,1,1).to(device) # N_rays*N_samples, N_obj, 3,3
    # sym_mtx = sym_mtx.view(-1,3,3)
    # pts_o = pts_o.view(-1,3,1)
    # dirs_o = dirs_o.view(-1,3,1)
    # pts_o_sym = sym_mtx.bmm(pts_o).view(N_rays, N_samples, N_obj, 3)
    # dirs_o_sym = sym_mtx.bmm(dirs_o).view(N_rays, N_samples, N_obj, 3)
    pts_o_sym = pts_o.clone().detach()
    # symmetry at y axis?
    pts_o_sym[...,1] *= -1
    dirs_o_sym = dirs_o.clone().detach()
    dirs_o_sym[...,1] *= -1
    return pts_o_sym,dirs_o_sym

def ray_box_intersection(ray_o, ray_d, aabb_min=None, aabb_max=None):
    """Returns 1-D intersection point along each ray if a ray-box intersection is detected

    If box frames are scaled to vertices between [-1., -1., -1.] and [1., 1., 1.] aabbb is not necessary

    Args:
        ray_o: Origin of the ray in each box frame, [rays, boxes, 3]
        ray_d: Unit direction of each ray in each box frame, [rays, boxes, 3]
        (aabb_min): Vertex of a 3D bounding box, [-1., -1., -1.] if not specified
        (aabb_max): Vertex of a 3D bounding box, [1., 1., 1.] if not specified

    Returns:
        z_ray_in:
        z_ray_out:
        intersection_map: Maps intersection values in z to their ray-box intersection
    """
    if aabb_min is None:
        aabb_min = torch.ones_like(ray_o) * -1.
    if aabb_max is None:
        aabb_max = torch.ones_like(ray_o)

    inv_d = torch.reciprocal(ray_d)

    t_min = (aabb_min - ray_o) * inv_d
    t_max = (aabb_max - ray_o) * inv_d

    t0 = torch.min(t_min, t_max)
    t1 = torch.max(t_min, t_max)

    t_near = torch.max(torch.max(t0[..., 0], t0[..., 1]), t0[..., 2])
    t_far = torch.min(torch.min(t1[..., 0], t1[..., 1]), t1[..., 2])

    intersection_map = torch.where(t_far > t_near)

    positive_far = torch.where(t_far[intersection_map].reshape(-1) > 0)
    intersection_map = intersection_map[0][positive_far], intersection_map[1][positive_far]

    if not intersection_map[0].shape[0] == 0:
        z_ray_in = t_near[intersection_map]
        z_ray_out = t_far[intersection_map]
    else:
        return None, None, None

    return z_ray_in, z_ray_out, intersection_map

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
            # pose_array[7:10]*=1e-5
            pose_array[7:10]=0
        elif t> recored_time.max():
            orient = Quaternion(track[-1,3:7])
            theta_z = np.array([orient.yaw_pitch_roll[0]])
            pose_array = np.concatenate([track[-1,:3],theta_z,track[-1,7:]])
            # invalid bbox
            # pose_array[7:10]*=1e-5
            pose_array[7:10]=0
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


# def get_pose(time,tracks):
#     # time [N_rays,1]
#     # tracks [N_obj,N_timestamp, N_info]
#     # obj_pose: N_rays, N_obj,N_info
#     if tracks is not None:
#         time_diff = torch.abs(time[...,None] - tracks[:,:,-2].unsqueeze(0))
#         time_indice = torch.argmin(time_diff,dim = -1)
        
#         tracks = tracks.unsqueeze(0).expand(time.shape[0],-1,-1,-1)
#         obj_pose = torch.gather(tracks , dim = -2 , index= time_indice[...,None,None].expand(-1,-1,-1,tracks.shape[-1])).squeeze(dim=-2) # N_rays, N_batch, 9
        
#         min_time = time_diff.min(dim=-1)[0]
#         no_valid_mask = min_time > 2
#         obj_pose[no_valid_mask][...,4:7] *= 0.001 # mini wlh , for no intersection

#         return obj_pose
#     else:
#         return None
def get_pose(time, tracks):
    # time [N_rays,1]
    # tracks [N_obj,N_timestamp, N_info]
    # obj_pose: N_rays, N_obj,N_info
    if tracks is not None:
        time_diff = torch.abs(time[...,None] - tracks[:,:,-2].unsqueeze(0)) # N_rays, N_obj, N_timestamp
        time_diff_sorted, indices = torch.sort(time_diff, dim=-1) # Sort along the timestamp dimension
        closest_indices = indices[...,:2] # Get the indices of the two closest timestamps

        # Expand dimensions for gather
        time_expand = time.unsqueeze(-1).expand(-1,tracks.shape[0],tracks.shape[-1])
        t_expand = tracks[:,:,-2].unsqueeze(0).expand(time.shape[0],-1,-1)
        tracks_expand = tracks.unsqueeze(0).expand(time.shape[0],-1,-1,-1)
        closest_indices_expand = closest_indices.unsqueeze(-1).expand(-1,-1,-1,tracks.shape[-1])

        # Get the two closest track timestamps
        t1 = torch.gather(t_expand, dim=-1, index=closest_indices_expand[...,0,:]).squeeze(-1)
        t2 = torch.gather(t_expand, dim=-1, index=closest_indices_expand[...,1,:]).squeeze(-1)
        
        # Calculate interpolation weights
        
        total_diff = torch.abs(t1 - t2) + 1e-9
        # weight1 = torch.abs(time_expand - t1) / total_diff
        weight1 = torch.abs(time_expand - t2) / total_diff
        weight1 = weight1.clamp(0,1)

        weight2 = 1 - weight1

        # weight2 = torch.abs(time_expand - t2) / total_diff
        # weight2 = weight2.clamp(0,1)
        # Get the two closest track info
        # .expand(-1,-1,tracks.shape[-2],-1)
        info1 = torch.gather(tracks_expand, dim=-2, index=closest_indices_expand[...,0,:].unsqueeze(-2)).squeeze(dim=-2)
        info2 = torch.gather(tracks_expand, dim=-2, index=closest_indices_expand[...,1,:].unsqueeze(-2)).squeeze(dim=-2)

        # Interpolate the track info
        obj_pose = weight1* info1 + weight2 * info2
        
        # min_time = time_diff_sorted[...,0]
        # no_valid_mask = min_time > 2
        # obj_pose[no_valid_mask][...,4:7] *= 0.001 # mini wlh , for no intersection

        return obj_pose
    else:
        return None
"""
city_labels:
  0: 'road' # driveable_surface
  1: 'sidewalk' # sidewalk
  2: 'building' # manmade
  3: 'wall' # manmade
  4: 'fence' # manmade
  5: 'pole' # manmade
  6: 'traffic-light' # manmade
  7: 'traffic-sign' # manmade
  8: 'vegetation'
  9: 'terrain'
  10: 'sky' # None
  11: 'person'
  12: 'rider'
  13: 'car'
  14: 'truck'
  15: 'bus'
  16: 'train'
  17: 'motorcycle'
  18: 'bicycle'
"""
def query_class(class_name):
    if 'human' in class_name:
        return 11
    elif 'truck' in class_name or 'trailer' in class_name  or 'construction' in class_name:
        return 14
    elif 'bus' in class_name:
        return 15
    elif 'car' in class_name:
        return 13
    else:
        return 255
    


def visualize_box(dataset):
    # get tracks
    obj_info = dataset.bboxes[0]
    tracks = []
    for track_id,bbox_infos in obj_info.items():
        tracks.append(bbox_infos) # obj_pose
    tracks = np.stack(tracks)
    tracks = torch.from_numpy(tracks)

    for idx in range(0,dataset.size,5):
        batch = dataset.generate_ray_batch(idx)
        height, width = batch['origins'].shape[:2]
        num_rays = height * width
        batch = {k: v.reshape((num_rays, -1)) for k, v in batch.items() if v is not None}
        rays_o = batch['origins']
        rays_d = batch['directions']
        device = batch['timestamp'].device
        obj_pose = get_pose(batch['timestamp'],tracks.to(device)) # N_obj, N_timestamp, 9

        # box in pts
        center = obj_pose[:,:,:3]; theta_z = obj_pose[:,:,3] ;wlh = obj_pose[:,:,4:7]

        rays_o_o, dirs_o = world2object(rays_o, rays_d, center, theta_z, dim=wlh, inverse=False)
        z_ray_in_o, z_ray_out_o, intersection_map = ray_box_intersection(rays_o_o, dirs_o)
        canvas = torch.zeros((height , width)).reshape(-1)
        interset_idx = intersection_map[0]
        interset_idx_ = intersection_map[1].float()
        canvas[interset_idx] = interset_idx_

        canvas = canvas.detach().cpu().numpy().reshape((height , width))
        os.makedirs('vis_bbox',exist_ok = True)
        import cv2
        cv2.imwrite('vis_bbox/mask_{:04d}.png'.format(idx),(canvas==0)*255)
        gt_image = batch['rgb'].detach().cpu().numpy().reshape((height , width,3))
        cv2.imwrite('vis_bbox/GT_{:04d}.png'.format(idx),np.clip(gt_image,a_min=0,a_max=1)*255)

        canvas_white = np.ones((height , width,3)) * 255
        canvas_white[canvas!=0] = gt_image[canvas!=0]*255
        cv2.imwrite('vis_bbox/crop_GT_{:04d}.png'.format(idx),canvas_white)
        print(f'Done visualization for {idx} image')


def simu_info (mode,tracks, angle = 0) :
    # replay, lane_shift, removal, rotate 
    
    if mode == 'replay':
        # do nothing
        angle = 0
        tracks = tracks
    elif mode == 'laneshift':
        angle = 0
        # move left
        tracks[:,:,1] += 0.03 
    elif mode == 'removal':
        # remove all bboxes
        angle = 0
        tracks = None
    elif mode == 'rotate':
        angle = 15
        tracks = tracks

    return angle , tracks


def edit_poses(poses,shift_dist = 0.03):
    # x,y,z forward, left, down
    poses[:,1,3] += shift_dist
    print('shift ego vehicle')
    return poses

def edit_tracks(bboxes,edit_track):
    obj_info , obj_type_info = bboxes
    num_tracks = len(obj_type_info)
    if edit_track.ndim!= 3:
        edit_track = edit_track[None,...]
    count = 0

    for track in edit_track:
        # obj_info = np.concatenate([obj_info,track],axis=0)
        obj_info[count+num_tracks] = track
        obj_type_info[count+num_tracks] = 'car_fusion'
    return (obj_info , obj_type_info)
