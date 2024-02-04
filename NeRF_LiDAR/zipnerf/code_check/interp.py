
import numpy as np
import torch
import os
def get_pose_raw(time,tracks):
    # time [N_rays,1]
    # tracks [N_obj,N_timestamp, N_info]
    # obj_pose: N_rays, N_obj,N_info
    if tracks is not None:
        time_diff = torch.abs(time[...,None] - tracks[:,:,-2].unsqueeze(0))
        time_indice = torch.argmin(time_diff,dim = -1)
        
        tracks = tracks.unsqueeze(0).expand(time.shape[0],-1,-1,-1)
        obj_pose = torch.gather(tracks , dim = -2 , index= time_indice[...,None,None].expand(-1,-1,-1,tracks.shape[-1])).squeeze(dim=-2) # N_rays, N_batch, 9
        
        min_time = time_diff.min(dim=-1)[0]
        no_valid_mask = min_time > 2
        obj_pose[no_valid_mask][...,4:7] *= 0.001 # mini wlh , for no intersection

        return obj_pose
    else:
        return None

def get_pose(time, tracks):
    # time [N_rays,1]
    # tracks [N_obj,N_timestamp, N_info]
    # obj_pose: N_rays, N_obj,N_info
    time_record = tracks[:,:,-2].unsqueeze(0).expand(time.shape[0],-1,-1)
    if tracks is not None:
        time_diff = torch.abs(time[...,None] - time_record) # N_rays, N_obj, N_timestamp
        
        time_diff_sorted, indices = torch.sort(time_diff, dim=-1) # Sort along the timestamp dimension
        closest_indices = indices[...,:2] # Get the indices of the two closest timestamps

        # Expand dimensions for gather
        time_expand = time.unsqueeze(-1).expand(-1,tracks.shape[0],tracks.shape[-1])
        t_expand = time_record
        tracks_expand = tracks.unsqueeze(0).expand(time.shape[0],-1,-1,-1)
        # tracks_expand = time_record
        closest_indices_expand = closest_indices.unsqueeze(-1).expand(-1,-1,-1,tracks.shape[-1])

        # Get the two closest track timestamps
        t1 = torch.gather(t_expand, dim=-1, index=closest_indices_expand[...,0,:]).squeeze(-1)
        t2 = torch.gather(t_expand, dim=-1, index=closest_indices_expand[...,1,:]).squeeze(-1)
        
        # Calculate interpolation weights
        
        total_diff = torch.abs(t1 - t2) + 1e-12
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
        
        min_time = time_diff_sorted[...,0]
        no_valid_mask = min_time > 2
        obj_pose[no_valid_mask][...,4:7] *= 0.001 # mini wlh , for no intersection

        return obj_pose
    else:
        return None


if __name__ == '__main__':
    tracks = np.load('/data1/junge/nuscenes/0213_front/tracks.npy')
    tracks = torch.from_numpy(tracks)
    from bboxes import load_time 
    timestamps, scale = load_time('/data1/junge/nuscenes/0213_front/')

    
    # obj_idx = 1
    for obj_idx in range(len(tracks)):
        times_ = []
        times= tracks[obj_idx,:,-2].reshape(-1,1)
        for time in times:
            if time in timestamps:
                times_.append(time)
            else:
                pass
        # if len(times_) == 0:
        #     import pdb;pdb.set_trace() 
        print(times_)
        times_ = np.stack(times_)
        times = times_
        times= torch.from_numpy(times)
        
        query1 = get_pose(times,tracks)
        query2 = get_pose_raw(times,tracks)

        for time_idx in range(50):
            if (query1[time_idx,obj_idx,:] - query2[time_idx,obj_idx,:]).abs().sum() >1e-3:
                print('quer1 for time {} is {}'.format(times_[time_idx],query1[time_idx,obj_idx,:]))
                print('quer2 for time {} is {}'.format(times_[time_idx],query2[time_idx,obj_idx,:]))
                import pdb;pdb.set_trace()

            