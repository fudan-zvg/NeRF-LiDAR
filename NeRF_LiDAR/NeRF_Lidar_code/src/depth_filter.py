import numpy as np
import os

def depth_filter(points,points_semantic = None,return_mask = False,threshold =1,radius = 1,width = 3):
    points_ = points.reshape(32,-1,3)
    matrix = np.stack([np.roll(points_, i, axis = 1) for i in range(-width,width +1) if i!=0],axis = -1)

    raw = np.broadcast_to(points_[...,None],matrix.shape)
    dist = np.linalg.norm(raw-matrix,axis = -2)
    count = (dist[...,:]<radius).sum(axis=-1)
    if points_semantic is None:
        mask = count> threshold
    else:
        # judge the edge via semantic label
        points_semantic_ = points_semantic.reshape(32,-1,)
        # 32,W,N, N is the window width
        # import pdb;pdb.set_trace()
        semantic_matrix = np.stack([np.roll(points_semantic_, i, axis = 1) != points_semantic_ for i in [-1,1]],axis = -1)
        semantic_count = semantic_matrix.sum(-1) # edge or not  
        mask = (count> threshold) | (semantic_count > 0) | (points_semantic_ == 13) # do not apply depth filter on car
    mask = mask.reshape(-1,)

    if not return_mask:
        return points[mask].reshape(-1,3)
    else:
        return mask


if __name__ == '__main__':
    

    points = np.load('points_0000.npy')
    points_semantic = np.load('points_semantic_0000.npy')

    with open('placement_vis/points_before.obj', 'w') as f:
        for v in points:
            f.write("v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} \n"
                .format(v=v))
    points_1 = depth_filter(points,points_semantic,threshold=1,width=1)
    with open('placement_vis/points_after1.obj', 'w') as f:
        for v in points_1:
            f.write("v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} \n"
                .format(v=v))
    points_2 = depth_filter(points,points_semantic,threshold=0,width=1)
    with open('placement_vis/points_after2.obj', 'w') as f:
        for v in points_2:
            f.write("v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} \n"
                .format(v=v))
    # import pdb;pdb.set_trace()
    points_ = points.reshape(32,1100,3)
    for i in range(32):
        with open(f'placement_vis/points_{i}.obj', 'w') as f:
            for v in points_[i]:
                f.write("v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} \n"
                    .format(v=v))
    points_tmp = points_[15] # 1100,3
    points_roll = np.roll(points_tmp,1,axis = 0)
    car = points_semantic.reshape(32,-1)[15]==13
    dist = np.linalg.norm(points_tmp-points_roll,axis=-1,keepdims=True)
    with open(f'placement_vis/points_{15}_car.obj', 'w') as f:
            for v in points_[15][car]:
                f.write("v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} \n"
                    .format(v=v))