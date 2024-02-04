import os
import numpy as np


if __name__ == '__main__':
    root_path = '/SSD_DISK/users/zhangjunge/Neural_sim/Neural_sim/nuscenes-data_scripts/nuScenes_scenes/0100/'
    poses = np.load(root_path + 'poses_bounds.npy')
    file = open(root_path+'timesteps.txt','r')
    file = file.readlines()
    file = [f.split('\n')[0] for f in file]
    time = np.array(file).astype(np.int64)
    time = time - time.min()
    frames = len(poses)//6
    for s in range(6): # 6 sensors
        print('time continues')
        print(time[s*frames:s*frames + frames])
    import pdb;pdb.set_trace()
