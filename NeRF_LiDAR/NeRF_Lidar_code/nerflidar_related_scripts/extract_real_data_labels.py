import os
import numpy as np
# scenes = [1,3,5,14,16,23,29,33,38,52]
# simu_path = '/SSD_DISK/users/zhangjunge/simulation_data_v6'
# scenes = [int(f.split('_')[0][5:]) for f in os.listdir(simu_path)]
# scenes = [33,38,16,29,52,60]
scenes = [33,52,]

# import pdb;pdb.set_trace()
velo = [];labels=[]
for i in scenes:
    scene_name = '6cam_scene{}_revision'.format(i)
    
    velo += [os.path.join(scene_name,'sample_labels','velodyne','{:06d}.bin'.format(j)) for j in range(10)]
    labels += [os.path.join(scene_name,'sample_labels','labels','{:06d}.label'.format(j)) for j in range(10)]
os.makedirs('extract_real_samples',exist_ok=True)
os.makedirs('extract_real_samples/velodyne',exist_ok=True)
os.makedirs('extract_real_samples/labels',exist_ok=True)
# import pdb;pdb.set_trace()
for k in range(len(labels)):
    print('cp {} {}'.format(velo[k],os.path.join('extract_real_samples','velodyne','{:06d}.bin'.format(k))))
    os.system('cp {} {}'.format(velo[k],os.path.join('extract_real_samples','velodyne','{:06d}.bin'.format(k))))
    print('cp {} {}'.format(labels[k],os.path.join('extract_real_samples','labels','{:06d}.label'.format(k))))
    os.system('cp {} {}'.format(labels[k],os.path.join('extract_real_samples','labels','{:06d}.label'.format(k))))