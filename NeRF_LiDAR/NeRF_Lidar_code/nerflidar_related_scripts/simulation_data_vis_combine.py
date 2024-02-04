import os
import numpy as np
# scenes = [1,3,5,14,16,23,29,33,38,52]
simu_path = '/SSD_DISK/users/zhangjunge/simulation_points_vis'
# scenes = [int(f.split('_')[0][5:]) for f in os.listdir(simu_path)]
start =210
num =10 
# import pdb;pdb.set_trace()
velo = [];labels=[]
scenes = [i for i in range(start,start+num)]
for i in scenes:
    # scene_name = '6cam_scene{}_revision'.format(i)
    scene_name = str(i)
    nums = len(os.listdir(os.path.join(simu_path,scene_name,'velodyne')))
    velo += [os.path.join(simu_path,scene_name,'velodyne','{:06d}.bin'.format(j)) for j in range(nums)]
    labels += [os.path.join(simu_path,scene_name,'labels','{:06d}.label'.format(j)) for j in range(nums)]
os.makedirs('simu_vis',exist_ok=True)
os.makedirs('simu_vis/velodyne',exist_ok=True)
os.makedirs('simu_vis/labels',exist_ok=True)
# import pdb;pdb.set_trace()
for k in range(len(labels)):
    print('cp {} {}'.format(velo[k],os.path.join('simu_vis','velodyne','{:06d}.bin'.format(k))))
    os.system('cp {} {}'.format(velo[k],os.path.join('simu_vis','velodyne','{:06d}.bin'.format(k))))
    print('cp {} {}'.format(labels[k],os.path.join('simu_vis','labels','{:06d}.label'.format(k))))
    os.system('cp {} {}'.format(labels[k],os.path.join('simu_vis','labels','{:06d}.label'.format(k))))