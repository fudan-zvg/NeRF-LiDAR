import numpy as np
points_datapath = '/SSD_DISK/users/zhangjunge/6cam_scene52_revision/sample_labels'
import os
index =1
points = np.fromfile(os.path.join(points_datapath,'velodyne','{:06d}.bin'.format(index)),dtype = np.float32).reshape(-1,5)[:,:3]
points_label = np.fromfile(os.path.join(points_datapath,'labels','{:06d}.label'.format(index)),dtype = np.uint8).reshape(-1,1)
import yaml
with open('/SSD_DISK/users/zhangjunge/6cam_scene1_revision/nuscenes_label.yaml', 'r') as stream:
    semkittiyaml = yaml.safe_load(stream)
learning_map = semkittiyaml['learning_map']
points_label = np.vectorize(learning_map.__getitem__)(points_label)
with open('/SSD_DISK/users/zhangjunge/6cam_scene1_revision/NeRFlidar_label_v7_fortest.yaml', 'r') as stream:
    semkittiyaml = yaml.safe_load(stream)
learning_map = semkittiyaml['learning_map']
points_label = np.vectorize(learning_map.__getitem__)(points_label)
points_new = np.concatenate([points,points_label],axis = -1)
for i in range(0,5):
    with open('gt/gt_{}.obj'.format(i), 'w') as f:
        for v in points_new:
            if v[3]==i:
                f.write("v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} {v[3]:.8f}\n"
                    .format(v=v))
