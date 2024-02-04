# nodrop 0
# randomdrop 1
# learn drop 2
# vgg drop 3
# gt drop 4
import numpy as np
idx =15
#(idx =10 which is scene14，idx =14，75)
nodrop = 200+ idx
randomdrop = 300 +idx
learn_drop = 220 +idx
vgg_drop = 240 + idx

feature_drop = 320+ idx
datadirs =[];lidarrender_paths = []
import os;
os.system('rm -rf ray_drop_compare')
os.makedirs('ray_drop_compare')
path = os.listdir('/SSD_DISK/users/zhangjunge/simulation_data_v6')
scenes = [int(f.split('_')[0][5:]) for f in path]
print('this is scene {}'.format(scenes[idx]))
datadir = '6cam_scene{}_revision'.format(scenes[idx])
gt_lidar =os.path.join(datadir,'lidar_points','{:06d}.bin'.format(5))
# gt_drop = 1
filename = []
for num in [nodrop,randomdrop,learn_drop,vgg_drop,feature_drop]:
    filename += [os.path.join('/SSD_DISK/users/zhangjunge/semantickitti_nerf_city/sequences',str(num),'velodyne/000000.bin')]
Points = []
exp = ['nodrop','randomdrop','learn_drop','vgg_drop','feature_drop','gt_drop']
for f in filename:
    points = np.fromfile(f,dtype=np.float32).reshape(-1,3)
    labelfile = f.replace('velodyne', 'labels')[:-3] + 'label'
    label = np.fromfile(labelfile,dtype=np.uint32)
    mask = (label==13)|(label==14)|(label==15)
    points = points[mask]
    Points.append(points)

gt_lidar = np.fromfile(gt_lidar,dtype=np.float32).reshape(-1,5)[:,:3]
Points.append(gt_lidar[gt_lidar[:,2]>-1.75,:])
# Points.append(gt_lidar)


for i in range(len(Points)):
    temp_points=Points[i]
    with open('ray_drop_compare/drop_{}.obj'.format(exp[i]), 'w') as f:
        for v in temp_points:
            f.write("v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} \n"
                .format(v=v))