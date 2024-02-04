import os
start =210
num =10
skip = 50
for i in range(start,start+num):
    os.makedirs('./simulation_points_vis/{:03d}'.format(i),exist_ok=True)
    os.makedirs('./simulation_points_vis/{:03d}/velodyne'.format(i),exist_ok=True)
    os.makedirs('./simulation_points_vis/{:03d}/labels'.format(i),exist_ok=True)
    for k in range(0,1000,skip):
        origin = '/SSD_DISK/users/zhangjunge/semantickitti_nerf_city/sequences/{:03d}/velodyne/{:06d}.bin'.format(i,k)
        target = './simulation_points_vis/{:03d}/velodyne/{:06d}.bin'.format(i,k//skip)
        os.system('cp {} {}'.format(origin,target))
        origin = '/SSD_DISK/users/zhangjunge/semantickitti_nerf_city/sequences/{:03d}/labels/{:06d}.label'.format(i,k)
        target = './simulation_points_vis/{:03d}/labels/{:06d}.label'.format(i,k//skip)
        os.system('cp {} {}'.format(origin,target))
    # os.makedirs('./simulation_points_vis/{:02d}'.format(i),exist_ok=True)
    # os.makedirs('./simulation_points_vis/{:02d}/velodyne'.format(i),exist_ok=True)
    # os.makedirs('./simulation_points_vis/{:02d}/labels'.format(i),exist_ok=True)
    # for k in range(0,1000,skip):
    #     origin = '/SSD_DISK/users/zhangjunge/semantickitti_nerf_city/sequences/{:02d}/velodyne/{:06d}.bin'.format(i,k)
    #     target = './simulation_points_vis/{:02d}/velodyne/{:06d}.bin'.format(i,k//skip)
    #     os.system('cp {} {}'.format(origin,target))
    #     origin = '/SSD_DISK/users/zhangjunge/semantickitti_nerf_city/sequences/{:02d}/labels/{:06d}.label'.format(i,k)
    #     target = './simulation_points_vis/{:02d}/labels/{:06d}.label'.format(i,k//skip)
    #     os.system('cp {} {}'.format(origin,target))