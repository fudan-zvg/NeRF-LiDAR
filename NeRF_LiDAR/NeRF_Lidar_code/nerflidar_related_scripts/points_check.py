import numpy as np
import os
# points = np.fromfile('/SSD_DISK/users/zhangjunge/6cam_scene1_revision/lidar_points/000003.bin',dtype=np.float32).reshape(-1,3)
# points = np.load(os.path.join('/SSD_DISK/users/zhangjunge/Snerf/exp/lidar_rendering/scene7_city_v8_lidar_simulation','points_{:04d}.npy'.format(2)))
# points = np.load('/SSD_DISK/users/zhangjunge/6cam_scene1_revision/lidar_points/points046.npy')[:3,:].T
# points = np.load('/SSD_DISK/users/zhangjunge/Snerf/exp/lidar_rendering/scene7_city_v11_finetune_v2_lidar_render_1100/points_048.npy')
# points = np.load('/SSD_DISK/users/zhangjunge/Snerf/exp/lidar_rendering/scene7_city_v11_finetune_v2_lidar_render_1100/points_semantic_048.npy')

# # points = np.fromfile('/SSD_DISK/users/zhangjunge/semantickitti_nerf_city/sequences/12/velodyne/000917.bin',dtype=np.float32).reshape(-1,3)
# with open('v2_nerf_048.obj', 'w') as f:
#     for v in points:
#         f.write("v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} \n"
#             .format(v=v))
points = np.load('/SSD_DISK/users/zhangjunge/Snerf/exp/lidar_rendering/scene23_finetune_v1_lidar_render_1100/points_001.npy')
points = np.fromfile('/SSD_DISK/users/zhangjunge/6cam_scene11_revision/lidar_points/000000.bin',dtype=np.float32).reshape(-1,5)[:,:3]
points = np.load('/SSD_DISK/users/zhangjunge/Snerf/exp/lidar_rendering/scene11/scene11_finetune_v1_lidar_render_1100_vis/points_000.npy')
points = np.fromfile('/SSD_DISK/users/zhangjunge/6cam_scene29_revision/lidar_points/000000.bin',dtype=np.float32).reshape(-1,5)[:,:3]
with open('29_lidar.obj', 'w') as f:
    for v in points:
        f.write("v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} \n"
            .format(v=v))


# for i in range(50):
# #     points = np.load('/SSD_DISK/users/zhangjunge/Snerf/exp/lidar_rendering/scene7/scene7_finetune_v7_lidar_render_2000/points_{:03d}.npy'.format(i))
# #     semantics = np.load('/SSD_DISK/users/zhangjunge/Snerf/exp/lidar_rendering/scene7/scene7_finetune_v7_lidar_render_2000/points_semantic_{:03d}.npy'.format(i))

#     # points = np.load('/SSD_DISK/users/zhangjunge/Snerf/exp/lidar_rendering/scene7_city_v11_finetune_v2_lidar_render_2000/points_{:03d}.npy'.format(i))
#     # semantics = np.load('/SSD_DISK/users/zhangjunge/Snerf/exp/lidar_rendering/scene7_city_v11_finetune_v2_lidar_render_2000/points_semantic_{:03d}.npy'.format(i))
#     # points = points[semantics!=10,:]
#     points = np.load('/SSD_DISK/users/zhangjunge/Snerf/exp/lidar_rendering/scene3/scene3_finetune_v1_lidar_render_2000/points_{:03d}.npy'.format(i))
#     semantics = np.load('/SSD_DISK/users/zhangjunge/Snerf/exp/lidar_rendering/scene3/scene3_finetune_v1_lidar_render_2000/points_semantic_{:03d}.npy'.format(i))
#     points = points[semantics!=10,:]
#     # import pdb;pdb.set_trace()
#     with open('points_vis/scene3/points_{:03d}.obj'.format(i), 'w') as f:
#         for v in points:
#             f.write("v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} \n"
#                 .format(v=v))