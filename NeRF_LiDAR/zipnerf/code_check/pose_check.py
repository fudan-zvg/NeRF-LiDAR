import torch

from internal import checkpoints
from internal import posenet_v2

ckpt = '/SSD_DISK/users/zhangjunge/Neural_sim/zipnerf_sem/exp/nuscenes_debug/0100/debug2/posenet_ckpt_25000.ckpt'
# num_poses = 300
# t_ratio = 1.
# pose_net = posenet_v2.LearnPose(num_poses, t_ratio=t_ratio)
checkpoint = torch.load(ckpt)
import pdb;pdb.set_trace()