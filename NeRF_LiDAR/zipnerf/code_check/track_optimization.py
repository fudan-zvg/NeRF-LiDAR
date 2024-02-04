import os
import numpy as np
import torch


if __name__ == '__main__':
    # ckpt_instance = torch.load('/data1/junge/Unisim/zipnerf/exp/nuscenes_obj_instance/0237_front/verision32/tracknet_ckpt_40000.ckpt')

    # ckpt_latent = torch.load('/data1/junge/Unisim/zipnerf/exp/nuscenes_obj_instance_sym/0237_front/version38/tracknet_ckpt_40000.ckpt')
    ckpt_latent = torch.load('/data1/junge/Unisim/zipnerf/exp/nuscenes_obj_instance/0237_front/version37/tracknet_ckpt_40000.ckpt')


    import pdb;pdb.set_trace()
    ckpt_latent['state_dict'].keys()
    # 'opt_r', 'opt_t'
    dist =  torch.linalg.norm(ckpt_latent['state_dict']['opt_t'],dim=-1).max()
