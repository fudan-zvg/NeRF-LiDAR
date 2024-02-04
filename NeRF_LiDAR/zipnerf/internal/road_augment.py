import numpy as np
import torch
import os

def batch_perturb(batch_aug,delta = 0.1):
    origins = batch_aug['origins']
    directions = batch_aug['directions']
    depths = batch_aug['depth']
    unit_dir = directions / (torch.linalg.norm(directions, dim=-1, keepdim=True) + 1e-8)
    target_pts = origins + depths[:,None] * unit_dir
    ptb_dir = torch.rand(origins.shape) 
    ptb_dir = ptb_dir/ (torch.linalg.norm(ptb_dir, dim=-1, keepdim=True)+ 1e-8) # unit vector
    origins_ptb = ptb_dir * delta + origins
    depths_ptb = torch.linalg.norm(target_pts - origins_ptb,dim=-1,keepdim=True)
    directions_ptb = (target_pts - origins_ptb) / depths_ptb
    # print('directions_ptb shape',directions_ptb.shape)
    # print('origins_ptb shape',origins_ptb.shape)
    # print('target_pts shape',target_pts.shape)
    # print('depths_ptb shape',depths_ptb.shape)
    batch_aug['origins'] = origins_ptb
    batch_aug['directions'] = directions_ptb
    batch_aug['depth'] = depths_ptb.squeeze(-1)
    batch_aug['aug_mask'][depths==0] = 1
    return batch_aug