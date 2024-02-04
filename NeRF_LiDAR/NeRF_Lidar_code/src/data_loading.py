import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, data_depends, transforms=None,scale: float = 1.0, mask_suffix: str = '',proj=False):
        super(Dataset, self).__init__()
        # self.images_dir = Path(images_dir)
        # self.masks_dir = Path(masks_dir)
       
        # self.imgs = np.load('raydata/rays.npy')
        # self.masks = np.load('raydata/rays_mask.npy')
        if not proj:
            imgs,masks,ranges = data_depends
        else:
            imgs,masks,ranges,proj_points,gt_proj_points = data_depends
            self.proj_points = proj_points
            self.gt_proj_points = gt_proj_points
        self.imgs = imgs
        self.masks = masks
        self.ranges = ranges
        self.transforms = transforms
        self.proj = proj
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        mask = self.masks[idx]
        range = self.ranges[idx]
        if not self.proj:
            if self.transforms is not None:
                return {
                'image': self.transforms(torch.as_tensor(img.copy()).float().contiguous()),
                'mask': self.transforms(torch.as_tensor(mask.copy()).long().contßiguous()),
                'range':self.transforms(torch.as_tensor(range.copy()).float.contiguous()),
            }
            return {
                'image': torch.as_tensor(img.copy()).float().contiguous(),
                'mask': torch.as_tensor(mask.copy()).long().contiguous(),
                'range':torch.as_tensor(range.copy()).float().contiguous(),
            }
        else:
            proj_points = self.proj_points[idx]
            gt_proj_points = self.gt_proj_points[idx]
            if self.transforms is not None:
                return {
                'image': self.transforms(torch.as_tensor(img.copy()).float().contiguous()),
                'mask': self.transforms(torch.as_tensor(mask.copy()).long().contiguous()),
                'range':self.transforms(torch.as_tensor(range.copy()).float.contiguous()),
                'proj_points':self.transforms(torch.as_tensor(proj_points.copy())).float().contiguous(),
                'gt_proj_points':self.transforms(torch.as_tensor(gt_proj_points.copy())).float().contiguous()
            }
            return {
                'image': torch.as_tensor(img.copy()).float().contiguous(),
                'mask': torch.as_tensor(mask.copy()).long().contiguous(),
                'range':torch.as_tensor(range.copy()).float().contiguous(),
                'proj_points':torch.as_tensor(proj_points.copy()).float().contiguous(),
                'gt_proj_points':torch.as_tensor(gt_proj_points.copy()).float().contiguous()
            }




class RayDropDataset(BasicDataset):
    def __init__(self, data_depends,transforms=None, scale=1,proj=False):
        super().__init__(data_depends, transforms=transforms,scale=1,proj=proj)
