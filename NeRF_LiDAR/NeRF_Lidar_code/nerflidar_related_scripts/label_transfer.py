from multiprocessing.dummy import Process
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import camera_segmentation_utils

from waymo_open_dataset.protos import camera_segmentation_pb2 as cs_pb2

from waymo_open_dataset import dataset_pb2
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

## label in waymo
Label=['undefined','ego_vehicle','car','truck','bus','other_large_vehicle','bicycle',
       'motorcycle','trailer','pedestrian','cyclist','motorcyclist','bird','ground_animal',
       'construction_cone_pole','pole','pedestrian_object','sign','traffic_light','building',
       'road','lane_marker','road_marker','sidewalk','vegetation','sky','ground','dynamic','static'
       ]

import cv2
import matplotlib.pyplot as plt
# from mseg.utils.mask_utils_detectron2 import Visualizer
import mseg
import os
import numpy as np



## labels in mseg
imgs=[]
fpath = '/SSD_DISK/users/zhangjunge/mseg/mseg-semantic/temp_files/mseg-3m_images_universal_ss/1080/gray'

for i in range(450):
    pred_label_img=cv2.imread(os.path.join(fpath, '{:04d}.png'.format(i)),-1)
    imgs.append(pred_label_img)
labels=open('/SSD_DISK/users/zhangjunge/mseg/mseg-semantic/namespace.txt','r')
namespace=[]
namespace.append(labels.read().strip('\n'))
Name=namespace[0].split('\n')
imgs=np.stack(imgs,axis=0)
index=np.unique(imgs)
waymo_label=[];mseg_label=[]
for i in range(len(index)):
    
    if Name[index[i]] in Label:
        waymo_label.append(Name[index[i]])
    else:
        mseg_label.append(Name[index[i]])

print('waymo',waymo_label)
print('mseg',mseg_label)

"""
traffic_sign --- sign
person --- pedestrian
bicycilist --cyclist
bike_rack --- bicycle
road_barrier --- road_marker
sidewalk_pavement --- sidewalk
"""
extra_label = ['traffic_sign','person','bicycilist','road_barrier','sidewalk_pavement']
label_in_waymo = ['sign','pedestrian','cyclist','bicycle','road_marker','sidewalk']

def label_mapping(x):
    ## transfer label of mseg into waymo
    if Name[x] in waymo_label:
        return Label.index(Name[x])
    elif Name[x] in extra_label:
        return Label.index(label_in_waymo[extra_label.index(Name[x])])
    else:
        return 0
test_img = imgs[10]

from numba import njit

def circulation(matrix):
    h,w =matrix.shape
    result = np.zeros_like(matrix)
    for i in range(h):
        for j in range(w):
            result[i,j]=label_mapping(matrix[i,j])
            
    return result
# @njit
# def circulation(matrix,waymo_label,Label,label_in_waymo,extra_label):
#     h,w =matrix.shape
#     result = np.zeros_like(matrix)
#     for i in range(h):
#         for j in range(w):
#             name = matrix[i,j]
#             if name in waymo_label:
#                 result[i,j]=Label.index(name)
#             elif name in extra_label:
#                 result[i,j]=Label.index(label_in_waymo[extra_label.index(name)])
#             else:
#                result[i,j]= 0
            
            
#     return result
import pandas as pd
import time
import multiprocessing
print()
t1=time.time()
test=multiprocessing.Pool(processes=32)
result=test.map(np.vectorize(label_mapping),imgs)
test.close()
test.join()
t2=time.time()
print('Done')
print('time',t2-t1)

# t1=time.time()
# data=pd.DataFrame(test_img)
# t2=time.time()
# print('time: ',t2-t1)
# data=data.applymap(label_mapping)
# t3=time.time()
# print('time: ',t3-t2)
# output = data.values
# t4=time.time()
# print('time: ',t4-t3)

# t1=time.time()
# # new_img=circulation(test_img,waymo_label,Label,label_in_waymo,extra_label)
# t2=time.time()
# print('time2: ',t2-t1)
# t1=time.time()
# # output=np.array(list(map(label_mapping,test_img)))
# output = np.vectorize(label_mapping)(test_img)
# t2=time.time()
# print('time2: ',t2-t1)
import pdb;pdb.set_trace()
