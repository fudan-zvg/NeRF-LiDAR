import os
import torch
import car_nerf
from collections import OrderedDict
if __name__ == '__main__':
    car_nerf = car_nerf.CarNeRF_Field()
    ckpt = torch.load('/data1/junge/model_weights/step-000520000.ckpt')
    stat_dict = ckpt['pipeline']
    tgt_weights = OrderedDict({}) 
    for key in stat_dict:
        if 'object_models' in key and 'decoder' in key:
            replace_key = key.replace('_model.object_models.', '')
            tgt_weights[replace_key] = stat_dict[key]
    class_wise_n = 3
    for i in range(class_wise_n):
        class_dict = OrderedDict({}) 
        for key in tgt_weights:
            if f'object_class_{i}' in key:
                tmp_key = key.replace(f'object_class_{i}.fields.', '')
                class_dict[tmp_key] = tgt_weights[key]
        torch.save(class_dict,f'../checkpoints/class_{i}.ckpt')
    # car_nerf.load_state_dict(class_dict)
    import pdb;pdb.set_trace()