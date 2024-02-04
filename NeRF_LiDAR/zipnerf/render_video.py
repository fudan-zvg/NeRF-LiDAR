import glob
import os
import time

from absl import app
import gin
from internal import configs
from internal import datasets
from internal import models
from internal import train_utils
from internal import checkpoints
from internal import utils
from internal import obj_utils
from matplotlib import cm
import mediapy as media
import torch
import numpy as np
import accelerate
import imageio
from PIL import Image
from torch.utils._pytree import tree_map
import cv2

configs.define_common_flags()

num_class = 19
def def_color_map():
    s = 256**3//num_class
    colormap = [[(i*s)//(256**2),((i*s)//(256)%256),(i*s)%(256) ] for i in range(num_class)]
    return colormap

color_map = np.array(def_color_map())

def visualize_depth(depth, near=0.2, far=13):
    colormap = cm.get_cmap('turbo')
    curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
    eps = np.finfo(np.float32).eps
    near = near if near else depth.min()
    far = far if far else depth.max()
    near -= eps
    far += eps
    near, far, depth = [curve_fn(x) for x in [near, far, depth]]
    depth = np.nan_to_num(
        np.clip((depth - np.minimum(near, far)) / np.abs(far - near), 0, 1))
    vis = colormap(depth)[:, :, :3]
    # os.makedirs('./test_depths/'+pathname,exist_ok=True)
    out_depth = np.clip(np.nan_to_num(vis), 0., 1.) * 255
    return out_depth
def write_video(path_fn,RGBs,DEPs):
        fps = 10
        height, width,_  = RGBs[0].shape
        video = cv2.VideoWriter(path_fn('rgb_video.mp4'), cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
        video_dep = cv2.VideoWriter(path_fn('depth_video.mp4'), cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

        for idx,image in enumerate(RGBs):
            # RGB to BGR
            frame_rgb = (np.clip(np.nan_to_num(image), 0., 1.) * 255.).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            video.write(frame_bgr)
            frame_rgb = DEPs[idx].astype(np.uint8)
            # frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            video_dep.write(frame_rgb)

        video.release()
        video_dep.release()
def main(unused_argv):
    config = configs.load_config()
    config.exp_path = os.path.join('exp', config.exp_name)
    config.render_dir = os.path.join(config.exp_path, 'render')

    accelerator = accelerate.Accelerator()
    config.world_size = accelerator.num_processes
    config.local_rank = accelerator.local_process_index


    simu_mode = config.simu_mode if config.simu_mode!='' else 'replay'
    if simu_mode == 'ego_edit':
        config.ego_edit = True # edit ego track
    if not config.render_instance:
        # Image rendering
        dataset = datasets.load_dataset('video', config.data_dir, config)
    else:
        # Intance rendering
        dataset = datasets.load_dataset('instance', config.data_dir, config)

    if dataset.semantics is not None:
        pass
    else:
        config.use_semantic = False 

    utils.seed_everything(config.seed + accelerator.local_process_index)

    # edit bboxes info
    if config.num_insert > 0:
        # more tracks
        extra_tracks = np.load('{}'.format(config.insert_track)) # insert tracks
        dataset.bboxes = obj_utils.edit_tracks(dataset.bboxes,extra_tracks)
        print('editing the raw annotation')
    if config.latent_size > 0:
        # latent_vector_dict = train_utils.create_latent(dataset.bboxes+config.num_insert,config.latent_size)
        latent_vector_dict = train_utils.create_latent(dataset.bboxes,config.latent_size)
    else:
        latent_vector_dict = None
    
    
        
    model = models.Model(config=config,bboxes = dataset.bboxes,latent_vector_dict= latent_vector_dict)

    if accelerator.is_local_main_process:
        print(f'The simulation mode is {simu_mode}')
    angle, tracks = obj_utils.simu_info(simu_mode,model.tracks)
    # angle=0
    model.manipulate_bboxes(angle=angle)
    if config.ignore_spec:
        indx = [10,1,6,7,12]
        tracks[indx,:,4:7] *=0.0001 # ignore
    model.tracks = tracks

    # model.tracks[:,:,0] += 0.4
    # model.tracks[:,:,1] += 0.03
    

    # model.tracks = None
    # mv_distance = torch.linalg.norm(model.tracks[:,0,:3] - model.tracks[:,-1,:3],dim=-1)
    # model.tracks = torch.cat([model.tracks[...,:6],model.tracks[...,7:]],dim=-1) # kick out track 7
    # model.tracks[:,:,0] += 0.4
    # model.tracks[:,:,1] += 0.03
    step = checkpoints.restore_checkpoint(config.exp_path, model)
    model.training = False
    model.num_prop_samples = (256,64)
    accelerator.print(f'Rendering checkpoint at step {step}.')
    model.to(accelerator.device)


    dataloader = torch.utils.data.DataLoader(np.arange(len(dataset)),
                                             num_workers=0,
                                             shuffle=False,
                                             batch_size=1,
                                             collate_fn=dataset.collate_fn,
                                             )
    dataiter = iter(dataloader)
    
    if config.rawnerf_mode:
        postprocess_fn = dataset.metadata['postprocess_fn']
    else:
        postprocess_fn = lambda z: z

    # out_name = 'path_renders' if config.render_path else 'test_preds'
    # out_name = f'{out_name}_step_{step}'
    
    out_name = 'video'+f'_angle{angle}_step_{step}_{simu_mode}'
    
    if simu_mode == 'ego_edit':
        out_name = f'{out_name}_{config.shift_dist}'
    if config.num_insert >0:
        track_name = config.insert_track.split('/')[-1][:-4]
        out_name = out_name + '_{}'.format(track_name)
        print(f'Track used is {config.insert_track}')

    if config.render_instance:
        out_name = f'instance_vis_step_{step}'
    if config.ignore_spec:
        out_name = out_name+'_ignore_spec'
    
    out_dir = os.path.join(config.render_dir, out_name)
    if not utils.isdir(out_dir):
        utils.makedirs(out_dir)

    path_fn = lambda x: os.path.join(out_dir, x)

    # Ensure sufficient zero-padding of image indices in output filenames.
    zpad = max(3, len(str(dataset.size - 1)))
    idx_to_str = lambda idx: str(idx).zfill(zpad)
    

    # image level
    if not config.render_instance:
        RGBs= [] 
        DEPs = []
        for idx in range(dataset.size):
            # If current image and next image both already exist, skip ahead.
            idx_str = idx_to_str(idx)
            curr_file = path_fn(f'color_{idx_str}.png')
        
            batch = next(dataiter)
        
            # if utils.file_exists(curr_file):
            #     accelerator.print(f'Image {idx + 1}/{dataset.size} already exists, skipping')
            #     continue
            
            batch = tree_map(lambda x: x.to(accelerator.device) if x is not None else None, batch)
            accelerator.print(f'Evaluating image {idx + 1}/{dataset.size}')
            eval_start_time = time.time()

            rendering = models.render_image(model, accelerator,
                                            batch, False, config,)

            accelerator.print(f'Rendered in {(time.time() - eval_start_time):0.3f}s')

            if accelerator.is_local_main_process:  # Only record via host 0.

                # obj_mask = rendering['obj_mask']
                rendering['rgb'] = postprocess_fn(rendering['rgb'])
                rendering = tree_map(lambda x: x.detach().cpu().numpy() if x is not None else None, rendering)
                utils.save_img_u8(rendering['rgb'], path_fn(f'color_{idx_str}.png'))
                RGBs.append(rendering['rgb'])
                if 'normals' in rendering:
                    utils.save_img_u8(rendering['normals'] / 2. + 0.5,
                                    path_fn(f'normals_{idx_str}.png'))
                if config.use_semantic:
                    logits_2_label = lambda x: np.argmax(x, axis=-1)
                    labels = logits_2_label(rendering['semantic'])
                    labels_im = color_map[labels]
                    Image.fromarray(labels_im.astype(np.uint8)).save(path_fn(f'sem_{idx_str}.png'))
                if config.instance_obj:
                    obj_mask = rendering['obj_mask']

                    canvas = np.ones_like(labels_im)*255.
                    canvas[obj_mask]=labels_im[obj_mask]
                    Image.fromarray(canvas.astype(np.uint8)).save(path_fn(f'objvis_{idx_str}.png'))
                depth = rendering['depth']
                if config.use_semantic:
                    depth[labels == 10 ] = depth.max()
                dep = visualize_depth(depth)
                Image.fromarray(dep.astype(np.uint8)).save(path_fn(f'dep_{idx_str}.png'))
                DEPs.append(dep)
        if accelerator.is_local_main_process:
            write_video(path_fn,RGBs,DEPs)
    # instance level
    if config.render_instance:
        for instance_id in range(len(dataset.bboxes)):
            RGBs= [] 
            DEPs = []
            path_fn = lambda x: os.path.join(out_dir+f'_{instance_id}', x)
            for idx in range(dataset.size):
                # If current image and next image both already exist, skip ahead.
                idx_str = idx_to_str(idx)
                curr_file = path_fn(f'color_{idx_str}.png')
                batch = next(dataiter)
                # if utils.file_exists(curr_file):
                #     accelerator.print(f'Image {idx + 1}/{dataset.size} already exists, skipping')
                #     continue
                
                batch = tree_map(lambda x: x.to(accelerator.device) if x is not None else None, batch)
                accelerator.print(f'Evaluating image {idx + 1}/{dataset.size}')
                eval_start_time = time.time()

                rendering = models.render_image(model, accelerator,
                                                batch, False, config,render_instance=True,instance_id=instance_id)

                accelerator.print(f'Rendered in {(time.time() - eval_start_time):0.3f}s')

                if accelerator.is_local_main_process:  # Only record via host 0.
                    rendering['rgb'] = postprocess_fn(rendering['rgb'])
                    rendering = tree_map(lambda x: x.detach().cpu().numpy() if x is not None else None, rendering)
                    utils.save_img_u8(rendering['rgb'], path_fn(f'color_{idx_str}.png'))
                    RGBs.append(rendering['rgb'])
                    if 'normals' in rendering:
                        utils.save_img_u8(rendering['normals'] / 2. + 0.5,
                                        path_fn(f'normals_{idx_str}.png'))
                    if config.use_semantic:
                        logits_2_label = lambda x: np.argmax(x, axis=-1)
                        labels = logits_2_label(rendering['semantic'])
                        labels_im = color_map[labels]
                        Image.fromarray(labels_im.astype(np.uint8)).save(path_fn(f'sem_{idx_str}.png'))
                    # if config.instance_obj:
                    #     obj_mask = rendering['obj_mask']

                    #     canvas = np.ones_like(labels_im)*255.
                    #     canvas[obj_mask]=labels_im[obj_mask]
                    #     Image.fromarray(canvas.astype(np.uint8)).save(path_fn(f'objvis_{idx_str}.png'))
                    depth = rendering['depth']
                    if config.use_semantic:
                        depth[labels == 10 ] = depth.max() # sky vis
                    dep = visualize_depth(depth)
                    Image.fromarray(dep.astype(np.uint8)).save(path_fn(f'dep_{idx_str}.png'))
                    DEPs.append(dep)
            if accelerator.is_local_main_process:
                write_video(path_fn,RGBs,DEPs)
    
    


    accelerator.wait_for_everyone()


if __name__ == '__main__':
    with gin.config_scope('eval'):  # Use the same scope as eval.py
        app.run(main)
