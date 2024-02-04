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
from torch.utils._pytree import tree_map

configs.define_common_flags()

num_class = 19

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

def main(unused_argv):
    config = configs.load_config()
    config.exp_path = os.path.join('exp', config.exp_name)
    config.render_dir = os.path.join(config.exp_path, 'render')

    accelerator = accelerate.Accelerator()
    config.world_size = accelerator.num_processes
    config.local_rank = accelerator.local_process_index

    dataset = datasets.load_dataset('lidar', config.data_dir, config)
    if dataset.semantics is not None:
        config.use_semantic = True
    else:
        config.use_semantic = False

    utils.seed_everything(config.seed + accelerator.local_process_index)
    if config.latent_size > 0:
        latent_vector_dict = train_utils.create_latent(dataset.bboxes,config.latent_size)
    else:
        latent_vector_dict = None
    
    model = models.Model(config=config,bboxes = dataset.bboxes,latent_vector_dict= latent_vector_dict)

    simu_mode = config.simu_mode if config.simu_mode!='' else 'replay'
    if accelerator.is_local_main_process:
        print(f'The simulation mode is {simu_mode}')
    angle, tracks = obj_utils.simu_info(simu_mode,model.tracks)
    # angle=0
    model.manipulate_bboxes(angle=angle)
    model.tracks = tracks
    

    step = checkpoints.restore_checkpoint(config.exp_path, model)
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
    out_name = 'lidar_simulation'
    if config.simulation_mode == 'replay':
        out_name = 'lidar_replay'
    out_dir = os.path.join(config.render_dir, out_name)
    if not utils.isdir(out_dir):
        utils.makedirs(out_dir)

    path_fn = lambda x: os.path.join(out_dir, x)

    # Ensure sufficient zero-padding of image indices in output filenames.
    zpad = max(3, len(str(dataset.size - 1)))
    idx_to_str = lambda idx: str(idx).zfill(zpad)

    for idx in range(dataset.size):
        # If current image and next image both already exist, skip ahead.
        idx_str = idx_to_str(idx)
        # curr_file = path_fn(f'color_{idx_str}.png')
        # if utils.file_exists(curr_file):
        #     accelerator.print(f'Image {idx + 1}/{dataset.size} already exists, skipping')
        #     continue
        batch = next(dataiter)
        batch = tree_map(lambda x: x.to(accelerator.device) if x is not None else None, batch)

        accelerator.print(f'Evaluating image {idx + 1}/{dataset.size}')
        eval_start_time = time.time()
        # rendering = models.render_image(
        #     lambda rand, x: model(rand,
        #                           x,
        #                           train_frac=1.,
        #                           compute_extras=True,
        #                           sample_n=config.sample_n_test,
        #                           sample_m=config.sample_m_test,
        #                           ),
        #     accelerator,
        #     batch, False, config,image = False)
        rendering = models.render_image(model, accelerator,
                                        batch, False, config,image=False)

        accelerator.print(f'Rendered in {(time.time() - eval_start_time):0.3f}s')

        if accelerator.is_local_main_process:  # Only record via host 0.
            rendering['rgb'] = postprocess_fn(rendering['rgb'])
            rendering = tree_map(lambda x: x.detach().cpu().numpy() if x is not None else None, rendering)

            if 'normals' in rendering:
                utils.save_img_u8(rendering['normals'] / 2. + 0.5,
                                  path_fn(f'normals_{idx_str}.png'))
                

            def get_points_corrds(rays,depth):
                origins =rays[0]
                directions =rays[1]
                points = origins + depth[...,None]*directions#/directions[:,-2:-1]
                return points
            rays_o = batch['origins'].detach().cpu().numpy()
            rays_d = batch['directions'].detach().cpu().numpy()
            depth = rendering['depth'].reshape(-1)

            scale_factor = np.load(os.path.join(config.data_dir,'scene_scale.npy'))


            points = get_points_corrds((rays_o,rays_d),depth) / scale_factor

            np.save(path_fn('points_{:04d}.npy'.format(idx)),points)

            logits_2_label = lambda x: np.argmax(x, axis=-1)
            labels = logits_2_label(rendering['semantic'])
            
            np.save(path_fn('points_semantic_{:04d}.npy'.format(idx)),labels)
            np.save(path_fn('points_rgb_{:04d}.npy'.format(idx)),rendering['rgb'])


    accelerator.wait_for_everyone()


if __name__ == '__main__':
    with gin.config_scope('eval'):  # Use the same scope as eval.py
        app.run(main)
