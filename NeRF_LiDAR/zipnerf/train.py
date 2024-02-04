import os
import numpy as np
import random

import time

from absl import app
import gin
from internal import configs
from internal import datasets
from internal import image
from internal import models
from internal import train_utils
from internal import utils
from internal import vis
from internal import checkpoints


import torch
import accelerate
import tensorboardX
from tqdm import tqdm
from torch.utils._pytree import tree_map
import torch.nn as nn

configs.define_common_flags()

TIME_PRECISION = 1000  # Internally represent integer times in milliseconds.


def main(unused_argv):
    config = configs.load_config()


    # # cyr set lr
    # config.lr_init = 0.00125*config.gpu_num
    # config.lr_final = config.lr_init/10


    config.exp_path = os.path.join("exp", config.exp_name)
    utils.makedirs(config.exp_path)
    with utils.open_file(os.path.join(config.exp_path, 'config.gin'), 'w') as f:
        f.write(gin.config_str())
    utils.makedirs(os.path.join(config.exp_path, 'code'))
    utils.makedirs(os.path.join(config.exp_path, 'code', 'internal'))
    os.system('cp -r internal/*.py {}'.format(os.path.join(config.exp_path,'code','internal')))
    os.system('cp -r *.py {}'.format(os.path.join(config.exp_path, 'code')))


    # accelerator for DDP

    ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs])
    # accelerator = accelerate.Accelerator()
    config.world_size = accelerator.num_processes
    config.local_rank = accelerator.local_process_index
    if config.batch_size % accelerator.num_processes != 0:
        config.batch_size -= config.batch_size % accelerator.num_processes != 0
        accelerator.print('turn batch size to', config.batch_size)
    # Shift random seed by local_process_index to shuffle data loaded by different hosts.
    utils.seed_everything(config.seed + accelerator.local_process_index)

    # load dataset
    dataset = datasets.load_dataset('train', config.data_dir, config)
    
    config.factor = 8  # for efficiency downsample 8x for test image while training
    test_dataset = datasets.load_dataset('test', config.data_dir, config)

   

    # pose_net.to(device)
    # pose_net.train()
    # pose_net_lr = 2e-5

    if dataset.semantics is not None:
        # config.use_semantic = True 
        pass 
    else:
        config.use_semantic = False

    # prepare latetn vector 
    if config.latent_size > 0:
        latent_vector_dict = train_utils.create_latent(dataset.bboxes,config.latent_size)
    else:
        latent_vector_dict = None

    # setup model and optimizer
    model = models.Model(config=config,bboxes = dataset.bboxes,latent_vector_dict= latent_vector_dict,training=True)

    optimizer, lr_fn = train_utils.create_optimizer(config, model)
    init_step = checkpoints.restore_checkpoint(config.exp_path, model, optimizer) + 1

    ### pose net init
    num_poses = dataset.num_poses
    posenet, pn_optimizer, pn_lr_fn = train_utils.create_posenet(num_poses, config,num_lidars= dataset.num_lidars)
    _ = checkpoints.restore_checkpoint(config.exp_path, posenet, pn_optimizer, prefix='posenet_ckpt_')
    posenet, pn_optimizer = accelerator.prepare(posenet, pn_optimizer)

    ### track net init
    tracknet, tn_optimizer, tn_lr_fn = train_utils.create_tracknet(config,model.tracks)
    _ = checkpoints.restore_checkpoint(config.exp_path, tracknet, pn_optimizer, prefix='tracknet_ckpt_')
    tracknet, tn_optimizer = accelerator.prepare(tracknet, tn_optimizer)
    tracknet_module = accelerator.unwrap_model(tracknet)
    
    
    start_step, end_step = config.start_step, config.end_step # set for different network
     
    # model = torch.compile(model)  # not work yet

    # load dataset
    dataloader = torch.utils.data.DataLoader(np.arange(len(dataset)),
                                             num_workers=8,  # ori 8
                                             shuffle=True,
                                             batch_size=1,
                                             collate_fn=dataset.collate_fn,
                                             persistent_workers=True,
                                             pin_memory=True,
                                             )
    test_dataloader = torch.utils.data.DataLoader(np.arange(len(test_dataset)),
                                                  num_workers=0,
                                                  shuffle=False,
                                                  batch_size=1,
                                                  collate_fn=test_dataset.collate_fn,
                                                  )
    if config.rawnerf_mode:
        postprocess_fn = test_dataset.metadata['postprocess_fn']
    else:
        postprocess_fn = lambda z, _=None: z

    # use accelerate to prepare.
    # no need to prepare dataloader because data in each process are loaded differently
    model, optimizer = accelerator.prepare(model, optimizer)
    module = accelerator.unwrap_model(model)
    dataiter = iter(dataloader)
    test_dataiter = iter(test_dataloader)


    num_params = train_utils.tree_len(list(model.parameters()))

    accelerator.print(f'Number of parameters being optimized: {num_params}')

    if (dataset.size > module.num_glo_embeddings and module.num_glo_features > 0):
        raise ValueError(f'Number of glo embeddings {module.num_glo_embeddings} '
                         f'must be at least equal to number of train images '
                         f'{dataset.size}')

    # metric handler
    metric_harness = image.MetricHarness()

    # tensorboard
    if accelerator.is_local_main_process:
        summary_writer = tensorboardX.SummaryWriter(config.exp_path)
        # function to convert image for tensorboard
        tb_process_fn = lambda x: x.transpose(2, 0, 1) if len(x.shape) == 3 else x[None]

        if config.rawnerf_mode:
            for name, data in zip(['train', 'test'], [dataset, test_dataset]):
                # Log shutter speed metadata in TensorBoard for debug purposes.
                for key in ['exposure_idx', 'exposure_values', 'unique_shutters']:
                    summary_writer.add_text(f'{name}_{key}', str(data.metadata[key]), 0)
    accelerator.print("Begin training...")
    total_time = 0
    total_steps = 0
    reset_stats = True
    if config.early_exit_steps is not None:
        num_steps = config.early_exit_steps
    else:
        num_steps = config.max_steps

    if accelerator.is_local_main_process:
        tbar = tqdm(range(init_step, num_steps + 1))
    else:
        tbar = range(init_step, num_steps + 1)
    for step in tbar:
        try:
            batch = next(dataiter)
        except StopIteration:
            dataiter = iter(dataloader)
            batch = next(dataiter)
        batch = accelerate.utils.send_to_device(batch, accelerator.device)

        if reset_stats and accelerator.is_local_main_process:
            stats_buffer = []
            train_start_time = time.time()
            reset_stats = False

        # use lr_fn to control learning rate
        learning_rate = lr_fn(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        # fraction of training period
        train_frac = np.clip((step - 1) / (config.max_steps - 1), 0, 1)

        # Indicates whether we need to compute output normal or depth maps in 2D.
        compute_extras = (config.compute_disp_metrics or config.compute_normal_metrics)
        optimizer.zero_grad()

        ## forward posenet
        if config.pose_refine:
            if step > start_step and step < end_step:
                pn_lr = pn_lr_fn(step)  # in the pn_lr_fn, it will minus start_step
                for param_group in pn_optimizer.param_groups:
                    param_group['lr'] = pn_lr
                pn_optimizer.zero_grad()

                # bpp = batch['origins'].shape[:3]
                bpp = batch['origins'].shape[0:1]
                refine_poses = posenet(batch['glo_idx'].reshape(-1).int())  # b 4 4
                # trans
                batch['origins'] = batch['origins'] + refine_poses[:, :3, 3].reshape(*bpp,3)
                # batch['directions'] = (batch['directions'].reshape(-1,1,3) * refine_poses[:, :3, :3]).sum(-1).reshape(*bpp,3)
                # batch['viewdirs'] = (batch['viewdirs'].reshape(-1,1,3) * refine_poses[:, :3, :3]).sum(-1).reshape(*bpp,3)
                # batch['base_x'] = (batch['base_x'].reshape(-1,1,3)  * refine_poses[:, :3, :3]).sum(-1).reshape(*bpp,3)
                # batch['base_y'] = (batch['base_y'].reshape(-1,1,3)  * refine_poses[:, :3, :3]).sum(-1).reshape(*bpp,3)

                # rot
                for key in ['directions','viewdirs','base_x','base_y']:
                    batch[key] = (batch[key].reshape(-1,1,3)  * refine_poses[:, :3, :3]).sum(-1).reshape(*bpp,3)
                if 'normals' in batch:
                    batch['normals'] = (batch['normals'].reshape(-1,1,3) * refine_poses[:, :3, :3]).sum(-1).reshape(*bpp,3)

            elif step < start_step:
                pass
            elif step > end_step:
                with torch.no_grad():
                    bpp = batch['origins'].shape[0:1]
                    refine_poses = posenet(batch['glo_idx'].reshape(-1).int())  # b 4 4
                    # trans
                    batch['origins'] = batch['origins'] + refine_poses[:, :3, 3].reshape(*bpp, 3)
                    # rot
                    for key in ['directions','viewdirs','base_x','base_y']:
                        batch[key] = (batch[key].reshape(-1,1,3)  * refine_poses[:, :3, :3]).sum(-1).reshape(*bpp,3)
                    # batch['directions'] = (batch['directions'].reshape(-1, 1, 3) * refine_poses[:, :3, :3]).sum(
                    #     -1).reshape(*bpp, 3)
                    # batch['viewdirs'] = (batch['viewdirs'].reshape(-1, 1, 3) * refine_poses[:, :3, :3]).sum(-1).reshape(
                    #     *bpp, 3)
                    # batch['base_x'] = (batch['base_x'].reshape(-1, 1, 3) * refine_poses[:, :3, :3]).sum(-1).reshape(
                    #     *bpp, 3)
                    # batch['base_y'] = (batch['base_y'].reshape(-1, 1, 3) * refine_poses[:, :3, :3]).sum(-1).reshape(
                    #     *bpp, 3)
                    if 'normals' in batch:
                        batch['normals'] = (batch['normals'].reshape(-1,1,3) * refine_poses[:, :3, :3]).sum(-1).reshape(*bpp,3)
        if config.track_refine:
            if step>config.track_start_opt and step < config.track_start_opt + 5000:
                tn_lr = tn_lr_fn(step)  # in the pn_lr_fn, it will minus start_step
                for param_group in tn_optimizer.param_groups:
                    param_group['lr'] = tn_lr
                tn_optimizer.zero_grad()

                refine_r = tracknet_module.opt_r.to(module.tracks.device)
                refine_t = tracknet_module.opt_t.to(module.tracks.device)
                raw_track = tracknet_module.tracks.to(module.tracks.device)
                track = raw_track.clone()
                track[:,:,:3] = raw_track[:,:,:3] + refine_t
                track[:,:,3:4] = raw_track[:,:,3:4] + refine_r
            elif step > config.track_start_opt + 5000:
                with torch.no_grad():
                    refine_r = tracknet_module.opt_r.to(module.tracks.device)
                    refine_t = tracknet_module.opt_t.to(module.tracks.device)
                    raw_track = tracknet_module.tracks.to(module.tracks.device)
                    track = raw_track.clone()
                    track[:,:,:3] = raw_track[:,:,:3] + refine_t
                    track[:,:,3:4] = raw_track[:,:,3:4] + refine_r
            else:
                track = None
        else:
            track = None
        with accelerator.autocast():
            # model.scale_sample_points(step, config.max_steps)
            compute_extras = True
            renderings, ray_history = model(
                True,
                batch,
                train_frac=train_frac,
                compute_extras=compute_extras,
                sample_n=config.sample_n_train,
                sample_m=config.sample_m_train,
                zero_glo=False,
                step=step, max_step=config.max_steps,
                curr_track = track)

        losses = {}

        
        if config.dataset_loader == 'nusc':
            batch['mask'] = batch['mask']==0 # only apply loss on mask == 0 
            if config.instance_obj:
                batch['mask'] = torch.zeros_like(batch['mask']) # No mask anymore
            if config.aug_road:
                batch['mask'][batch['aug_mask']==1] = 1 # not on aug_mask
        # supervised by data
        # patch_mask = torch.zeros_like(batch['mask'], device=batch['mask'].device)
        # if config.patch_size>1: # use patch sample
        #     num_patch = batch['mask'].shape[0]//4//config.patch_size**2
        #     patch_mask[:num_patch*config.patch_size**2] = 1
        # else:
        #     num_patch = 0
        
        # patch_mask = torch.zeros_like(batch['mask'], device=batch['mask'].device)
        patch_mask = batch['patch_mask']
        if config.patch_size>1: # use patch sample
            num_patch = (batch['patch_mask']==1).sum()//config.patch_size**2
            # patch_mask[:num_patch*config.patch_size**2] = 1
        else:
            num_patch = 0
        
        # rgb_mask

        rgb_mask = torch.logical_and(batch['mask'] == 0, patch_mask == 0)
        # depth_mask
        depth_mask = torch.logical_and(batch['depth']>0, rgb_mask)
        # sem_mask
        sem_mask = torch.logical_and(batch['semantic']!=255, rgb_mask)
        
        if config.lidar_supervision:
            # only depth loss
            rgb_mask[batch['lidar_mask']==1] = 0
            depth_mask[batch['lidar_mask']==1] = 1
            sem_mask[batch['lidar_mask']==1] = 0
            if config.only_lidar_supervison:
                depth_mask[batch['lidar_mask']==0] = 0
        # Loss
        batch['mask_rgb'] = rgb_mask

        data_loss, stats = train_utils.compute_data_loss(batch, renderings, config)

        losses['data'] = data_loss

        if config.depth_loss and len(os.listdir(os.path.join(config.data_dir,'depth'))) > 0:
            dep_lam = 0.4
            # dep_lam = max(0, dep_lam*(1-step/(0.8*num_steps)))
            dep_lam = 0 if step>start_step and step< int (0.6*end_step) and config.pose_refine else (dep_lam if step> end_step else 0.1)
            # dep_dist = 1/(renderings[-1]['depth'][depth_mask]+1e-5)-1/(1e-5+batch['depth'][depth_mask]) 
            dep_dist = renderings[-1]['depth'][depth_mask] - batch['depth'][depth_mask]

            # loss_dep_new = torch.abs(dep_dist.mean()).clamp_min(0.05)+0.1*dep_dist.std().clamp_min(0.05)
            depth_thre = torch.quantile(torch.abs(dep_dist),0.9)
            loss_dep_new = torch.log(torch.abs(dep_dist[dep_dist < depth_thre])+1).mean()
            losses['depth'] = dep_lam*loss_dep_new

        # loss_dep = torch.abs(dep_dist).mean()
        # losses['depth'] = dep_lam*loss_dep


        # err = 0.05
        # tmids = (ray_history[-1]['tdist'][...,:-1]+ray_history[-1]['tdist'][...,1:])/2
        # # t_itv = ray_history[-1]['tdist'][...,1:] - ray_history[-1]['tdist'][...,:-1]
        # # loss_dep_kl = -torch.log(ray_history[-1]['weights'])*torch.exp(-(tmids-batch['depth'][...,None])**2/(2*err))*t_itv
        # # loss_dep_kl = loss_dep_kl[depth_mask].sum(-1).mean()
        # # losses['depth'] = loss_dep_kl
        #
        # loss_dep_self = (torch.log(tmids+1e-3)-torch.log(renderings[-1]['depth']+1e-3)[...,None])**2/(2*err)*ray_history[-1]['weights']
        # loss_dep_self = loss_dep_self.sum(-1).mean()
        # losses['sl_dep'] = 0.001*loss_dep_self


        if config.normal_supervision:
            normal_mask = torch.logical_and(rgb_mask,batch['semantic']!=10) # not apply on sky
            pred_normals = renderings[-1]['normals'][normal_mask] # align with rgb image
            pseudo_normals = batch['normals'][normal_mask] 
            normal_loss = (pred_normals - pseudo_normals).abs().sum(dim = -1) + (1 -torch.sum(pred_normals*pseudo_normals,dim = -1))
            losses['normals'] = normal_loss.mean() * 0.1

        # smoothness loss cyr
        if num_patch>0:
            smo_lam = 0.01
            
            norm_lam = 0.01
            patch_size = config.patch_size
            shape = [num_patch, patch_size, patch_size]
            
            # mask_patch = batch['mask'][:num_patch*patch_size**2].reshape(*shape)
            mask_patch = batch['mask'][patch_mask==1].reshape(*shape)
            mask_patch = torch.where(mask_patch > 0, 0, 1)
            # dep_patch = renderings[-1]['depth'][:num_patch*patch_size**2].reshape(*shape, -1)
            dep_patch = renderings[-1]['depth'][patch_mask==1].reshape(*shape, -1)
            rgb_patch = batch['rgb'][patch_mask==1].reshape(*shape, -1)
            
            # smoothness_loss = train_utils.edge_aware_loss_v2(rgb_patch, 1/(dep_patch+1e-5), mask=mask_patch)
            smoothness_loss = train_utils.edge_aware_loss_v2(rgb_patch, dep_patch, mask=mask_patch)
            
            losses['d_smo'] = torch.nan_to_num(smo_lam * smoothness_loss)
            if config.use_semantic:
                sem_patch = renderings[-1]['semantic'][patch_mask==1].reshape(*shape, -1)
                sem_smooth = train_utils.edge_aware_loss_for_semantic(rgb_patch, sem_patch, mask=mask_patch)
                s_lam = 0.01
                losses['s_smo'] = torch.nan_to_num(s_lam * sem_smooth)
            if 'normals' in renderings[-1]:
                normal_patch = renderings[-1]['normals'][patch_mask==1].reshape(*shape, -1)
                norm_smooth = train_utils.edge_aware_loss_v2_norm(rgb_patch, normal_patch, mask=mask_patch)
                losses['n_smo'] = torch.nan_to_num(norm_lam * norm_smooth)

        # latent regularization in NSG
        if config.latent_size > 0:
            # reg = 1/config.latent_balance    # 1/0.001
            reg = 1/config.latent_reg    # 1/0.001
            latent_reg = train_utils.latentReg(list(latent_vector_dict.values()), reg)
            losses['latent_reg'] = latent_reg
        ## semantic loss cyr
        if config.use_semantic:
            # sem_mask = rgb_mask
            nllloss = nn.NLLLoss()
            sem_lam = 0.04
            sem_lam = 0 if step>start_step and step< int (0.6*end_step) and config.pose_refine else (sem_lam if step> end_step else 0.01)
            # renderings[-1]['semantic'] = torch.softmax(renderings[-1]['semantic'], -1)
            if sem_mask.sum()>0:
                loss_sem = nllloss(torch.log(renderings[-1]['semantic'][sem_mask]+1e-6), batch['semantic'][sem_mask].long())
            else:
                loss_sem = torch.tensor(0.)
            losses['sem'] = loss_sem*sem_lam  # ori 0.04
        ## lidar intensity loss 
        if config.use_intensity:
            pred_intensity = renderings[-1]['intensity'].reshape(-1)
            target_intensity = batch['intensity'].reshape(-1)
            intensity_mask = batch['lidar_mask']==1 # only apply on lidar rays
            loss_intensity = (pred_intensity - target_intensity)[intensity_mask].pow(2).mean()
            losses['int'] = loss_intensity * 0.1
        # interlevel loss in MipNeRF360
        if config.interlevel_loss_mult > 0 and not module.single_mlp:
            losses['interlevel'] = train_utils.interlevel_loss(ray_history, config)

        # interlevel loss in ZipNeRF360
        
        if config.anti_interlevel_loss_mult > 0 and not module.single_mlp:
            losses['interlevel'] = train_utils.anti_interlevel_loss(ray_history, config)

        # distortion loss
        if config.distortion_loss_mult > 0:
            losses['distortion'] = train_utils.distortion_loss(ray_history, config)
        
        # symmetry loss
        if config.symmetrize and step >config.sym_start:
            losses['sym'] = renderings[-1]['loss_sym']
        # orientation loss in RefNeRF
        if (config.orientation_coarse_loss_mult > 0 or
                config.orientation_loss_mult > 0):
            losses['orientation'] = train_utils.orientation_loss(batch, module, ray_history,
                                                                 config)
        # hash grid l2 weight decay
        if config.hash_decay_mults > 0:
            if 'hash_decay' in renderings[-1]:
                # precompute hash decay loss
                losses['hash_decay'] = renderings[-1]['hash_decay']
            else:
                losses['hash_decay'] = train_utils.hash_decay_loss(model, config)

        # normal supervision loss in RefNeRF
        if (config.predicted_normal_coarse_loss_mult > 0 or
                config.predicted_normal_loss_mult > 0):
            losses['predicted_normals'] = train_utils.predicted_normal_loss(
                model, ray_history, config)
        loss = sum(losses.values())
        stats['loss'] = loss.item()
        stats['losses'] = tree_map(lambda x: x.item(), losses)

        # accelerator automatically handle the scale
        # accelerator.backward(loss,retain_graph = True)
        accelerator.backward(loss)
        # clip gradient by max/norm/nan
        train_utils.clip_gradients(model, accelerator, config)
        optimizer.step()
        # posenet step
        if step>start_step and step<end_step and config.pose_refine:
            train_utils.clip_gradients(posenet, accelerator, config)
            pn_optimizer.step()
        # tracknet step
        if step>config.track_start_opt and step < config.track_start_opt + 5000 and config.track_refine:
            train_utils.clip_gradients(tracknet, accelerator, config)
            # update params
            tn_optimizer.step()


        stats['psnrs'] = image.mse_to_psnr(stats['mses'])
        stats['psnr'] = stats['psnrs'][-1]

        # Log training summaries. This is put behind a host_id check because in
        # multi-host evaluation, all hosts need to run inference even though we
        # only use host 0 to record results.
        if accelerator.is_local_main_process:
            stats_buffer.append(stats)
            if step == init_step or step % config.print_every == 0:
                elapsed_time = time.time() - train_start_time
                steps_per_sec = config.print_every / elapsed_time
                rays_per_sec = config.batch_size * steps_per_sec

                # A robust approximation of total training time, in case of pre-emption.
                total_time += int(round(TIME_PRECISION * elapsed_time))
                total_steps += config.print_every
                approx_total_time = int(round(step * total_time / total_steps))

                # Transpose and stack stats_buffer along axis 0.

                fs = [utils.flatten_dict(s, sep='/') for s in stats_buffer]
                stats_stacked = {k: np.stack([f[k] for f in fs]) for k in fs[0].keys()}

                # Split every statistic that isn't a vector into a set of statistics.
                stats_split = {}
                for k, v in stats_stacked.items():
                    if v.ndim not in [1, 2] and v.shape[0] != len(stats_buffer):
                        raise ValueError('statistics must be of size [n], or [n, k].')
                    if v.ndim == 1:
                        stats_split[k] = v
                    elif v.ndim == 2:
                        for i, vi in enumerate(tuple(v.T)):
                            stats_split[f'{k}/{i}'] = vi

                # Summarize the entire histogram of each statistic.
                # for k, v in stats_split.items():
                #     summary_writer.add_histogram('train_' + k, v, step)

                # Take the mean and max of each statistic since the last summary.
                avg_stats = {k: np.mean(v) for k, v in stats_split.items()}
                max_stats = {k: np.max(v) for k, v in stats_split.items()}

                summ_fn = lambda s, v: summary_writer.add_scalar(s, v, step)  # pylint:disable=cell-var-from-loop

                # Summarize the mean and max of each statistic.
                for k, v in avg_stats.items():
                    summ_fn(f'train_avg_{k}', v)
                for k, v in max_stats.items():
                    summ_fn(f'train_max_{k}', v)

                summ_fn('train_num_params', num_params)
                summ_fn('train_learning_rate', learning_rate)
                summ_fn('train_steps_per_sec', steps_per_sec)
                summ_fn('train_rays_per_sec', rays_per_sec)

                summary_writer.add_scalar('train_avg_psnr_timed', avg_stats['psnr'],
                                          total_time // TIME_PRECISION)
                summary_writer.add_scalar('train_avg_psnr_timed_approx', avg_stats['psnr'],
                                          approx_total_time // TIME_PRECISION)

                if dataset.metadata is not None and module.learned_exposure_scaling:
                    scalings = module.exposure_scaling_offsets.weight
                    num_shutter_speeds = dataset.metadata['unique_shutters'].shape[0]
                    for i_s in range(num_shutter_speeds):
                        for j_s, value in enumerate(scalings[i_s]):
                            summary_name = f'exposure/scaling_{i_s}_{j_s}'
                            summary_writer.add_scalar(summary_name, value, step)

                precision = int(np.ceil(np.log10(config.max_steps))) + 1
                avg_loss = avg_stats['loss']
                avg_psnr = avg_stats['psnr']
                str_losses = {  # Grab each "losses_{x}" field and print it as "x[:4]".
                    k[7:11]: (f'{v:0.5f}' if v >= 1e-4 and v < 10 else f'{v:0.1e}')
                    for k, v in avg_stats.items()
                    if k.startswith('losses/')
                }
                tbar.write(f'{step:{precision}d}' + f'/{config.max_steps:d}: ' +
                           f'loss={avg_loss:0.5f}, ' + f'psnr={avg_psnr:6.3f}, ' +
                           f'lr={learning_rate:0.2e} | ' +
                           ', '.join([f'{k}={s}' for k, s in str_losses.items()]) +
                           f', {rays_per_sec:0.0f} r/s')

                # Reset everything we are tracking between summarizations.
                reset_stats = True

            if step > 0 and step % config.checkpoint_every == 0:
                checkpoints.save_checkpoint(
                    config.exp_path,
                    {
                        'state_dict': accelerator.unwrap_model(model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                    },
                    int(step), keep=1)
                if config.pose_refine:
                    checkpoints.save_checkpoint(
                        config.exp_path,
                        {
                            'state_dict': accelerator.unwrap_model(posenet).state_dict(),
                            'optimizer': pn_optimizer.state_dict(),
                        },
                        int(step), keep=1, prefix='posenet_ckpt_')
                if config.track_refine:
                    checkpoints.save_checkpoint(
                        config.exp_path,
                        {
                            'state_dict': accelerator.unwrap_model(tracknet).state_dict(),
                            'optimizer': tn_optimizer.state_dict(),
                        },
                        int(step), keep=1, prefix='tracknet_ckpt_')

        # Test-set evaluation.
        if config.train_render_every > 0 and step % config.train_render_every == 0:
            # We reuse the same random number generator from the optimization step
            # here on purpose so that the visualization matches what happened in
            # training.
            eval_start_time = time.time()
            try:
                test_batch = next(test_dataiter)
            except StopIteration:
                test_dataiter = iter(test_dataloader)
                test_batch = next(test_dataiter)
            test_batch = accelerate.utils.send_to_device(test_batch, accelerator.device)

            # render a single image with all distributed processes
            # rendering = models.render_image(
            #     lambda rand, x: model(rand,
            #                           x,
            #                           train_frac=train_frac,
            #                           compute_extras=True,
            #                           sample_n=config.sample_n_test,
            #                           sample_m=config.sample_m_test,
            #                           ),
            #     accelerator,
            #     test_batch, False, config)
            rendering = models.render_image(model, accelerator,
                                        test_batch, False, config,train_frac)

            # move to numpy
            rendering = tree_map(lambda x: x.detach().cpu().numpy(), rendering)
            test_batch = tree_map(lambda x: x.detach().cpu().numpy() if x is not None else None, test_batch)
            # Log eval summaries on host 0.
            if accelerator.is_local_main_process:
                eval_time = time.time() - eval_start_time
                num_rays = np.prod(test_batch['directions'].shape[:-1])
                rays_per_sec = num_rays / eval_time
                summary_writer.add_scalar('test_rays_per_sec', rays_per_sec, step)
                print(f'Eval {step}: {eval_time:0.3f}s., {rays_per_sec:0.0f} rays/sec')

                metric_start_time = time.time()
                metric = metric_harness(
                    postprocess_fn(rendering['rgb']), postprocess_fn(test_batch['rgb']))
                print(f'Metrics computed in {(time.time() - metric_start_time):0.3f}s')
                for name, val in metric.items():
                    if not np.isnan(val):
                        print(f'{name} = {val:.4f}')
                        summary_writer.add_scalar('train_metrics/' + name, val, step)

                if config.vis_decimate > 1:
                    d = config.vis_decimate
                    decimate_fn = lambda x, d=d: None if x is None else x[::d, ::d]
                else:
                    decimate_fn = lambda x: x
                rendering = tree_map(decimate_fn, rendering)
                test_batch = tree_map(decimate_fn, test_batch)
                vis_start_time = time.time()
                vis_suite = vis.visualize_suite(rendering, test_batch)
                print(f'Visualized in {(time.time() - vis_start_time):0.3f}s')
                if config.rawnerf_mode:
                    # Unprocess raw output.
                    vis_suite['color_raw'] = rendering['rgb']
                    # Autoexposed colors.
                    vis_suite['color_auto'] = postprocess_fn(rendering['rgb'], None)
                    summary_writer.add_image('test_true_auto',
                                             tb_process_fn(postprocess_fn(test_batch['rgb'], None)), step)
                    # Exposure sweep colors.
                    exposures = test_dataset.metadata['exposure_levels']
                    for p, x in list(exposures.items()):
                        vis_suite[f'color/{p}'] = postprocess_fn(rendering['rgb'], x)
                        summary_writer.add_image(f'test_true_color/{p}',
                                                 tb_process_fn(postprocess_fn(test_batch['rgb'], x)), step)
                summary_writer.add_image('test_true_color', tb_process_fn(test_batch['rgb']), step)
                # vis.render_test_semantic(summary_writer, batch, step)

                if config.compute_normal_metrics:
                    summary_writer.add_image('test_true_normals',
                                             tb_process_fn(test_batch['normals']) / 2. + 0.5, step)
                for k, v in vis_suite.items():
                    if v is not None:
                        summary_writer.add_image('test_output_' + k, tb_process_fn(v), step)

    if accelerator.is_local_main_process and config.max_steps % config.checkpoint_every != 0:
        checkpoints.save_checkpoint(
            config.exp_path,
            {
                "state_dict": accelerator.unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            int(config.max_steps), keep=1)


if __name__ == '__main__':
    with gin.config_scope('train'):
        app.run(main)
