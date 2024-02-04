import accelerate
import gin
from internal import coord
from internal import geopoly
from internal import image
from internal import math
from internal import ref_utils
from internal import render
from internal import stepfun
from internal import utils
from internal import obj_utils

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_map
from tqdm import tqdm
from gridencoder import GridEncoder
from torch_scatter import segment_coo

gin.config.external_configurable(math.safe_exp, module='math')


def set_kwargs(self, kwargs):
    for k, v in kwargs.items():
        setattr(self, k, v)


@gin.configurable
class Model(nn.Module):
    """A mip-Nerf360 model containing all MLPs."""
    # num_prop_samples = (128, 64, 64)  # The number of samples for each proposal level.
    num_prop_samples = (64, 64)  # The number of samples for each proposal level.
    num_nerf_samples: int = 32  # The number of samples the final nerf level.
    num_nerf_samples_final: int = 32
    num_levels: int = 3  # The number of sampling levels (3==2 proposals, 1 nerf).
    bg_intensity_range = (1., 1.)  # The range of background colors.
    anneal_slope: float = 10  # Higher = more rapid annealing.
    stop_level_grad: bool = True  # If True, don't backprop across levels.
    use_viewdirs: bool = True  # If True, use view directions as input.
    raydist_fn = 'contract'  # The curve used for ray dists.
    single_jitter: bool = True  # If True, jitter whole rays instead of samples.
    dilation_multiplier: float = 0.5  # How much to dilate intervals relatively.
    dilation_bias: float = 0.0025  # How much to dilate intervals absolutely.
    num_glo_features: int = 0  # GLO vector length, disabled if 0.
    num_glo_embeddings: int = 1000  # Upper bound on max number of train images.
    learned_exposure_scaling: bool = False  # Learned exposure scaling (RawNeRF).
    near_anneal_rate = None  # How fast to anneal in near bound.
    near_anneal_init: float = 0.95  # Where to initialize near bound (in [0, 1]).
    single_mlp: bool = False  # Use the NerfMLP for all rounds of sampling.
    distinct_prop: bool = True  # Use the NerfMLP for all rounds of sampling.
    resample_padding: float = 0.0  # Dirichlet/alpha "padding" on the histogram.
    opaque_background: bool = False  # If true, make the background opaque.
    power_lambda: float = -1.5
    std_scale: float = 0.35
    prop_desired_grid_size = [512, 2048]
    training: bool = False
    # latent_vector_dict: dict =  None

    def __init__(self, config=None, **kwargs):
        super().__init__()
        set_kwargs(self, kwargs)
        self.config = config

        # Construct MLPs. WARNING: Construction order may matter, if MLP weights are
        # being regularized.
        self.nerf_mlp = NerfMLP(num_glo_features=self.num_glo_features,
                                num_glo_embeddings=self.num_glo_embeddings,
                                use_semantic=config.use_semantic,
                                analytic_gradient = config.analytic_gradient,
                                use_intensity=config.use_intensity,
                                no_sem_layer = config.no_sem_layer)
        if self.single_mlp:
            self.prop_mlp = self.nerf_mlp
        elif not self.distinct_prop:
            self.prop_mlp = PropMLP()
        else:
            for i in range(self.num_levels - 1):
                self.register_module(f'prop_mlp_{i}', PropMLP(grid_disired_resolution=self.prop_desired_grid_size[i]))
        # self.prop_mlp = torch.compile(self.prop_mlp)
        if self.num_glo_features > 0 and not config.zero_glo:
            # Construct/grab GLO vectors for the cameras of each input ray.
            self.glo_vecs = nn.Embedding(self.num_glo_embeddings, self.num_glo_features)

        if self.learned_exposure_scaling:
            # Setup learned scaling factors for output colors.
            max_num_exposures = self.num_glo_embeddings
            # Initialize the learned scaling offsets at 0.
            self.exposure_scaling_offsets = nn.Embedding(max_num_exposures, 3)
            torch.nn.init.zeros_(self.exposure_scaling_offsets.weight)

        if config.instance_obj:
            # use obj network for dynamic objects
            obj_info , obj_type_info = self.bboxes
            self.obj_type_info = obj_type_info
            # Track
            self.tracks = []
            # naive thoughts: assign different mlp to instances
            # latent vector: assing different latent for instances, same mlp for same class
            
            # fusion
            if config.latent_size == 0 and not config.fuse_render:
                self.mlp_type = ['instance' for i in range(len(obj_info))]
            else:
                self.mlp_type = ['latent' for i in range(len(obj_info))]
            if config.fuse_render:
                self.mlp_type[-1] = 'latent' # hard code for one fusion
            # self.mlp_type = ['instance' for i in range(len(obj_info))]
            # self.mlp_type[-1] = 'latent'
            for track_id, bbox_infos in obj_info.items():
                class_type = self.obj_type_info[track_id]
                class_id = obj_utils.query_class(class_type)
                # rgb , density, semantic
                # ObjMLP.net_width_viewdirs = 32
                if self.mlp_type[track_id] != 'latent':
                    obj_mlp = ObjMLP(num_glo_features=self.num_glo_features,
                                num_glo_embeddings=self.num_glo_embeddings,
                                # bottleneck_width = 16,
                                # net_width_viewdirs = 16,
                                deg_view = 2,
                                grid_level_interval = 2,
                                # grid_level_dim = 4,
                                grid_base_resolution = 16,
                                warp_fn = None, # no need to use warp function
                                re_weights = False,
                                fixed_semantic = True,
                                use_semantic=config.use_semantic,
                                class_type = class_id,
                                # latent_size = config.latent_size,
                                latent_size = 0,
                                )
                else:
                     obj_mlp = ObjMLP(num_glo_features=self.num_glo_features,
                                num_glo_embeddings=self.num_glo_embeddings,
                                # bottleneck_width = 64,
                                # net_width_viewdirs = 32,
                                deg_view = 2,
                                grid_level_interval = 2,
                                grid_level_dim = 2,
                                grid_base_resolution = 16,
                                warp_fn = None, # no need to use warp function
                                re_weights = False,
                                fixed_semantic = True,
                                use_semantic=config.use_semantic,
                                class_type = class_id,
                                latent_size = config.latent_size,
                                )
                if self.mlp_type[track_id] == 'instance':
                    # assign one mlp to each obj  
                    self.register_module(f'obj_mlp_{track_id}',
                                         obj_mlp
                                        )
                if 'fusion' in class_type:
                    # fusion means come from other scenes
                    self.register_module(f'obj_mlp_{class_id}_fusion',
                                         obj_mlp
                                        )
                else:
                    # 'fusion' not in class_type
                    # latent vector dict is not None
                    # assign one mlp for each class, one latent for each obj
                    self.register_module(f'obj_mlp_{class_id}',
                                         obj_mlp
                                        )
                   
            
                self.tracks.append(bbox_infos) # obj_pose

            self.init_tracks()
            self.instance_obj = True
            if self.latent_vector_dict is not None:
                self.latent_vector_dict = nn.ParameterDict(self.latent_vector_dict)
            
            
        else:
            self.instance_obj = False


    def init_tracks(self,learnable = False):
        tracks = np.stack(self.tracks) # to numpy 
        if not learnable:
            self.tracks = torch.from_numpy(tracks).float() # to tensor 
        else:
            # learnable tracks
            pass

    def manipulate_bboxes(self,angle=5):
        # tracks: n * 100 * 3
        degree_delat = np.deg2rad(angle)
        self.tracks[:,:,4] += degree_delat

    def scale_sample_points(self, step, max_step):
        ratio = max((step/max_step)-0.2,0)
        ratio = min(ratio, 0.75)
        self.num_nerf_samples = int(4*self.num_nerf_samples_final*(1-ratio)//8*8)
        # if step/max_step < 1/10:
        #     self.num_nerf_samples = self.num_nerf_samples_final*4
        # elif step/max_step < 1/3:
        #     self.num_nerf_samples = self.num_nerf_samples_final*2
        # else:
        #     self.num_nerf_samples = self.num_nerf_samples_final
    def hash_decay_loss(self,):
        params = []
        idxs = []
        loss_hash_decay = 0.
        for name, param in sorted(self.named_parameters(), key=lambda x: x[0]):
            if 'encoder' in name:
                if self.config.obj_nodecay and 'obj' in name:
                    continue
                params.append(param)
                if hasattr(self, 'module'):
                    idxs.append(getattr(self.module, name.split('.')[1]).encoder.idx)
                else:
                    idxs.append(getattr(self, name.split('.')[0]).encoder.idx)

        for param, idx in zip(params, idxs):
            loss_hash_decay += segment_coo(param ** 2,
                                        idx,
                                        torch.zeros(idx.max() + 1, param.shape[-1], device=param.device),
                                        reduce='mean'
                                        ).mean()
        return self.config.hash_decay_mults * loss_hash_decay

    def symmetry_constraint(self,results):
        loss_ = 0
        related_keys = ['density','rgb']
        for k in related_keys:
            v = results[k]
            N_rays = v.shape[0]//2
            raw = v[:N_rays,...].detach() # not apply on supervision
            sym = v[N_rays:,...]
        
            # loss_ += ((raw- sym)**2).mean()
            # loss_ += ((raw- sym)**2).mean()
            loss_ += ((raw- sym).abs()).mean()
        return self.config.sym_loss * loss_

    def forward(
            self,
            rand,
            batch,
            train_frac,
            compute_extras,
            zero_glo=True,
            sample_n=7,
            sample_m=3,
            step=0,
            max_step=25000,
            curr_track = None,
    ):
        """The mip-NeRF Model.

    Args:
      rand: random number generator (or None for deterministic output).
      batch: util.Rays, a pytree of ray origins, directions, and viewdirs.
      train_frac: float in [0, 1], what fraction of training is complete.
      compute_extras: bool, if True, compute extra quantities besides color.
      zero_glo: bool, if True, when using GLO pass in vector of zeros.
      sample_n: int, multisamples per frustum
      sample_m: int, loops per frustum

    Returns:
      ret: list, [*(rgb, distance, acc)]
    """
        # cyr
        # self.scale_sample_points(step, max_step)


        device = batch['origins'].device

        if self.num_glo_features > 0:
            if not zero_glo:
                # Construct/grab GLO vectors for the cameras of each input ray.
                cam_idx = batch['cam_idx'][..., 0]
                glo_vec = self.glo_vecs(cam_idx.long())
            else:
                glo_vec = torch.zeros(batch['origins'].shape[:-1] + (self.num_glo_features,), device=device)
        else:
            glo_vec = None

        # Define the mapping from normalized to metric ray distance.
        _, s_to_t = coord.construct_ray_warps(self.raydist_fn, batch['near'], batch['far'], self.power_lambda)

        # Initialize the range of (normalized) distances for each ray to [0, 1],
        # and assign that single interval a weight of 1. These distances and weights
        # will be repeatedly updated as we proceed through sampling levels.
        # `near_anneal_rate` can be used to anneal in the near bound at the start
        # of training, eg. 0.1 anneals in the bound over the first 10% of training.
        if self.near_anneal_rate is None:
            init_s_near = 0.
        else:
            init_s_near = np.clip(1 - train_frac / self.near_anneal_rate, 0,
                                  self.near_anneal_init)
        init_s_far = 1.
        sdist = torch.cat([
            torch.full_like(batch['near'], init_s_near),
            torch.full_like(batch['far'], init_s_far)
        ], dim=-1)
        weights = torch.ones_like(batch['near'])
        prod_num_samples = 1

        ray_history = []
        renderings = []
        # apply for dynamic objects
        if self.instance_obj:
            if curr_track is None:
                if self.tracks is not None:
                    track = self.tracks.to(device)
                else:
                    track = None
            else:
                track = curr_track.to(device)
            # query timestamp for each rays
            obj_pose = obj_utils.get_pose(batch['timestamp'],track)
        for i_level in range(self.num_levels):
            is_prop = i_level < (self.num_levels - 1)
            num_samples = self.num_prop_samples[i_level] if is_prop else self.num_nerf_samples

            # Dilate by some multiple of the expected span of each current interval,
            # with some bias added in.
            dilation = self.dilation_bias + self.dilation_multiplier * (
                    init_s_far - init_s_near) / prod_num_samples

            # Record the product of the number of samples seen so far.
            prod_num_samples *= num_samples

            # After the first level (where dilation would be a no-op) optionally
            # dilate the interval weights along each ray slightly so that they're
            # overestimates, which can reduce aliasing.
            use_dilation = self.dilation_bias > 0 or self.dilation_multiplier > 0
            if i_level > 0 and use_dilation:
                sdist, weights = stepfun.max_dilate_weights(
                    sdist,
                    weights,
                    dilation,
                    domain=(init_s_near, init_s_far),
                    renormalize=True)
                sdist = sdist[..., 1:-1]
                weights = weights[..., 1:-1]

            # Optionally anneal the weights as a function of training iteration.
            if self.anneal_slope > 0:
                # Schlick's bias function, see https://arxiv.org/abs/2010.09714
                bias = lambda x, s: (s * x) / ((s - 1) * x + 1)
                anneal = bias(train_frac, self.anneal_slope)
            else:
                anneal = 1.

            # A slightly more stable way to compute weights**anneal. If the distance
            # between adjacent intervals is zero then its weight is fixed to 0.
            logits_resample = torch.where(
                sdist[..., 1:] > sdist[..., :-1],
                anneal * torch.log(weights + self.resample_padding),
                torch.full_like(sdist[..., :-1], -torch.inf))

            # Draw sampled intervals from each ray's current weights.
            sdist = stepfun.sample_intervals(
                rand,
                sdist,
                logits_resample,
                num_samples,
                single_jitter=self.single_jitter,
                domain=(init_s_near, init_s_far))
            
            # Optimization will usually go nonlinear if you propagate gradients
            # through sampling.
            if self.stop_level_grad:
                sdist = sdist.detach()

            # Convert normalized distances to metric distances.
            tdist = s_to_t(sdist)

            raydist = tdist * batch['directions'].norm(dim=-1)[..., None]
            # Cast our rays, by turning our distance intervals into Gaussians.
            means, stds = render.cast_rays(
                tdist,
                batch['origins'],
                batch['directions'],
                batch['radii'],
                rand,
                n=sample_n,
                m=sample_m,
                std_scale=self.std_scale,
                batch=batch)

            # Push our Gaussians through one of our two MLPs.
            mlp = (self.get_submodule(
                f'prop_mlp_{i_level}') if self.distinct_prop else self.prop_mlp) if is_prop else self.nerf_mlp
            
            
            ray_results = mlp(
                rand,
                means, stds,
                viewdirs=batch['viewdirs'] if self.use_viewdirs else None,
                # imageplane=batch.get('imageplane'),
                glo_vec=None if is_prop else glo_vec,
                exposure=batch.get('exposure_values'),
            )
            # apply for dynamic objects
            if self.instance_obj and obj_pose is not None:
                # tdist: N_batch, N_samples, 3
                t_mids = 0.5 * (tdist[..., :-1] + tdist[..., 1:])
                pts_w = t_mids[...,None] * batch['directions'][:,None,:] + batch['origins'][:,None,:]
                
                is_sym = self.config.symmetrize and not is_prop and self.training
                if is_sym:
                    loss_sym = 0.
                # world2obj
                pts_o, viewdirs_o,intersection_map = obj_utils.box_pts(pts = pts_w,viewdirs=batch['viewdirs'],obj_pose = obj_pose,sym=is_sym)
                
                # naive thoughts: assign different mlp to instances
                # points:  N_batch, N_samples, N_obj,3
                # sym_points:  2* N_batch, N_samples, N_obj,3
                for track_id in range(len(self.bboxes[0].keys())):
                    # get points intersection
                    intersect_idx = intersection_map[:,:,track_id]
                    if intersect_idx.sum() == 0: # no points insertion for current track
                        continue
                    # statics of points, e.g. coords, dirs,... 
                    pts_k = pts_o[intersect_idx][:,track_id,:] # N,3
                    stds = torch.zeros_like(pts_k)[...,0] # N,
                    viewdirs_o_k = viewdirs_o[intersect_idx][:,track_id,:]
                    
                    if self.mlp_type[track_id] == 'instance':
                        # instance mode
                        obj_mlp = self.get_submodule(f'obj_mlp_{track_id}')
                    else:
                        # latent mode
                        class_type = self.obj_type_info[track_id]
                        class_id = obj_utils.query_class(class_type)
                        if 'fusion' in class_type:
                            obj_mlp = self.get_submodule(f'obj_mlp_{class_id}_fusion')
                        else:
                            obj_mlp = self.get_submodule(f'obj_mlp_{class_id}')
                        obj_latent = self.latent_vector_dict[f'obj_latent_{track_id}'].to(device)
                        # expand obj_latent to shape of pts_k
                        obj_latent = obj_latent[None,...].repeat(pts_k.shape[0],1) 
                    obj_ray_results = obj_mlp(
                        rand,
                        pts_k, stds,
                        viewdirs=viewdirs_o_k if self.use_viewdirs else None,
                        latent = obj_latent if self.mlp_type[track_id] != 'instance' else None,
                        glo_vec=None if is_prop else glo_vec,
                        exposure=batch.get('exposure_values'),
                        )
                    if is_prop:
                        # detach the gradient if is_prop
                        obj_ray_results = {k: v.detach() if v is not None else None for k, v in obj_ray_results.items()}

                    elif is_sym:
                        # only apply on final sampling
                        loss_sym += self.symmetry_constraint(obj_ray_results)
                        # half part for appearance loss(composing)
                        obj_ray_results = {k: v[:v.shape[0]//2] if v is not None else None for k, v in obj_ray_results.items()}
                        
                        
                    for key, value in ray_results.items():
                        if value is None:
                            continue

                        if not is_sym:
                            mask = intersection_map[:, :, track_id]
                        else:
                            mask = intersection_map[:intersection_map.shape[0]//2, :, track_id]
                        # Create zero tensor with the same shape as ray_results[key]
                        obj_results_temp = torch.zeros_like(ray_results[key])
                        # Update obj_results_temp using mask and obj_ray_results
                        obj_results_temp[mask] = obj_ray_results[key]
                        if mask.shape != ray_results[key].shape:
                            mask = mask[...,None].expand(ray_results[key].shape)
                        # Update ray_results using mask, ray_results[key], and obj_results_temp
                        ray_results[key] = torch.where(mask, obj_results_temp, ray_results[key])
                # add obj_mask
                obj_mask = intersection_map.sum(-1) > 0 if not is_sym else intersection_map[:intersection_map.shape[0]//2].sum(-1) > 0
                ray_results['obj_mask'] = obj_mask
                ray_results['instance_mask'] = obj_mask

            # Get the weights used by volumetric rendering (and our other losses).
            weights = render.compute_alpha_weights(
                ray_results['density'],
                tdist,
                batch['directions'],
                opaque_background=self.opaque_background,
            )[0]

            # Define or sample the background color for each ray.
            if self.bg_intensity_range[0] == self.bg_intensity_range[1]:
                # If the min and max of the range are equal, just take it.
                bg_rgbs = self.bg_intensity_range[0]
            elif rand is None:
                # If rendering is deterministic, use the midpoint of the range.
                bg_rgbs = (self.bg_intensity_range[0] + self.bg_intensity_range[1]) / 2
            else:
                # Sample RGB values from the range for each ray.
                minval = self.bg_intensity_range[0]
                maxval = self.bg_intensity_range[1]
                bg_rgbs = torch.rand(weights.shape[:-1] + (3,), device=device) * (maxval - minval) + minval

            # RawNeRF exposure logic.
            if batch.get('exposure_idx') is not None:
                # Scale output colors by the exposure.
                ray_results['rgb'] *= batch['exposure_values'][..., None, :]
                if self.learned_exposure_scaling:
                    exposure_idx = batch['exposure_idx'][..., 0]
                    # Force scaling offset to always be zero when exposure_idx is 0.
                    # This constraint fixes a reference point for the scene's brightness.
                    mask = exposure_idx > 0
                    # Scaling is parameterized as an offset from 1.
                    scaling = 1 + mask[..., None] * self.exposure_scaling_offsets(exposure_idx.long())
                    ray_results['rgb'] *= scaling[..., None, :]

            # Render each ray.
            sem = ray_results['semantic'] if i_level==self.num_levels-1 and self.config.use_semantic else None
            intensity = ray_results['intensity'] if i_level==self.num_levels-1 and self.config.use_intensity else None
            rendering = render.volumetric_rendering(
                ray_results['rgb'],
                weights,
                tdist,
                bg_rgbs,
                batch['far'],
                compute_extras,
                semantic=sem,
                intensity = intensity,
                extras={
                    k: v
                    for k, v in ray_results.items()
                    if k.startswith('normals') or k in ['roughness']
                },
                sem_detach=self.config.sem_detach
                )

            if compute_extras:
                # Collect some rays to visualize directly. By naming these quantities
                # with `ray_` they get treated differently downstream --- they're
                # treated as bags of rays, rather than image chunks.
                n = self.config.vis_num_rays
                rendering['ray_sdist'] = sdist.reshape([-1, sdist.shape[-1]])[:n, :]
                rendering['ray_weights'] = (
                    weights.reshape([-1, weights.shape[-1]])[:n, :])
                rgb = ray_results['rgb']
                rendering['ray_rgbs'] = (rgb.reshape((-1,) + rgb.shape[-2:]))[:n, :, :]
            if self.config.instance_obj:
                rendering['obj_mask'] = ray_results['obj_mask'].sum(-1)>0 # obj_mask
                rendering['instance_mask'] = ray_results['instance_mask'].sum(-1)>0 # instance_mask
                # sem = ray_results['semantic']
                # if sem is not None:
                #     rendering['ray_semantics'] = (sem.reshape((-1,) + sem.shape[-2:]))[:n, :, :]

                # dep = ray_results['depth']


            renderings.append(rendering)
            ray_results['sdist'] = sdist.clone()
            ray_results['weights'] = weights.clone()
            ray_results['tdist'] = tdist.clone()
            ray_history.append(ray_results)

        if compute_extras:
            # Because the proposal network doesn't produce meaningful colors, for
            # easier visualization we replace their colors with the final average
            # color.
            weights = [r['ray_weights'] for r in renderings]
            rgbs = [r['ray_rgbs'] for r in renderings]
            final_rgb = torch.sum(rgbs[-1] * weights[-1][..., None], dim=-2)
            avg_rgbs = [
                torch.broadcast_to(final_rgb[:, None, :], r.shape) for r in rgbs[:-1]
            ]
            for i in range(len(avg_rgbs)):
                renderings[i]['ray_rgbs'] = avg_rgbs[i]
        if self.config.hash_decay_mults > 0 and self.training: # only compute hash loss during training.
            hash_decay_loss = self.hash_decay_loss()
            renderings[-1]['hash_decay'] = hash_decay_loss
        if self.config.symmetrize and self.training:
            renderings[-1]['loss_sym'] = loss_sym
        return renderings, ray_history

    # only for obj rendering
    @torch.no_grad()
    def obj_rendering(
            self,
            rand,
            batch,
            train_frac,
            compute_extras,
            zero_glo=True,
            sample_n=7,
            sample_m=3,
            step=0,
            max_step=25000,
            curr_track = None,
            track_id = 0
    ):
        """The mip-NeRF Model.
    """
        # cyr
        device = batch['origins'].device

        if self.num_glo_features > 0:
            if not zero_glo:
                # Construct/grab GLO vectors for the cameras of each input ray.
                cam_idx = batch['cam_idx'][..., 0]
                glo_vec = self.glo_vecs(cam_idx.long())
            else:
                glo_vec = torch.zeros(batch['origins'].shape[:-1] + (self.num_glo_features,), device=device)
        else:
            glo_vec = None

        # Define the mapping from normalized to metric ray distance.
        _, s_to_t = coord.construct_ray_warps(None, batch['near'], batch['far'], self.power_lambda)
        init_s_near = 0.
        init_s_far = 1.
        sdist = torch.cat([
            torch.full_like(batch['near'], init_s_near),
            torch.full_like(batch['far'], init_s_far)
        ], dim=-1)
        weights = torch.ones_like(batch['near'])

        ray_history = []
        renderings = []
        # apply for dynamic objects
        if self.instance_obj:
            if curr_track is None:
                if self.tracks is not None:
                    track = self.tracks.to(device)
                else:
                    track = None
            else:
                track = curr_track.to(device)
            # query timestamp for each rays
            obj_pose = obj_utils.get_pose(batch['timestamp'],track)
        is_prop = False
        num_samples = 64
        anneal = 1.
        # A slightly more stable way to compute weights**anneal. If the distance
        # between adjacent intervals is zero then its weight is fixed to 0.
        logits_resample = torch.where(
            sdist[..., 1:] > sdist[..., :-1],
            anneal * torch.log(weights + self.resample_padding),
            torch.full_like(sdist[..., :-1], -torch.inf))

        # Draw sampled intervals from each ray's current weights.
        sdist = stepfun.sample_intervals(
            rand,
            sdist,
            logits_resample,
            num_samples,
            single_jitter=self.single_jitter,
            domain=(init_s_near, init_s_far))
        
        # Optimization will usually go nonlinear if you propagate gradients
        # through sampling.
        if self.stop_level_grad:
            sdist = sdist.detach()

        # Convert normalized distances to metric distances.
        tdist = s_to_t(sdist)

        t_mids = 0.5 * (tdist[..., :-1] + tdist[..., 1:])
        pts_w = t_mids[...,None] * batch['directions'][:,None,:] + batch['origins'][:,None,:]
        # world2obj
        pts_o, viewdirs_o,intersection_map = obj_utils.box_pts(pts = pts_w,viewdirs=batch['viewdirs'],obj_pose = obj_pose,transform = False)
        pts_o = pts_w; viewdirs_o=batch['viewdirs']
        # naive thoughts: assign different mlp to instances
        track_id = track_id
        # for track_id in range(len(self.bboxes[0].keys())):
            # get points intersection
        # intersect_idx = intersection_map[:,:,track_id]
        # intersect_idx = torch.ones_like(pts_o[...,0]) # all intersection
       
        # statics of points, e.g. coords, dirs,... 
        pts_k = pts_o # N,3
        
        stds = torch.zeros_like(pts_k)[...,0] # N,
        viewdirs_o_k = viewdirs_o
        
        if self.latent_vector_dict is None:
            obj_mlp = self.get_submodule(f'obj_mlp_{track_id}')
        else:
            class_id = obj_utils.query_class(self.obj_type_info[track_id])
            obj_mlp = self.get_submodule(f'obj_mlp_{class_id}')
            obj_latent = self.latent_vector_dict[f'obj_latent_{track_id}'].to(device)
            # expand obj_latent to shape of pts_k
            obj_latent = obj_latent[None,...].repeat(pts_k.shape[0],1) 
        obj_ray_results = obj_mlp(
            rand,
            pts_k, stds,
            viewdirs=viewdirs_o_k if self.use_viewdirs else None,
            latent = obj_latent if self.latent_vector_dict is not None else None,
            glo_vec=None if is_prop else glo_vec,
            exposure=batch.get('exposure_values'),
            )

        #TODO replace this part
        ray_results = obj_ray_results
        # add obj_mask
        obj_mask = intersection_map.sum(-1) > 0
        ray_results['obj_mask'] = obj_mask
        ray_results['instance_mask'] = obj_mask

        # Get the weights used by volumetric rendering (and our other losses).
        weights = render.compute_alpha_weights(
            ray_results['density'],
            tdist,
            batch['directions'],
            opaque_background=self.opaque_background,
        )[0]

        # Define or sample the background color for each ray.
        if self.bg_intensity_range[0] == self.bg_intensity_range[1]:
            # If the min and max of the range are equal, just take it.
            bg_rgbs = self.bg_intensity_range[0]
        elif rand is None:
            # If rendering is deterministic, use the midpoint of the range.
            bg_rgbs = (self.bg_intensity_range[0] + self.bg_intensity_range[1]) / 2
        else:
            # Sample RGB values from the range for each ray.
            minval = self.bg_intensity_range[0]
            maxval = self.bg_intensity_range[1]
            bg_rgbs = torch.rand(weights.shape[:-1] + (3,), device=device) * (maxval - minval) + minval

        # RawNeRF exposure logic.
        if batch.get('exposure_idx') is not None:
            # Scale output colors by the exposure.
            ray_results['rgb'] *= batch['exposure_values'][..., None, :]
            if self.learned_exposure_scaling:
                exposure_idx = batch['exposure_idx'][..., 0]
                # Force scaling offset to always be zero when exposure_idx is 0.
                # This constraint fixes a reference point for the scene's brightness.
                mask = exposure_idx > 0
                # Scaling is parameterized as an offset from 1.
                scaling = 1 + mask[..., None] * self.exposure_scaling_offsets(exposure_idx.long())
                ray_results['rgb'] *= scaling[..., None, :]

        # Render each ray.
        sem = ray_results['semantic'] if self.config.use_semantic else None
        intensity = ray_results['intensity'] if self.config.use_intensity else None
        rendering = render.volumetric_rendering(
            ray_results['rgb'],
            weights,
            tdist,
            bg_rgbs,
            batch['far'],
            compute_extras,
            semantic=sem,
            intensity = intensity,
            extras={
                k: v
                for k, v in ray_results.items()
                if k.startswith('normals') or k in ['roughness']
            })

        if compute_extras:
            # Collect some rays to visualize directly. By naming these quantities
            # with `ray_` they get treated differently downstream --- they're
            # treated as bags of rays, rather than image chunks.
            n = self.config.vis_num_rays
            rendering['ray_sdist'] = sdist.reshape([-1, sdist.shape[-1]])[:n, :]
            rendering['ray_weights'] = (
                weights.reshape([-1, weights.shape[-1]])[:n, :])
            rgb = ray_results['rgb']
            rendering['ray_rgbs'] = (rgb.reshape((-1,) + rgb.shape[-2:]))[:n, :, :]
        if self.config.instance_obj:
            rendering['obj_mask'] = ray_results['obj_mask'].sum(-1)>0 # obj_mask
            rendering['instance_mask'] = ray_results['instance_mask'].sum(-1)>0 # instance_mask
            # sem = ray_results['semantic']
            # if sem is not None:
            #     rendering['ray_semantics'] = (sem.reshape((-1,) + sem.shape[-2:]))[:n, :, :]

            # dep = ray_results['depth']


        renderings.append(rendering)
        ray_results['sdist'] = sdist.clone()
        ray_results['weights'] = weights.clone()
        ray_results['tdist'] = tdist.clone()
        ray_history.append(ray_results)

        if compute_extras:
            # Because the proposal network doesn't produce meaningful colors, for
            # easier visualization we replace their colors with the final average
            # color.
            weights = [r['ray_weights'] for r in renderings]
            rgbs = [r['ray_rgbs'] for r in renderings]
            final_rgb = torch.sum(rgbs[-1] * weights[-1][..., None], dim=-2)
            avg_rgbs = [
                torch.broadcast_to(final_rgb[:, None, :], r.shape) for r in rgbs[:-1]
            ]
            for i in range(len(avg_rgbs)):
                renderings[i]['ray_rgbs'] = avg_rgbs[i]
        if self.config.hash_decay_mults > 0 and self.training: # only compute hash loss during training.
            hash_decay_loss = self.hash_decay_loss()
            renderings[-1]['hash_decay'] = hash_decay_loss
        return renderings, ray_history

class MLP(nn.Module):
    """A PosEnc MLP."""
    bottleneck_width: int = 256  # The width of the bottleneck vector.
    net_depth_viewdirs: int = 2  # The depth of the second part of ML.
    net_width_viewdirs: int = 256  # The width of the second part of MLP.
    skip_layer_dir: int = 0  # Add a skip connection to 2nd MLP after Nth layers.
    num_rgb_channels: int = 3  # The number of RGB channels.
    deg_view: int = 4  # Degree of encoding for viewdirs or refdirs.
    use_reflections: bool = False  # If True, use refdirs instead of viewdirs.
    use_directional_enc: bool = False  # If True, use IDE to encode directions.
    # If False and if use_directional_enc is True, use zero roughness in IDE.
    enable_pred_roughness: bool = False
    roughness_bias: float = -1.  # Shift added to raw roughness pre-activation.
    use_diffuse_color: bool = False  # If True, predict diffuse & specular colors.
    use_specular_tint: bool = False  # If True, predict tint.
    use_n_dot_v: bool = False  # If True, feed dot(n * viewdir) to 2nd MLP.
    bottleneck_noise: float = 0.0  # Std. deviation of noise added to bottleneck.
    density_bias: float = -1.  # Shift added to raw densities pre-activation.
    density_noise: float = 0.  # Standard deviation of noise added to raw density.
    rgb_premultiplier: float = 1.  # Premultiplier on RGB before activation.
    rgb_bias: float = 0.  # The shift added to raw colors pre-activation.
    rgb_padding: float = 0.001  # Padding added to the RGB outputs.
    enable_pred_normals: bool = False  # If True compute predicted normals.
    disable_density_normals: bool = False  # If True don't compute normals.
    disable_rgb: bool = False  # If True don't output RGB.
    warp_fn = 'contract'
    num_glo_features: int = 0  # GLO vector length, disabled if 0.
    num_glo_embeddings: int = 1000  # Upper bound on max number of train images.
    scale_featurization: bool = False
    grid_num_levels: int = 10
    grid_level_interval: int = 2
    grid_level_dim: int = 4
    grid_base_resolution: int = 16
    grid_disired_resolution: int = 8192
    grid_log2_hashmap_size: int = 21
    net_width_glo: int = 128  # The width of the second part of MLP.
    net_depth_glo: int = 2  # The width of the second part of MLP.

    class_num: int = 19
    use_semantic: bool = False
    analytic_gradient: bool = True
    use_intensity: bool = False
    no_sem_layer: bool = True
    density_init: bool = False
    re_weights: bool = True
    fixed_semantic: bool = False
    class_type: int = 255
    obj_mode: bool = False
    complex_decoder: bool = False
    latent_size: int = 0
    split_latent: bool = False
    def __init__(self, **kwargs):
        super().__init__()
        set_kwargs(self, kwargs)
        # Make sure that normals are computed if reflection direction is used.
        if self.use_reflections and not (self.enable_pred_normals or
                                         not self.disable_density_normals):
            raise ValueError('Normals must be computed for reflection directions.')

        # Precompute and define viewdir or refdir encoding function.
        if self.use_directional_enc:
            self.dir_enc_fn = ref_utils.generate_ide_fn(self.deg_view)
            dim_dir_enc = self.dir_enc_fn(torch.zeros(1, 3), torch.zeros(1, 1)).shape[-1]
        else:

            def dir_enc_fn(direction, _):
                return coord.pos_enc(
                    direction, min_deg=0, max_deg=self.deg_view, append_identity=True)

            self.dir_enc_fn = dir_enc_fn
            dim_dir_enc = self.dir_enc_fn(torch.zeros(1, 3), None).shape[-1]
        self.grid_num_levels = int(np.log(self.grid_disired_resolution/self.grid_base_resolution)/np.log(self.grid_level_interval)) + 1
        self.encoder = GridEncoder(input_dim=3,
                                   num_levels=self.grid_num_levels,
                                   level_dim=self.grid_level_dim,
                                   base_resolution=self.grid_base_resolution,
                                   desired_resolution=self.grid_disired_resolution,
                                   log2_hashmap_size=self.grid_log2_hashmap_size,
                                   gridtype='hash',
                                   align_corners=False)
        last_dim = self.encoder.output_dim
        if self.scale_featurization:
            last_dim += self.encoder.num_levels
        if self.latent_size > 0:
            if not self.split_latent:
                last_dim += self.latent_size
            else:
                # split shape and texture
                last_dim += self.latent_size //2
        if not self.obj_mode:
            if not self.complex_decoder:
                self.density_layer = nn.Sequential(nn.Linear(last_dim, 64),
                                           nn.ReLU(),
                                           nn.Linear(64, 1 if self.disable_rgb else self.bottleneck_width))  # Hardcoded to a single channel.
            else:
                self.density_layer = nn.Sequential(nn.Linear(last_dim, 128),
                                           nn.ReLU(),
                                           nn.Linear(128, 128),
                                           nn.ReLU(),
                                           nn.Linear(128, 128),
                                           nn.ReLU(),
                                           nn.Linear(128, 1 if self.disable_rgb else self.bottleneck_width))  # Hardcoded to a single channel.
        else:
            self.density_layer = nn.Sequential(nn.Linear(last_dim, 32),
                                           nn.ReLU(),
                                           nn.Linear(32, 1 if self.disable_rgb else self.bottleneck_width))  # Hardcoded to a single channel.
        if self.density_init:
            self.density_layer[2].bias.data[0] = self.density_layer[2].bias.data[0] + 0.1 # apply init

        last_dim = 1 if self.disable_rgb and not self.enable_pred_normals else self.bottleneck_width
        if self.enable_pred_normals:
            self.normal_layer = nn.Linear(last_dim, 3)

        if not self.disable_rgb:
            if self.use_diffuse_color:
                self.diffuse_layer = nn.Linear(last_dim, self.num_rgb_channels)

            if self.use_specular_tint:
                self.specular_layer = nn.Linear(last_dim, 3)

            if self.enable_pred_roughness:
                self.roughness_layer = nn.Linear(last_dim, 1)

            # Output of the first part of MLP.
            if self.bottleneck_width > 0:
                last_dim_rgb = self.bottleneck_width
            else:
                last_dim_rgb = 0
            if self.split_latent:
                last_dim_rgb += self.latent_size//2 # for texture
            last_dim_rgb += dim_dir_enc

            if self.use_n_dot_v:
                last_dim_rgb += 1

            if self.num_glo_features > 0:
                last_dim_glo = self.num_glo_features
                for i in range(self.net_depth_glo - 1):
                    self.register_module(f"lin_glo_{i}", nn.Linear(last_dim_glo, self.net_width_glo))
                    last_dim_glo = self.net_width_glo
                self.register_module(f"lin_glo_{self.net_depth_glo - 1}", nn.Linear(last_dim_glo, self.bottleneck_width * 2))

            input_dim_rgb = last_dim_rgb
            for i in range(self.net_depth_viewdirs):
                lin = nn.Linear(last_dim_rgb, self.net_width_viewdirs)
                torch.nn.init.kaiming_uniform_(lin.weight)
                self.register_module(f"lin_second_stage_{i}", lin)

                # sem_lin = nn.Linear(last_dim_rgb, self.net_width_viewdirs)
                # torch.nn.init.kaiming_uniform_(sem_lin.weight)
                # self.register_module(f"lin_second_stage_sem_{i}", sem_lin)

                last_dim_rgb = self.net_width_viewdirs
                if i == self.skip_layer_dir:
                    last_dim_rgb += input_dim_rgb
            self.rgb_layer = nn.Linear(last_dim_rgb, self.num_rgb_channels)

            # self.sem_layer = nn.Linear(last_dim_rgb, self.class_num)
            if not self.no_sem_layer and not self.fixed_semantic:
                self.sem_layer = nn.Sequential(nn.Linear(self.bottleneck_width, 64), # v3
                                           nn.ReLU(),
                                           nn.Linear(64, self.class_num))
            if self.use_intensity:
                self.intensity_layer = nn.Sequential(nn.Linear(self.bottleneck_width, 64), # v3
                                           nn.ReLU(),
                                           nn.Linear(64, 1))
            # if self.fixed_semantic:
            #     assert self.class_type != 255

    def predict_density(self, means, stds, return_weights=False, rand=False, latent= None):
        """Helper function to output density."""
        # Encode input positions
        if self.warp_fn is not None:
            means, stds = coord.track_linearize(self.warp_fn, means, stds)
            # contract [-2, 2] to [-1, 1]
            bound = 2
            means = means / bound
            stds = stds / bound
        features = self.encoder(means, bound=1).unflatten(-1, (self.encoder.num_levels, -1))
        if self.re_weights:
            weights = torch.erf(1 / torch.clamp(torch.sqrt(8 * stds[..., None] ** 2 * self.encoder.grid_sizes ** 2),min=1e-10))
            features = (features * weights[..., None]).mean(dim=-3).flatten(-2, -1) 
        else:
            features = features.flatten(-2, -1) 
        if self.scale_featurization:
            with torch.no_grad():
                vl2mean = segment_coo((self.encoder.embeddings ** 2).sum(-1),
                                      self.encoder.idx,
                                      torch.zeros(self.grid_num_levels, device=weights.device),
                                      self.grid_num_levels,
                                      reduce='mean'
                                      )
            featurized_w = (2 * weights.mean(dim=-2) - 1) * (self.encoder.init_std ** 2 + vl2mean).sqrt()
            features = torch.cat([features, featurized_w], dim=-1)
        if latent is not None:
            if not self.split_latent:
                features = torch.cat([features,latent],dim =-1)
            else:
                shape_latent = latent[...,:self.latent_size//2]
                features = torch.cat([features,shape_latent],dim =-1)
        x = self.density_layer(features)
        raw_density = x[..., 0]  # Hardcoded to a single channel.

        # Add noise to regularize the density predictions if needed.
        if rand and (self.density_noise > 0):
            raw_density += self.density_noise * torch.randn_like(raw_density)
        if return_weights:
            return raw_density, x, weights
        return raw_density, x
    def feature_encode(self,means,stds):
        if self.warp_fn is not None:
            means, stds = coord.track_linearize(self.warp_fn, means, stds)
            # contract [-2, 2] to [-1, 1]
            bound = 2
            means = means / bound
            stds = stds / bound
        features = self.encoder(means, bound=1).unflatten(-1, (self.encoder.num_levels, -1))
        weights = torch.erf(1 / torch.sqrt(8 * stds[..., None] ** 2 * self.encoder.grid_sizes ** 2))
        features = (features * weights[..., None]).mean(dim=-3).flatten(-2, -1)
        x = self.density_layer(features)
        raw_density = x[..., 0]
        return raw_density, x
    def finite_difference_normal(self, means,stds, epsilon=1e-2):
        # x: [N, 3]
        
        dx_pos, _ = self.feature_encode((means + torch.tensor([[epsilon, 0.00, 0.00]], device=means.device)).clamp(-1, 1),stds)
        dx_neg, _ = self.feature_encode((means + torch.tensor([[-epsilon, 0.00, 0.00]], device=means.device)).clamp(-1, 1),stds)
        dy_pos, _ = self.feature_encode((means + torch.tensor([[0.00, epsilon, 0.00]], device=means.device)).clamp(-1, 1),stds)
        dy_neg, _ = self.feature_encode((means + torch.tensor([[0.00, -epsilon, 0.00]], device=means.device)).clamp(-1, 1),stds)
        dz_pos, _ = self.feature_encode((means + torch.tensor([[0.00, 0.00, epsilon]], device=means.device)).clamp(-1, 1),stds)
        dz_neg, _ = self.feature_encode((means + torch.tensor([[0.00, 0.00, -epsilon]], device=means.device)).clamp(-1, 1),stds)
        
        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon, 
            0.5 * (dy_pos - dy_neg) / epsilon, 
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return -normal
    
    def forward(self,
                rand,
                means, stds,
                viewdirs=None,
                # imageplane=None,
                latent = None,
                glo_vec = None,
                exposure = None,
                ):
        """Evaluate the MLP.

    Args:
      rand: if random .
      means: [..., n, 3], coordinate means.
      stds: [..., n], coordinate stds.
      viewdirs: [..., 3], if not None, this variable will
        be part of the input to the second part of the MLP concatenated with the
        output vector of the first part of the MLP. If None, only the first part
        of the MLP will be used with input x. In the original paper, this
        variable is the view direction.
      imageplane:[batch, 2], xy image plane coordinates
        for each ray in the batch. Useful for image plane operations such as a
        learned vignette mapping.
      glo_vec: [..., num_glo_features], The GLO vector for each ray.
      exposure: [..., 1], exposure value (shutter_speed * ISO) for each ray.
      re_weights: whether to apply different weights in the ray_samples. True for multisampling, False for instance.

    Returns:
      rgb: [..., num_rgb_channels].
      density: [...].
      normals: [..., 3], or None.
      normals_pred: [..., 3], or None.
      roughness: [..., 1], or None.
    """
        if self.disable_density_normals:
            raw_density, x = self.predict_density(means, stds, rand=rand,latent = latent)
            raw_grad_density = None
            normals = None
        else:
            if self.analytic_gradient:
                # analytical gradient
                means.requires_grad_(True)
                # means: N_rays * N_samples * sample_n * sample_m * 3
                raw_density, x, weights = self.predict_density(means, stds, True, rand=rand)
                d_output = torch.ones_like(raw_density, requires_grad=False, device=raw_density.device)
                
                raw_grad_density = torch.autograd.grad(
                    outputs=raw_density,
                    inputs=means,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)[0]
                raw_grad_density = raw_grad_density.mean(-2)
                # Compute normal vectors as negative normalized density gradient.
                # We normalize the gradient of raw (pre-activation) density because
                # it's the same as post-activation density, but is more numerically stable
                # when the activation function has a steep or flat gradient.
                normals = -ref_utils.l2_normalize(raw_grad_density)
                # ana_normals = normals
            # numerical gradient: finite difference
            else:
                normals = self.finite_difference_normal(means,stds)
                normals = utils.safe_normalize(normals)
                normals = torch.nan_to_num(normals)

                raw_density, x, weights = self.predict_density(means, stds, True, rand=rand)
                raw_grad_density = None
        if self.enable_pred_normals:
            grad_pred = self.normal_layer(x)

            # Normalize negative predicted gradients to get predicted normal vectors.
            normals_pred = -ref_utils.l2_normalize(grad_pred)
            normals_to_use = normals_pred
        else:
            grad_pred = None
            normals_pred = None
            normals_to_use = normals

        # Apply bias and activation to raw density
        density = F.softplus(raw_density + self.density_bias)

        roughness = None
        if self.disable_rgb:
            rgb = torch.zeros(density.shape + (3,), device=density.device)
            sem = None
            intensity = None
        else:
            if self.use_semantic:
                if self.fixed_semantic:
                    # sem = torch.ones_like(x[...,0]) * self.class_type
                    sem = torch.repeat_interleave(torch.zeros_like(x[...,0:1]),self.class_num,dim=-1)
                    if self.class_type != 255:
                        sem[...,self.class_type] = 1.
                    sem = sem.detach() # no need for gradients
                else:
                    if self.no_sem_layer:
                        sem = x[..., 1:(1+self.class_num)] # v4
                    else:
                        sem = self.sem_layer(x)  # v3
                    sem = torch.softmax(sem, -1) # probability instead of logits
            else:
                sem = None
            # sem = None
            if self.use_intensity:
                intensity = self.intensity_layer(x)  # v3
            else:
                intensity = None
            if viewdirs is not None:
                # Predict diffuse color.
                if self.use_diffuse_color:
                    raw_rgb_diffuse = self.diffuse_layer(x)

                if self.use_specular_tint:
                    tint = torch.sigmoid(self.specular_layer(x))

                if self.enable_pred_roughness:
                    raw_roughness = self.roughness_layer(x)
                    roughness = (F.softplus(raw_roughness + self.roughness_bias))

                # Output of the first part of MLP.
                if self.bottleneck_width > 0:
                    bottleneck = x
                    # Add bottleneck noise.
                    if rand and (self.bottleneck_noise > 0):
                        bottleneck += self.bottleneck_noise * torch.randn_like(bottleneck)

                    # Append GLO vector if used.
                    if glo_vec is not None:
                        for i in range(self.net_depth_glo):
                            glo_vec = self.get_submodule(f"lin_glo_{i}")(glo_vec)
                            if i != self.net_depth_glo - 1:
                                glo_vec = F.relu(glo_vec)
                        glo_vec = torch.broadcast_to(glo_vec[..., None, :],
                                                     bottleneck.shape[:-1] + glo_vec.shape[-1:])
                        scale, shift = glo_vec.chunk(2, dim=-1)
                        bottleneck = bottleneck * torch.exp(scale) + shift

                    x = [bottleneck]

                else:
                    x = []

                # Encode view (or reflection) directions.
                if self.use_reflections:
                    # Compute reflection directions. Note that we flip viewdirs before
                    # reflecting, because they point from the camera to the point,
                    # whereas ref_utils.reflect() assumes they point toward the camera.
                    # Returned refdirs then point from the point to the environment.
                    refdirs = ref_utils.reflect(-viewdirs[..., None, :], normals_to_use)
                    # Encode reflection directions.
                    dir_enc = self.dir_enc_fn(refdirs, roughness)
                else:
                    # Encode view directions.
                    dir_enc = self.dir_enc_fn(viewdirs, roughness)

                    if dir_enc.dim() != bottleneck.dim():
                        # apply broadcast
                        dir_enc = torch.broadcast_to(
                        dir_enc[..., None, :],
                        bottleneck.shape[:-1] + (dir_enc.shape[-1],))

                # Append view (or reflection) direction encoding to bottleneck vector.
                x.append(dir_enc)

                if self.split_latent:
                    texture_latent = latent[...,self.latent_size//2:]
                    x.append(texture_latent)

                # Append dot product between normal vectors and view directions.
                if self.use_n_dot_v:
                    dotprod = torch.sum(
                        normals_to_use * viewdirs[..., None, :], dim=-1, keepdim=True)
                    x.append(dotprod)

                # Concatenate bottleneck, directional encoding, and GLO.
                x = torch.cat(x, dim=-1)

                # if self.use_semantic:
                #     inputs = x+0
                #     y = x+0
                #     for i in range(self.net_depth_viewdirs):
                #         y = self.get_submodule(f"lin_second_stage_sem_{i}")(y)
                #         y = F.relu(y)
                #         if i == self.skip_layer_dir:
                #             y = torch.cat([y, inputs], dim=-1)
                # Output of the second part of MLP.
                inputs = x
                for i in range(self.net_depth_viewdirs):
                    x = self.get_submodule(f"lin_second_stage_{i}")(x)
                    x = F.relu(x)
                    if i == self.skip_layer_dir:
                        x = torch.cat([x, inputs], dim=-1)

            # If using diffuse/specular colors, then `rgb` is treated as linear
            # specular color. Otherwise it's treated as the color itself.
            rgb = torch.sigmoid(self.rgb_premultiplier *
                                self.rgb_layer(x) +
                                self.rgb_bias)
            # if self.use_semantic:
            #     sem = self.sem_layer(y)

            if self.use_diffuse_color:
                # Initialize linear diffuse color around 0.25, so that the combined
                # linear color is initialized around 0.5.
                diffuse_linear = torch.sigmoid(raw_rgb_diffuse - np.log(3.0))
                if self.use_specular_tint:
                    specular_linear = tint * rgb
                else:
                    specular_linear = 0.5 * rgb

                # Combine specular and diffuse components and tone map to sRGB.
                rgb = torch.clip(image.linear_to_srgb(specular_linear + diffuse_linear), 0.0, 1.0)

            # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

        return dict(
            density=density,
            rgb=rgb,
            raw_grad_density=raw_grad_density,
            grad_pred=grad_pred,
            normals=normals,
            normals_pred=normals_pred,
            roughness=roughness,
            semantic=sem,
            intensity = intensity,
        )


@gin.configurable
class NerfMLP(MLP):
    pass

# for dynamic objects
@gin.configurable
class ObjMLP(MLP):
    pass

@gin.configurable
class PropMLP(MLP):
    pass


# @torch.no_grad()
# def render_image(render_fn,
#                  accelerator: accelerate.Accelerator,
#                  batch,
#                  rand,
#                  config,
#                  image = True):
#     """Render all the pixels of an image (in test mode).

#   Args:
#     render_fn: function, jit-ed render function mapping (rand, batch) -> pytree.
#     accelerator: used for DDP.
#     batch: a `Rays` pytree, the rays to be rendered.
#     rand: if random
#     config: A Config class.

#   Returns:
#     rgb: rendered color image.
#     disp: rendered disparity image.
#     acc: rendered accumulated weights per pixel.
#   """
#     if image:
#         height, width = batch['origins'].shape[:2]
#         num_rays = height * width
#     else:

#         num_rays = batch['origins'].shape[0]
#     batch = {k: v.reshape((num_rays, -1)) for k, v in batch.items() if v is not None}


#     local_rank = accelerator.local_process_index
#     chunks = []
#     if accelerator.is_local_main_process:
#         idx0s = tqdm(range(0, num_rays, config.render_chunk_size), desc="Rendering chunk")
#     else:
#         idx0s = range(0, num_rays, config.render_chunk_size)

#     for i_chunk, idx0 in enumerate(idx0s):
#         chunk_batch = tree_map(lambda r: r[idx0:idx0 + config.render_chunk_size], batch)
#         actual_chunk_size = chunk_batch['origins'].shape[0]
#         rays_remaining = actual_chunk_size % accelerator.num_processes
#         if rays_remaining != 0:
#             padding = accelerator.num_processes - rays_remaining
#             chunk_batch = tree_map(lambda v: torch.cat([v, torch.zeros_like(v[-padding:])], dim=0), chunk_batch)
#         else:
#             padding = 0
#         # After padding the number of chunk_rays is always divisible by host_count.
#         rays_per_host = chunk_batch['origins'].shape[0] // accelerator.num_processes
#         start, stop = local_rank * rays_per_host, (local_rank + 1) * rays_per_host
#         chunk_batch = tree_map(lambda r: r[start:stop], chunk_batch)

#         with accelerator.autocast():
#             chunk_renderings, _ = render_fn(rand, chunk_batch)

#         # Unshard the renderings.
#         chunk_renderings = tree_map(
#             lambda v: accelerator.gather(v.contiguous())[:-padding]
#             if padding > 0 else accelerator.gather(v.contiguous()), chunk_renderings)

#         # Gather the final pass for 2D buffers and all passes for ray bundles.
#         chunk_rendering = chunk_renderings[-1]
#         for k in chunk_renderings[0]:
#             if k.startswith('ray_'):
#                 chunk_rendering[k] = [r[k] for r in chunk_renderings]

#         chunks.append(chunk_rendering)
#     # Concatenate all chunks within each leaf of a single pytree.
#     rendering = {}
#     for k in chunks[0].keys():
#         if 'hash' in k:
#             continue
#         if isinstance(chunks[0][k], list):
#             rendering[k] = []
#             for i in range(len(chunks[0][k])):
#                 rendering[k].append(torch.cat([item[k][i] for item in chunks]))
#         else:
#             rendering[k] = torch.cat([item[k] for item in chunks])

#     for k, z in rendering.items():
#         if not k.startswith('ray_'):
#             # Reshape 2D buffers into original image shape.
#             if 'hash' in k:
#                 continue
#             if image:
#                 rendering[k] = z.reshape((height, width) + z.shape[1:])

#     # After all of the ray bundles have been concatenated together, extract a
#     # new random bundle (deterministically) from the concatenation that is the
#     # same size as one of the individual bundles.
#     keys = [k for k in rendering if k.startswith('ray_')]
#     if keys:
#         num_rays = rendering[keys[0]][0].shape[0]
#         ray_idx = torch.randperm(num_rays)
#         ray_idx = ray_idx[:config.vis_num_rays]
#         for k in keys:
#             rendering[k] = [r[ray_idx] for r in rendering[k]]

#     return rendering

@torch.no_grad()
def render_image(model,
                 accelerator: accelerate.Accelerator,
                 batch,
                 rand,
                 config,
                 train_frac = 1,
                 verbose=True,
                 return_weights=False,
                 image = True,
                 render_instance = False,
                 instance_id = None):
    """Render all the pixels of an image (in test mode).

  Args:
    render_fn: function, jit-ed render function mapping (rand, batch) -> pytree.
    accelerator: used for DDP.
    batch: a `Rays` pytree, the rays to be rendered.
    rand: if random
    config: A Config class.

  Returns:
    rgb: rendered color image.
    disp: rendered disparity image.
    acc: rendered accumulated weights per pixel.
  """
    model.eval()

    # height, width = batch['origins'].shape[:2]
    # num_rays = height * width
    if render_instance:
        assert instance_id is not None
    if image:
        height, width = batch['origins'].shape[:2]
        num_rays = height * width
    else:
        num_rays = batch['origins'].shape[0]

    batch = {k: v.reshape((num_rays, -1)) for k, v in batch.items() if v is not None}

    global_rank = accelerator.process_index
    chunks = []
    idx0s = tqdm(range(0, num_rays, config.render_chunk_size),
                 desc="Rendering chunk", leave=False,
                 disable=not (accelerator.is_main_process and verbose))

    for i_chunk, idx0 in enumerate(idx0s):
        chunk_batch = tree_map(lambda r: r[idx0:idx0 + config.render_chunk_size], batch)
        actual_chunk_size = chunk_batch['origins'].shape[0]
        rays_remaining = actual_chunk_size % accelerator.num_processes
        if rays_remaining != 0:
            padding = accelerator.num_processes - rays_remaining
            chunk_batch = tree_map(lambda v: torch.cat([v, torch.zeros_like(v[-padding:])], dim=0), chunk_batch)
        else:
            padding = 0
        # After padding the number of chunk_rays is always divisible by host_count.
        rays_per_host = chunk_batch['origins'].shape[0] // accelerator.num_processes
        start, stop = global_rank * rays_per_host, (global_rank + 1) * rays_per_host
        chunk_batch = tree_map(lambda r: r[start:stop], chunk_batch)

        with accelerator.autocast():
            if not render_instance:
                chunk_renderings, ray_history = model(rand,
                                                  chunk_batch,
                                                  train_frac=train_frac,
                                                  compute_extras=True,
                                                  zero_glo=True)
            else:
                chunk_renderings, ray_history = model.obj_rendering(rand,
                                                  chunk_batch,
                                                  train_frac=train_frac,
                                                  compute_extras=True,
                                                  zero_glo=True,
                                                  track_id = instance_id)

        gather = lambda v: accelerator.gather(v.contiguous())[:-padding] \
            if padding > 0 else accelerator.gather(v.contiguous())
        # Unshard the renderings.
        chunk_renderings = tree_map(gather, chunk_renderings)

        # Gather the final pass for 2D buffers and all passes for ray bundles.
        chunk_rendering = chunk_renderings[-1]
        for k in chunk_renderings[0]:
            if k.startswith('ray_'):
                chunk_rendering[k] = [r[k] for r in chunk_renderings]
            # if 'hash' in k:
            #     continue
            # if isinstance(chunks[0][k], list):
            #     rendering[k] = []
            #     for i in range(len(chunks[0][k])):
            #         rendering[k].append(torch.cat([item[k][i] for item in chunks]))
            # else:
            #     rendering[k] = torch.cat([item[k] for item in chunks])


        if return_weights:
            chunk_rendering['weights'] = gather(ray_history[-1]['weights'])
            chunk_rendering['coord'] = gather(ray_history[-1]['coord'])
        chunks.append(chunk_rendering)

    # Concatenate all chunks within each leaf of a single pytree.
    rendering = {}
    for k in chunks[0].keys():
        if isinstance(chunks[0][k], list):
            rendering[k] = []
            for i in range(len(chunks[0][k])):
                rendering[k].append(torch.cat([item[k][i] for item in chunks]))
        else:
            rendering[k] = torch.cat([item[k] for item in chunks])

    for k, z in rendering.items():
        if not k.startswith('ray_') and 'hash' not in k:
            # Reshape 2D buffers into original image shape.
            if image:
                rendering[k] = z.reshape((height, width) + z.shape[1:])
            else:
                rendering[k] = z.reshape(num_rays,-1)
    # After all of the ray bundles have been concatenated together, extract a
    # new random bundle (deterministically) from the concatenation that is the
    # same size as one of the individual bundles.
    keys = [k for k in rendering if k.startswith('ray_')]
    if keys:
        num_rays = rendering[keys[0]][0].shape[0]
        ray_idx = torch.randperm(num_rays)
        ray_idx = ray_idx[:config.vis_num_rays]
        for k in keys:
            rendering[k] = [r[ray_idx] for r in rendering[k]]
    model.train()
    return rendering
