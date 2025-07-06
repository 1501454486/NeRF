import torch
import torch.nn as nn
from src.models.kilonerf.renderer.sampler import Sampler
from src.config import cfg
import time
import nerfacc


class VolumnRenderer(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.sampler = Sampler()
        self.net = net
        # self.raw_noise_std = cfg.task_arg.raw_noise_std
        # self.ert_threshold = cfg.sampler.ert_threshold
        self.white_bkgd = cfg.task_arg.white_bkgd
        # self.chunk_size = cfg.task_arg.renderer_chunk_size


    def forward(self, batch, is_training: bool = True):
        rays_o, viewdirs = batch['xyz'], batch['viewdirs']
        batch_size, N_rays, _ = rays_o.shape
        rays_o_flat, viewdirs_flat = rays_o.view(-1, 3), viewdirs.view(-1, 3)
        num_total_rays = batch_size * N_rays

        # call sampler to get sampling regions
        with torch.no_grad():
            ray_indices, t_start, t_ends = self.sampler(rays_o_flat, viewdirs_flat)

        # define callback function
        def rgb_alpha_fn(t_starts, t_ends, ray_indices):
            t_mid = (t_start + t_ends) / 2.0
            pts = rays_o_flat[ray_indices] + viewdirs_flat[ray_indices] * t_mid.unsqueeze(-1)

            # query network
            rgb_chunk, density_chunk = self.net(pts.unsqueeze(1), viewdirs_flat[ray_indices])

            # calculate alpha
            delta = (t_ends - t_starts).unsqueeze(-1)
            alpha = 1.0 - torch.exp(-torch.relu(density_chunk) * delta)

            return torch.cat([torch.sigmoid(rgb_chunk), alpha], dim = -1)

        # call rendering function of nerfacc
        colors, opacities, depths, _ = nerfacc.rendering(
            ray_indices = ray_indices,
            t_starts = t_starts,
            t_ends = t_ends,
            n_rays = total_num_rays,
            rgb_alpha_fn = rgb_alpha_fn
        )

        # deal with background
        colors += (1.0 - opacities) * (1.0 if self.white_bkgd else 0.0)

        image = {
            'rgb_map': colors.view(batch_size, N_rays, 3),
            'depth_map': depths.view(batch_size, N_rays, 1),
            'alpha_map': opacities.view(batch_size, N_rays, 1)
        }
        return image

    # def forward(self, batch, is_training: bool = True, verbose: bool = False):
    #     """
    #     Performs volumn rendering on raw NeRF output

    #     @param batch: (batch_size, N_rays, N_samples, 4) raw output from the model (rgb, density)
    #     @return: Dictionary with rendered RGB values and depth values
    #     """
    #     # shape: (batch_size, N_rays, 3)
    #     rays_o, viewdirs = batch['xyz'], batch['viewdirs']
    #     batch_size, N_rays, _ = rays_o.shape
    #     total_num_rays = batch_size * N_rays
    #     device = rays_o.device
    #     # flatten a batch
    #     rays_o = rays_o.view(-1, 3)
    #     viewdirs = viewdirs.view(-1, 3)

    #     # --- timer ---
    #     if verbose is True:
    #         torch.cuda.synchronize()
    #         time_start = time.time()

    #     # 1. Use Sampler to get initial z_vals, Sampler.forward will return z_vals(batch_size * N_rays, N_samples), which represent all the potential sample regions
    #     # shape of z_vals: (batch_size * N_rays, N_samples)
    #     _, z_vals = self.sampler(batch, is_training)

    #     # --- timer ---
    #     if verbose is True:
    #         torch.cuda.synchronize()
    #         time_sampler_end = time.time()
    #     # -------------
        
    #     # 2. Initialize color, depth and T accumulation
    #     C_acc = torch.zeros((total_num_rays, 3), device = device)
    #     D_acc = torch.zeros((total_num_rays, 1), device = device)
    #     T_acc = torch.ones((total_num_rays, 1), device = device)
    #     A_acc = torch.zeros((total_num_rays, 1), device = device)

    #     active_rays_mask = torch.ones(total_num_rays, dtype = torch.bool, device = device)

    #     # 3. begin ray marching loop
    #     # shape of z_vals is (batch_size * N_rays, N_samples), march in the sample dimension
    #     for i in range(0, z_vals.shape[1] - 1, self.chunk_size):
    #         # if all rays have terminated
    #         if not active_rays_mask.any():
    #             break

    #         start_idx = i
    #         end_idx = min(i + self.chunk_size, z_vals.shape[1] - 1)
    #         if start_idx >= end_idx:
    #             continue
            
    #         # a. only sample for active rays
    #         active_o = rays_o[active_rays_mask]
    #         active_v = viewdirs[active_rays_mask]
    #         z_start = z_vals[active_rays_mask, start_idx : end_idx]
    #         z_end = z_vals[active_rays_mask, start_idx + 1 : end_idx + 1]

    #         # sample for middle regions
    #         t_mid = (z_start + z_end) * 0.5
    #         delta = (z_end - z_start).unsqueeze(-1)
    #         # shape: (N_active_rays, N_chunk_samples, 3)
    #         pts_chunk = active_o.unsqueeze(1) + active_v.unsqueeze(1) * t_mid.unsqueeze(-1)

    #         # b. query network in batch
    #         # shape of pts_chunk: (batch_size * N_rays, renderer_chunk_size, 3)
    #         # shape of active_v: (batch_size * N_rays, 3)
    #         rgb_chunk, density_chunk = self.net(pts_chunk, active_v)
            
    #         # add Gaussian noise to density if specified and only during trainings
    #         if self.raw_noise_std > 0 and is_training:
    #             density_chunk = self.add_noise(density_chunk)

    #         # c. render in a single step
    #         sigma_chunk = torch.relu(density_chunk)
    #         T_step_chunk = torch.exp(-sigma_chunk * delta)
    #         alpha_chunk = 1.0 - T_step_chunk

    #         incoming_T = T_acc[active_rays_mask].unsqueeze(1)
    #         T_acc_in_chunk = torch.cumprod(
    #             torch.cat([incoming_T, T_step_chunk], dim = 1), dim = 1
    #         )[:, :-1, :]

    #         weights_chunk = T_acc_in_chunk * alpha_chunk

    #         rgb_map_chunk = torch.sum(weights_chunk * torch.sigmoid(rgb_chunk), dim = 1)
    #         depth_map_chunk = torch.sum(weights_chunk.squeeze(-1) * t_mid, dim = 1).unsqueeze(-1)
    #         alpha_map_chunk = torch.sum(weights_chunk, dim = 1)

    #         # update color and occupancy
    #         C_acc[active_rays_mask] += rgb_map_chunk
    #         D_acc[active_rays_mask] += depth_map_chunk
    #         A_acc[active_rays_mask] += alpha_map_chunk
    #         T_acc[active_rays_mask] *= torch.prod(T_step_chunk, dim = 1)

    #         # d/e. Early Ray Termination
    #         terminated_mask_in_active = T_acc[active_rays_mask] < self.ert_threshold
    #         current_active_indices = torch.where(active_rays_mask)[0]
    #         rays_to_terminate_indices = current_active_indices[terminated_mask_in_active.squeeze(-1)]

    #         if rays_to_terminate_indices.numel() > 0:
    #             active_rays_mask[rays_to_terminate_indices] = False

    #     # --- timer ---
    #     if verbose is True:
    #         torch.cuda.synchronize()
    #         time_renderer_end = time.time()

    #         sampler_time = time_sampler_end - time_start
    #         renderer_time = time_renderer_end - time_sampler_end
    #         print(f"\n[PROFILING] Sampler Time: {sampler_time:.4f}s | Renderer Loop Time: {renderer_time:.4f}s\n")
    #     # -------------


    #     if self.white_bkgd > 0:
    #         C_acc += T_acc

    #     D_acc = D_acc + T_acc * self.sampler.far
    #     image = {}
    #     image['rgb_map'] = C_acc.view(batch_size, N_rays, 3)
    #     image['depth_map'] = D_acc.view(batch_size, N_rays, 1)
    #     image['alpha_map'] = A_acc.view(batch_size, N_rays, 1)
    #     return image


    def add_noise(self, raw_density):
        """
        This function is responsible for adding noise to the raw density values.

        @param raw_density: The raw density values.
        @return: The raw density values with added noise.

        """
        noise = torch.randn(raw_density.shape, device = raw_density.device) * self.raw_noise_std
        raw_density = raw_density + noise
        return raw_density
