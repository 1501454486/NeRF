import torch
import torch.nn as nn
from src.models.kilonerf.renderer.sampler import Sampler


class VolumnRenderer():
    def __init__(self, sampler, net):
        super().__init__()
        self.sampler = sampler
        self.net = net
        self.raw_noise_std = cfg.task_arg.raw_noise_std
        self.ert_threshold = cfg.sampler.ert_threshold


    def render(self, batch, is_training: bool = True):
        """
        Performs volumn rendering on raw NeRF output

        @param batch: (N_rays, N_samples, 4) raw output from the model (rgb, density)
        @return: Dictionary with rendered RGB values and depth values
        """
        # shape: (N_rays, 3)
        rays_o, viewdirs = batch['xyz'].squeeze(0), batch['viewdirs'].squeeze(0)
        N_rays = rays_o.shape[0]
        device = rays_o.device

        # 1. Use Sampler to get initial z_vals, Sampler.forward will return z_vals(N_rays, N_samples), which represent all the potential sample regions
        _, z_vals = self.sampler(batch, is_training)

        # 2. Initialize color, depth and T accumulation
        C_acc = torch.zeros((N_rays, 3), device = device)
        D_acc = torch.zeros((N_rays, 1), device = device)
        T_acc = torch.ones((N_rays, 1), device = device)
        active_rays_mask = torch.ones(N_rays, dtype = torch.bool, device = device)

        # 3. begin ray marching loop
        # shape of z_vals is (N_rays, N_samples), march in the sample dimension
        for i in range(z_vals.shape[1] - 1):
            # if all rays have terminated
            if not active_rays_mask.any():
                break
            
            # a. only sample for active rays
            active_z_vals_start = z_vals[active_rays_mask, i]
            active_z_vals_end = z_vals[active_rays_mask, i + 1]

            # sample for middle regions
            t_mid = (active_z_vals_start + active_z_vals_end) * 0.5
            pts = rays_o[active_rays_mask] + viewdirs[active_rays_mask] * t_mid.unsqueeze(-1)
            delta = (active_z_vals_end - active_z_vals_start).unsqueeze(-1)

            # b. query network in batch
            rgb, density = self.net(pts.unsqueeze(1), viewdirs[active_rays_mask])
            rgb = rgb.squeeze(1)
            
            # add Gaussian noise to density if specified and only during trainings
            if self.raw_noise_std > 0 and is_training:
                density = add_noise(density)

            # c. render in a single step
            sigma = torch.relu(density)
            T_step = torch.exp(-sigma * delta)
            alpha = 1.0 - T_step

            weight = T_acc[active_rays_mask] * alpha

            # update color and occupancy
            C_acc[active_rays_mask] += weight * torch.sigmoid(rgb)
            D_acc[active_rays_mask] += weight * t_mid.unsqueeze(-1)
            T_acc[active_rays_mask] *= T_step

            # d/e. check if terminated and compression
            current_active_indices = torch.where(active_rays_mask)[0]
            terminated_mask = T_acc[active_rays_mask] < self.ert_threshold

            if terminated_mask.any():
                active_rays_mask[current_active_indices[terminated_mask.squeeze(-1)]] = False

        if self.white_bkgd > 0:
            C_acc += T_acc

        D_acc = D_acc + T_acc * self.sampler.far
        
        image = {}
        image['rgb_map'] = C_acc
        image['depth_map'] = D_acc
        return image


    def add_noise(self, raw_density):
        """
        This function is responsible for adding noise to the raw density values.

        @param raw_density: The raw density values.
        @return: The raw density values with added noise.

        """
        noise = torch.randn(raw_density.shape, device = raw_density.device) * self.raw_noise_std
        raw_density = raw_density + noise
        return raw_density