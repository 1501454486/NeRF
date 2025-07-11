import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import os
import nerfacc

from src.config import cfg


class Renderer(nn.Module):
    def __init__(self, net):
        """
        This function is responsible for defining the rendering parameters, including the number of samples, the step size, and the background color.

        @param net: The neural network that will be used for rendering.

        Write your codes here.
        """
        super().__init__()
        self.net = net
        self.N_samples = cfg.task_arg.N_samples
        self.N_importance = cfg.task_arg.N_importance
        self.steps = None
        self.white_bkgd = cfg.task_arg.white_bkgd
        self.perturb = cfg.task_arg.perturb
        self.raw_noise_std = cfg.task_arg.raw_noise_std
        self.near = cfg.task_arg.near
        self.far = cfg.task_arg.far

        self.coarse_samples = None
        self.fine_samples = None
        self.t_bins = None
        
        self.chunk_size = cfg.task_arg.chunk_size if cfg.task_arg.chunk_size > 0 else 1024

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.occ_grid_resolution = [grid * cfg.task_arg.occ_factor for grid in cfg.task_arg.grid_resolution] # e.g., [256, 256, 256]
        self.scene_aabb = torch.tensor(cfg.task_arg.aabb['min'] + cfg.task_arg.aabb['max'], dtype=torch.float32, device=self.device)
        self.estimator = nerfacc.OccGridEstimator(
            roi_aabb = self.scene_aabb,
            resolution = self.occ_grid_resolution,
            levels = 1
        ).to(self.device)
        
        print("Building occupancy grid...")
        self._build_occ_grid()

    def forward(self, batch, is_training: bool = True):
        """
        This function is responsible for rendering the output of the model, which includes the RGB values and the depth values.
        1. 输入是从dataloader中获得的batch, 包括光线方向rays(1024, 3)和rgb(1024, 3)
        2. 对rays方向上near到far, 采样N_samples, 得到 N evenly-spaced bins
        3. 在 N evenly-sapced bins 的每个区间抽一个, 共64, 然后计算 step_sizes, 输入coarse net

        @param batch: A batch from Dataloader, (rays, rgb)
        @return: A dictionary containing the rendered RGB values and depth values.

        Write your codes here.
        """

        rays_o, viewdirs = batch['xyz'], batch['viewdirs']
        batch_size, N_rays, _ = rays_o.shape
        rays_o_flat, viewdirs_flat = rays_o.view(-1, 3), viewdirs.view(-1, 3)
        num_total_rays = batch_size * N_rays

        def sigma_fn(
            t_starts: Tensor, t_ends:Tensor, ray_indices: Tensor
        ) -> Tensor:
            """ Define how to query density for the estimator."""

            t_origins = rays_o_flat[ray_indices]  # (n_samples, 3)
            t_dirs = viewdirs_flat[ray_indices]  # (n_samples, 3)
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            sigmas = self.net(positions.unsqueeze(1), viewdirs_flat[ray_indices])[:, -1]
            return torch.relu(sigmas)  # (n_samples,)

        # Perform sampling with ESS
        # ray_indices: (N_samples, )
        # t_starts: (N_samples, )
        # t_ends: (N_samples, )
        ray_indices, t_starts, t_ends = self.estimator.sampling(
            rays_o = rays_o_flat,
            rays_d = viewdirs_flat,
            sigma_fn = sigma_fn,
            near_plane = self.near,
            far_plane = self.far,
        )

        # define callback function
        def rgb_alpha_fn(t_starts, t_ends, ray_indices):
            t_mid = (t_starts + t_ends) / 2.0
            # shape of pts: (N_samples, 3)
            pts = rays_o_flat[ray_indices] + viewdirs_flat[ray_indices] * t_mid.unsqueeze(-1)
            # query network
            outputs = self.net(pts, viewdirs_flat[ray_indices])
            # rgb_chunk: (num_points, 1, 3)
            # density_chunk: (num_points, 1, 1)
            rgb_chunk = torch.sigmoid(outputs[:, :-1])
            # calculate alpha
            delta = (t_ends - t_starts)
            alpha = 1.0 - torch.exp(-torch.relu(outputs[:, -1]) * delta)
            return rgb_chunk, alpha

        # call rendering function of nerfacc
        # colors: (N_rays, 3)
        # opacities: (N_rays, 1)
        # depths: (N_rays, 1)
        colors, opacities, depths, _ = nerfacc.rendering(
            t_starts = t_starts,
            t_ends = t_ends,
            ray_indices = ray_indices,
            n_rays = batch_size * N_rays,
            rgb_alpha_fn = rgb_alpha_fn,
        )

        # deal with background
        colors += (1.0 - opacities) * (1.0 if cfg.task_arg.white_bkgd else 0.0)

        image = {
            'fine_rgb_map': colors.view(batch_size, N_rays, 3),
            'fine_depth_map': depths.view(batch_size, N_rays, 1),
            'alpha_map': opacities.view(batch_size, N_rays, 1)
        }
        return image



        # coarse_rgb_map, coarse_depth_map, w_i_coarse, _ = self.render_coarse(batch, is_training)
        # fine_rgb_map, fine_depth_map, fine_alpha_map = self.render_fine(batch, w_i_coarse, is_training)

        # image = {}
        # image['coarse_rgb_map'] = coarse_rgb_map
        # image['coarse_depth_map'] = coarse_depth_map
        # image['fine_rgb_map'] = fine_rgb_map
        # image['fine_depth_map'] = fine_depth_map
        # image['alpha_map'] = fine_alpha_map

        # return image


    @torch.no_grad()
    def _build_occ_grid(self, occ_thre: float = 0.01):
        """
        【修正版本】
        使用标准的 PyTorch 函数来生成网格坐标，以兼容所有 nerfacc 版本。
        """
        print("Baking occupancy grid with PyTorch standard functions...")

        # 步骤 1: 使用标准的 PyTorch 函数来生成网格坐标
        # 从 estimator 的 aabbs 属性获取边界。aabbs 形状为 [1, 6]
        min_bound = self.estimator.aabbs[0, :3]
        max_bound = self.estimator.aabbs[0, 3:]
        resolution = self.estimator.resolution

        # 为 x, y, z 三个轴创建线性间隔点
        x_coords = torch.linspace(min_bound[0], max_bound[0], resolution[0], device=self.device)
        y_coords = torch.linspace(min_bound[1], max_bound[1], resolution[1], device=self.device)
        z_coords = torch.linspace(min_bound[2], max_bound[2], resolution[2], device=self.device)
        
        # 使用 meshgrid 创建三维坐标网格
        grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        
        # 将坐标堆叠并展平为 (N, 3) 的形状
        grid_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(-1, 3)

        # 步骤 2: 标准的 PyTorch 推理流程，分块查询 self.net 获取密度 (此部分逻辑不变)
        def occ_eval_fn(x):
            dummy_viewdirs = torch.zeros_like(x)
            _, sigma = self.net(x, dummy_viewdirs)
            return sigma.squeeze()
        
        sigmas = []
        from tqdm import tqdm
        for i in tqdm(range(0, grid_coords.shape[0], self.chunk_size), desc="Baking Occupancy Grid"):
            coords_chunk = grid_coords[i:i+self.chunk_size]
            sigmas_chunk = occ_eval_fn(coords_chunk)
            sigmas.append(sigmas_chunk)
        sigmas = torch.cat(sigmas)

        # 步骤 3: 直接设置 nerfacc.OccGridEstimator 的官方公开属性 .binaries (此部分逻辑不变)
        binaries = (sigmas > occ_thre).view(self.estimator.resolution)
        self.estimator.binaries = binaries.unsqueeze(0)



    def render_coarse(self, batch, is_training: bool = True):
        """
        This function is responsible for rendering the coarse model.

        @param batch: A batch from Dataloader, (rays, rgb)
        @return: A dictionary containing the rendered RGB values and depth values.

        """
        xyz, viewdirs, gt_rgb = batch['xyz'], batch['viewdirs'], batch['gt_rgb']
        batch_size, num_rays = xyz.shape[0], xyz.shape[1]         # shape of xyz: (1, 1024, 3)
        near = self.near * torch.ones((num_rays, 1), device = xyz.device)  # shape: (1024, 1)
        far = self.far * torch.ones((num_rays, 1), device = xyz.device)  # shape: (1024, 1)

        samples = torch.linspace(0., 1., steps = self.N_samples, device = xyz.device)
        samples = near * (1 - samples) + far * samples # shape: (1024, 64)
        self.t_bins = samples.clone()  # save the t_bins for importance sampling later
        
        if self.perturb > 0 and is_training:
            samples = self.add_perturbation(samples)
        self.coarse_samples = samples

        delta = samples[:, 1:] - samples[:, :-1]  # shape: (1024, 63)
        inf_delta = torch.full((num_rays, 1), 1e10, dtype=samples.dtype, device=samples.device)
        delta = torch.cat([delta, inf_delta], dim=-1)   # shape: (1024, 64)

        # change the shape so that they can match
        xyz = xyz.unsqueeze(2).squeeze(0)                       # shape: (1024, 1, 3)
        viewdirs = viewdirs.squeeze(0)                          # shape: (1024, 3)
        # normalize viewdirs
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        pts = xyz + samples.unsqueeze(-1) * viewdirs.unsqueeze(1)     # shape: (1024, 64, 3)

        # Render the coarse model, output_flat: (1024, 64, 4)
        output_flat = self._run_network(pts, viewdirs)  # use the _run_network function to handle chunking

        density = output_flat[..., 3]   # shape: (1024, 64)
        raw_rgb = output_flat[..., :3]      # shape: (1024, 64, 3)

        # Add Gausssian noise to density(sigma) (only for training)
        if self.raw_noise_std > 0 and is_training:
            density = add_noise(density)    # shape: (1024, 64)

        # Apply ReLU to the density
        sigma = torch.relu(density)         # shape: (1024, 64)
        # Apply sigmoid to the RGB values
        rgb = torch.sigmoid(raw_rgb)        # shape: (1024, 64, 3)

        alpha = 1 - torch.exp(-sigma * delta)   # shape: (1024, 64)
        T_i = torch.cumprod((1 - alpha)[:, :-1], dim = -1)  # shape: (1024, 63)
        T_i = torch.cat([torch.ones((num_rays, 1), dtype=samples.dtype, device=samples.device), T_i], dim=-1)   # shape: (1024, 64)

        w_i = T_i * alpha   # shape: (1024, 64)
        
        rgb_map = torch.sum(w_i.unsqueeze(-1) * rgb, dim=-2)  # shape: (1024, 3)
        # accumulated alpha
        alpha_map = torch.sum(w_i, dim = -1, keepdim = True)

        # add background color if white_bkgd is True
        if self.white_bkgd > 0:
            acc_map = torch.sum(w_i, -1)
            rgb_map = rgb_map + (1. - acc_map.unsqueeze(-1))
        # depth value is the expected value of weight
        depth_map = torch.sum(w_i * samples, dim=-1)      # shape: (1024)

        return rgb_map, depth_map, w_i, alpha_map
        
    def render_fine(self, batch, weights, is_training: bool = True):
        """
        This function is responsible for rendering the fine model.

        @param batch: A batch from Dataloader, (rays, rgb)
        @param weights: The weights of the coarse model.
        @return: A dictionary containing the rendered RGB values and depth values.

        """
        xyz, viewdirs, gt_rgb = batch['xyz'], batch['viewdirs'], batch['gt_rgb']
        batch_size, num_rays = xyz.shape[0], xyz.shape[1]         # shape of xyz: (1, 1024, 3)
        near = self.near * torch.ones((num_rays, 1), device = xyz.device)  # shape: (1024, 1)
        far = self.far * torch.ones((num_rays, 1), device = xyz.device)  # shape: (1024, 1)

        coarse_samples = self.coarse_samples
        fine_samples = self.importance_sampling(weights)  # shape: (1024, N_importance)
        if self.perturb > 0 and is_training:
            fine_samples = self.add_perturbation(fine_samples)
        self.fine_samples = fine_samples

        # concatenate coarse and fine samples
        samples = torch.cat([coarse_samples, fine_samples], dim=-1)  # shape: (1024, N_samples + N_importance)
        samples, _ = torch.sort(samples, dim=-1)  # sort the samples

        delta = samples[:, 1:] - samples[:, :-1]  # shape: (1024, 191)
        inf_delta = torch.full((num_rays, 1), 1e10, dtype=samples.dtype, device=samples.device)
        delta = torch.cat([delta, inf_delta], dim=-1)   # shape: (1024, 192)

        # change the shape so that they can match
        xyz = xyz.unsqueeze(2).squeeze(0)                       # shape: (1024, 1, 3)
        viewdirs = viewdirs.squeeze(0)                          # shape: (1024, 3)
        # normalize viewdirs
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        pts = xyz + samples.unsqueeze(-1) * viewdirs.unsqueeze(1)     # shape: (1024, 192, 3)

        # Render the fine model, output_flat: (1024, 192, 4)
        output_flat = self._run_network(pts, viewdirs, "fine")  # use the _run_network function to handle chunking

        density = output_flat[..., 3]   # shape: (1024, 192)
        raw_rgb = output_flat[..., :3]      # shape: (1024, 192, 3)

        # Add Gausssian noise to density(sigma) (only for training)
        if self.raw_noise_std > 0 and is_training:
            density = add_noise(density)    # shape: (1024, 192)

        # Apply ReLU to the density
        sigma = torch.relu(density)         # shape: (1024, 192)
        # Apply sigmoid to the RGB values
        rgb = torch.sigmoid(raw_rgb)        # shape: (1024, 192, 3)

        alpha = 1 - torch.exp(-sigma * delta)   # shape: (1024, 192)
        T_i = torch.cumprod((1 - alpha)[:, :-1], dim = -1)  # shape: (1024, 191)
        T_i = torch.cat([torch.ones((num_rays, 1), dtype=samples.dtype, device=samples.device), T_i], dim=-1)   # shape: (1024, 192)

        w_i = T_i * alpha   # shape: (1024, 192)
        
        rgb_map = torch.sum(w_i.unsqueeze(-1) * rgb, dim=-2)  # shape: (1024, 3)
        # accumulated alpha
        alpha_map = torch.sum(w_i, dim = -1, keepdim = True)
        
        # add background color if white_bkgd is True
        if self.white_bkgd > 0:
            acc_map = torch.sum(w_i, -1)
            rgb_map = rgb_map + (1. - acc_map.unsqueeze(-1))
        # depth value is the expected value of weight
        depth_map = torch.sum(w_i * samples, dim=-1)      # shape: (1024)

        return rgb_map, depth_map, alpha_map

    def add_perturbation(self, samples):
        """
        This function is responsible for adding random perturbation to the sampling points.

        @param samples: The sampling points.
        @return: The perturbed sampling points.

        """
        # add random perturbation to the sampling points
        mid = 0.5 * (samples[:, 1:] + samples[:, :-1])  # shape: (1024, 63)
        upper = torch.cat([mid, samples[:, -1:]], dim=-1)  # shape: (1024, 64)
        lower = torch.cat([samples[:, :1], mid], dim=-1)   # shape: (1024, 64)
        samples = lower + (upper - lower) * torch.rand(samples.shape, dtype=samples.dtype, device=samples.device)
        return samples

    def add_noise(self, raw_density):
        """
        This function is responsible for adding noise to the raw density values.

        @param raw_density: The raw density values.
        @return: The raw density values with added noise.

        """
        noise = torch.randn(raw_density.shape, device = raw_density.device) * self.raw_noise_std
        raw_density = raw_density + noise
        return raw_density

    def importance_sampling(self, weights):
        """
        This function is responsible for performing importance sampling on the coarse samples.

        @param weights: The weights of the coarse model.
        @return: The fine samples.

        """
        # 1. turn weights into pdf and cdf
        batch_size, n_bins = weights.shape
        # normalize weights, shape: (batch_size, n_bins)
        pdf = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-10)
        cdf = torch.cumsum(pdf, dim=-1)  # shape: (batch_size, n_bins)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)

        # 2. generate random numbers
        u = torch.rand((batch_size, self.N_importance), device=weights.device)
        # shape: (batch_size, N_importance)
        bin_indices = torch.searchsorted(cdf, u, right=True) - 1
        # shape: (batch_size, N_importance)
        # IMPORTANT! Clamp the bin indices to be within the range of [0, n_bins-1]
        bin_indices = torch.clamp(bin_indices, min=0, max=self.t_bins.shape[1] - 1)

        # 3. sample in the given bins
        bins_lower = torch.gather(self.t_bins, 1, bin_indices)
        bins_upper = torch.gather(self.t_bins, 1, (bin_indices + 1).clamp(max=self.t_bins.shape[1] - 1))
        u_samples = torch.rand(batch_size, self.N_importance, device = weights.device)
        samples = bins_lower + (bins_upper - bins_lower) * u_samples
        return samples

    def _run_network(self, pts, viewdirs, model_type="coarse"):
        """
        This function is responsible for running the network on the given points and view directions in chunk_size.

        @param pts: The points to be rendered.
        @param viewdirs: The view directions.
        @param model_type: The type of the model, either "coarse" or "fine".
        @return: The output of the network.

        """
        # pts: [N_rays, N_samples, 3]
        # viewdirs: [N_rays, 3]
        N_rays = pts.shape[0]
        
        all_outputs = []
        # deal with chunk_size rays at a time
        for i in range(0, N_rays, self.chunk_size):
            chunk_pts = pts[i:i+self.chunk_size]        # shape: [chunk_size, N_samples, 3]
            chunk_viewdirs = viewdirs[i:i+self.chunk_size]      # shape: [chunk_size, 3]

            if model_type == "fine":
                chunk_output = self.net(chunk_pts, chunk_viewdirs, "fine")
            else:
                chunk_output = self.net(chunk_pts, chunk_viewdirs)
            
            all_outputs.append(chunk_output)
            
        # merge all outputs
        output = torch.cat(all_outputs, dim=0)
        
        return output