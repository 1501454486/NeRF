import numpy as np
import torch
from src.config import cfg


class Renderer:
    def __init__(self, net):
        """
        This function is responsible for defining the rendering parameters, including the number of samples, the step size, and the background color.

        @param net: The neural network that will be used for rendering.

        Write your codes here.
        """
        self.net = net
        self.N_samples = cfg.task_arg.N_samples
        self.N_importance = cfg.task_arg.N_importance
        self.steps = None
        self.white_bkgd = cfg.task_arg.white_bkgd
        self.perturb = cfg.task_arg.perturb
        self.raw_noise_std = cfg.task_arg.raw_noise_std
        self.near = 2
        self.far = 6

        self.coarse_samples = None
        self.fine_samples = None
        self.t_bins = None

    def render(self, batch):
        """
        This function is responsible for rendering the output of the model, which includes the RGB values and the depth values.
        1. 输入是从dataloader中获得的batch, 包括光线方向rays(1024, 3)和rgb(1024, 3)
        2. 对rays方向上near到far, 采样N_samples, 得到 N evenly-spaced bins
        3. 在 N evenly-sapced bins 的每个区间抽一个, 共64, 然后计算 step_sizes, 输入coarse net

        @param batch: A batch from Dataloader, (rays, rgb)
        @return: A dictionary containing the rendered RGB values and depth values.

        Write your codes here.
        """
        coarse_rgb_map, coarse_depth_map, w_i_coarse = self.render_coarse(batch)
        fine_rgb_map, fine_depth_map = self.render_fine(batch, w_i_coarse)

        image = {}
        image['coarse_rgb_map'] = coarse_rgb_map
        image['coarse_depth_map'] = coarse_depth_map
        image['fine_rgb_map'] = fine_rgb_map
        image['fine_depth_map'] = fine_depth_map

        return image

    def render_coarse(self, batch):
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
        
        if self.perturb > 0:
            samples = self.add_perturbation(samples)
        self.coarse_samples = samples

        delta = samples[:, 1:] - samples[:, :-1]  # shape: (1024, 63)
        inf_delta = torch.full((num_rays, 1), 1e10, dtype=samples.dtype, device=samples.device)
        delta = torch.cat([delta, inf_delta], dim=-1)   # shape: (1024, 64)

        # change the shape so that they can match
        xyz = xyz.unsqueeze(2).squeeze(0)                       # shape: (1024, 1, 3)
        viewdirs = viewdirs.squeeze(0)                          # shape: (1024, 3)
        pts = xyz + samples.unsqueeze(-1) * viewdirs.unsqueeze(1)     # shape: (1024, 64, 3)
        # normalize viewdirs
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        # Render the coarse model, output_flat: (1024, 64, 4)
        output_flat = self.net(pts, viewdirs)

        density = output_flat[..., 3]   # shape: (1024, 64)
        raw_rgb = output_flat[..., :3]      # shape: (1024, 64, 3)

        # Add Gausssian noise to density(sigma) (only for training)
        if self.raw_noise_std > 0:
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

        # depth value is the expected value of weight
        depth_map = torch.sum(w_i * samples, dim=-1)      # shape: (1024)

        return rgb_map, depth_map, w_i
        
    def render_fine(self, batch, weights):
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
        if self.perturb > 0:
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
        pts = xyz + samples.unsqueeze(-1) * viewdirs.unsqueeze(1)     # shape: (1024, 192, 3)
        # normalize viewdirs
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        # Render the fine model, output_flat: (1024, 192, 4)
        output_flat = self.net(pts, viewdirs, "fine")

        density = output_flat[..., 3]   # shape: (1024, 192)
        raw_rgb = output_flat[..., :3]      # shape: (1024, 192, 3)

        # Add Gausssian noise to density(sigma) (only for training)
        if self.raw_noise_std > 0:
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

        # depth value is the expected value of weight
        depth_map = torch.sum(w_i * samples, dim=-1)      # shape: (1024)

        return rgb_map, depth_map

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