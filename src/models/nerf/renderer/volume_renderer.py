import numpy as np
import torch
import os
import natsort  # 用于自然排序
import glob

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
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
        
        self.chunk_size = cfg.task_arg.chunk_size if cfg.task_arg.chunk_size > 0 else 1024

    def render(self, batch, is_training: bool = True):
        """
        This function is responsible for rendering the output of the model, which includes the RGB values and the depth values.
        1. 输入是从dataloader中获得的batch, 包括光线方向rays(1024, 3)和rgb(1024, 3)
        2. 对rays方向上near到far, 采样N_samples, 得到 N evenly-spaced bins
        3. 在 N evenly-sapced bins 的每个区间抽一个, 共64, 然后计算 step_sizes, 输入coarse net

        @param batch: A batch from Dataloader, (rays, rgb)
        @return: A dictionary containing the rendered RGB values and depth values.

        Write your codes here.
        """
        coarse_rgb_map, coarse_depth_map, w_i_coarse = self.render_coarse(batch, is_training)
        fine_rgb_map, fine_depth_map = self.render_fine(batch, w_i_coarse, is_training)

        image = {}
        image['coarse_rgb_map'] = coarse_rgb_map
        image['coarse_depth_map'] = coarse_depth_map
        image['fine_rgb_map'] = fine_rgb_map
        image['fine_depth_map'] = fine_depth_map

        return image

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
        output_flat = self.run_network(pts, viewdirs)  # use the run_network function to handle chunking

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
        # add background color if white_bkgd is True
        if self.white_bkgd > 0:
            acc_map = torch.sum(w_i, -1)
            rgb_map = rgb_map + (1. - acc_map.unsqueeze(-1))
        # depth value is the expected value of weight
        depth_map = torch.sum(w_i * samples, dim=-1)      # shape: (1024)

        return rgb_map, depth_map, w_i
        
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
        output_flat = self.run_network(pts, viewdirs, "fine")  # use the run_network function to handle chunking

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
        # add background color if white_bkgd is True
        if self.white_bkgd > 0:
            acc_map = torch.sum(w_i, -1)
            rgb_map = rgb_map + (1. - acc_map.unsqueeze(-1))
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

    def run_network(self, pts, viewdirs, model_type="coarse"):
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

    def render_video_from_images(self, image_folder: str, fps: int = 24):
        """
        从图片序列渲染视频。
        此版本硬编码为仅查找 'view*_pred.png' 格式的图片。
        """
        print(f"开始从文件夹 '{image_folder}' 渲染视频...")
        print("模式: 将只使用 'view*_pred.png' 格式的图片。")

        output_dir = os.path.join(cfg.result_dir, "videos")
        os.makedirs(output_dir, exist_ok=True)
        video_path = os.path.join(output_dir, "view_pred_video.mp4")

        # 1. 定义文件搜索模式
        # '*' 是一个通配符, 它会匹配 'view' 和 '_pred.png' 之间的任何字符
        search_pattern = os.path.join(image_folder, 'view*_pred.png')

        # 2. 使用 glob 查找所有匹配模式的图片文件
        # glob.glob 会返回一个包含完整路径的列表
        found_files = glob.glob(search_pattern)

        # 3. 使用 natsort 对找到的文件路径进行自然排序
        # 确保 view2.png 在 view10.png 之前
        full_image_paths = natsort.natsorted(found_files)
        
        if not full_image_paths:
            print(f"错误：在文件夹 '{image_folder}' 中没有找到符合 'view*_pred.png' 模式的图片。")
            return

        print(f"找到了 {len(full_image_paths)} 张匹配的图片进行渲染。")

        # 4. 使用 MoviePy 基于处理后的完整路径列表创建剪辑
        clip = ImageSequenceClip(full_image_paths, fps=fps)

        # 5. 写入视频文件
        print(f"正在将视频写入到 '{video_path}'...")
        try:
            clip.write_videofile(video_path, codec='libx264', logger='bar')
            print(f"视频渲染成功！已保存在: {video_path}")
        except Exception as e:
            print(f"视频渲染失败: {e}")