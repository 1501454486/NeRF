import torch
from torch import Tensor
import torch.nn as nn
# from src.models.kilonerf.renderer.sampler import Sampler
from src.config import cfg
import time
import nerfacc
import os
from tqdm import tqdm


class VolumnRenderer(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.sampler = Sampler()
        self.net = net
        self.white_bkgd = cfg.task_arg.white_bkgd

        self.occ_threshold = cfg.sampler.occ_threshold
        self.result_dir = cfg.result_dir
        grid_resolution = cfg.task_arg.grid_resolution
        occ_factor = cfg.sampler.occ_factor
        self.occ_grid_resolution = [grid * occ_factor for grid in grid_resolution] # e.g., [256, 256, 256]
        self.scene_aabb = torch.tensor(cfg.task_arg.aabb['min'] + cfg.task_arg.aabb['max'], dtype=torch.float32, device=self.device)
        self.register_buffer('aabb_min', torch.tensor(cfg.task_arg.aabb['min'], device = self.device))
        self.register_buffer('aabb_max', torch.tensor(cfg.task_arg.aabb['max'], device = self.device))
        self.register_buffer('occ_grid_resolution_tensor', torch.tensor(self.occ_grid_resolution, dtype = torch.float32, device = self.device))

        # grid path
        res_str = f"{self.occ_grid_resolution[0]}_{self.occ_grid_resolution[1]}_{self.occ_grid_resolution[2]}"
        self.grid_path = os.path.join(self.result_dir, f'occ_grid_res{res_str}_thresh{self.occ_threshold}.pth')
        self.occ_grid = None
        if os.path.exists(self.grid_path):
            self._load_grid()
        else:
            self._evaluate_grid()
            self._save_grid()
        
        # Initialize an estimator for the grid
        self.estimator = nerfacc.OccGridEstimator(
            roi_aabb = self.scene_aabb,
            resolution = self.occ_grid_resolution,
            levels = 1
        ).to(self.device)

        # load grid into estimator
        self.estimator.binaries = self.occ_grid.unsqueeze(0)
        self.render_step_size = cfg.sampler.render_step_size
        self.early_stop_eps = cfg.sampler.ert_threshold


    def forward(self, batch, is_training: bool = True):
        rays_o, viewdirs = batch['xyz'], batch['viewdirs']
        batch_size, N_rays, _ = rays_o.shape
        rays_o_flat, viewdirs_flat = rays_o.view(-1, 3), viewdirs.view(-1, 3)
        num_total_rays = batch_size * N_rays

        # ===================== DEBUG CODE START =====================
        print("\n\n--- NERFACC DEBUG INFO ---")
        # 检查 estimator 对象本身及其关键属性
        print(f"Estimator object: {self.estimator}")
        if hasattr(self.estimator, 'roi_aabb'):
            print(f"ROI AABB shape: {self.estimator.roi_aabb.shape}, dtype: {self.estimator.roi_aabb.dtype}")
            print(f"ROI AABB value: {self.estimator.roi_aabb}")
        if hasattr(self.estimator, 'resolution'):
            print(f"Resolution value: {self.estimator.resolution}, type: {type(self.estimator.resolution)}")
        if hasattr(self.estimator, 'binaries'):
            print(f"Binaries shape: {self.estimator.binaries.shape}, dtype: {self.estimator.binaries.dtype}")
            print(f"Binaries num elements: {self.estimator.binaries.numel()}")

        # 检查传入的光线数据
        print(f"Rays_o_flat shape: {rays_o_flat.shape}, dtype: {rays_o_flat.dtype}")
        print(f"Viewdirs_flat shape: {viewdirs_flat.shape}, dtype: {viewdirs_flat.dtype}")

        # 检查张量是否在内存中连续，这对于一些CUDA操作很重要
        print(f"Is rays_o_flat contiguous? {rays_o_flat.is_contiguous()}")
        print(f"Is viewdirs_flat contiguous? {viewdirs_flat.is_contiguous()}")

        print(f"Scene AABB value: {self.scene_aabb}")
        print("--------------------------\n\n")
        # ====================== DEBUG CODE END ======================

        def sigma_fn(
            t_starts: Tensor, t_ends:Tensor, ray_indices: Tensor
        ) -> Tensor:
            """ Define how to query density for the estimator."""
            ############### DEBUG #################
            print("shape of t_starts: ", t_starts.shape)
            print("shape of t_ends: ", t_ends.shape)
            print("shape of ray_indices: ", ray_indices.shape)
            print("num_total_rays: ", num_total_rays)

            t_origins = rays_o_flat[ray_indices]  # (n_samples, 3)
            t_dirs = viewdirs_flat[ray_indices]  # (n_samples, 3)
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            _, sigmas = self.net(positions.unsqueeze(1), viewdirs_flat[ray_indices])
            return torch.relu(sigmas.view(-1))  # (n_samples,)

        # Perform sampling with ESS
        # ray_indices: (N_samplesm, )
        # t_starts: (N_samples, )
        # t_ends: (N_samples, )
        ray_indices, t_starts, t_ends = self.estimator.sampling(
            rays_o = rays_o_flat,
            rays_d = viewdirs_flat,
            sigma_fn = sigma_fn,
            render_step_size = self.render_step_size,
            early_stop_eps = self.early_stop_eps,
        )

        # define callback function
        def rgb_alpha_fn(t_starts, t_ends, ray_indices):
            t_mid = (t_starts + t_ends) / 2.0
            pts = rays_o_flat[ray_indices] + viewdirs_flat[ray_indices] * t_mid.unsqueeze(-1)
            # query network
            rgb_chunk, density_chunk = self.net(pts.unsqueeze(1), viewdirs_flat[ray_indices])
            # rgb_chunk: (num_points, 1, 3)
            # density_chunk: (num_points, 1, 1)
            rgb_chunk = torch.sigmoid(rgb_chunk.view(-1, 3))
            # calculate alpha
            delta = (t_ends - t_starts)
            alpha = 1.0 - torch.exp(-torch.relu(density_chunk.view(-1)) * delta)
            return rgb_chunk, alpha

        # call rendering function of nerfacc
        # colors: (N_rays, 3)
        # opacities: (N_rays, 1)
        # depths: (N_rays, 1)
        colors, opacities, depths, extras = nerfacc.rendering(
            t_starts = t_starts,
            t_ends = t_ends,
            ray_indices = ray_indices,
            n_rays = N_rays,
            rgb_alpha_fn = rgb_alpha_fn,
        )

        # deal with background
        colors += (1.0 - opacities) * (1.0 if self.white_bkgd else 0.0)

        image = {
            'rgb_map': colors.view(batch_size, N_rays, 3),
            'depth_map': depths.view(batch_size, N_rays, 1),
            'alpha_map': extras['alphas'].view(batch_size, N_rays, 1)
        }
        return image


    def _save_grid(self):
        """
        Save grid to the disk
        """
        os.makedirs(self.result_dir, exist_ok = True)
        torch.save(self.occ_grid, self.grid_path)
        print(f"Occupancy grid saved to {self.grid_path}")

    
    def _load_grid(self):
        """
        Load grid from disk
        """
        loaded_obj = torch.load(self.grid_path, map_location = self.device)
        print(f"--- DEBUG: Loaded object from {self.grid_path} ---")
        print(f"Type of loaded object: {type(loaded_obj)}")
        if isinstance(loaded_obj, torch.Tensor):
            print(f"Initial shape of loaded tensor: {loaded_obj.shape}")
            self.occ_grid = loaded_obj.squeeze(0)
            print(f"Shape after squeeze(0): {self.occ_grid.shape}")
        else:
            # 如果加载的是字典或其他类型，这里会打印出来
            print(f"Loaded object is not a tensor, content: {loaded_obj}")
            # 根据实际情况处理，例如: self.occ_grid = loaded_obj['grid'].squeeze(0)
            # 这里暂时先报错，以便我们知道发生了什么
            raise TypeError("Loaded occupancy grid is not a tensor!")
        print("--------------------------------------------------")

        print(f"Occupancy grid loaded from {self.grid_path}. shape: ", self.occ_grid.shape)

    @torch.no_grad()
    def _evaluate_grid(self, verbose: bool = False):
        """
        Use a teacher model (NeRF) to evaluate and generate grid.
        This process will traverse all the grids, and in each grid, it will generate sub_grid^3 to evaluate whether this grid is occupird or not.
        """
        teacher_model = make_network(cfg).cuda()
        load_network(teacher_model, cfg.teacher_model_dir)
        if verbose is True:
            print("teacher model: ", teacher_model)
        # initialize a bool grid with all False
        self.occ_grid = torch.zeros(self.occ_grid_resolution, dtype = torch.bool, device = self.device)

        # calculate size of every grid from aabb and resolution
        cell_size = (self.aabb_max - self.aabb_min) / self.occ_grid_resolution_tensor

        # calculate central coordinates of every cell
        x = torch.linspace(self.aabb_min[0], self.aabb_max[0], self.occ_grid_resolution[0] + 1, device = self.device)[:-1]
        y = torch.linspace(self.aabb_min[1], self.aabb_max[1], self.occ_grid_resolution[1] + 1, device = self.device)[:-1]
        z = torch.linspace(self.aabb_min[2], self.aabb_max[2], self.occ_grid_resolution[2] + 1, device = self.device)[:-1]
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing = 'ij')
        cell_centers = torch.stack([grid_x, grid_y, grid_z], dim = -1).to(self.device)
        cell_centers += cell_size / 2.0

        if verbose is True:
            print("shape of cell_centers: ", cell_centers.shape)

        # generate a 3x3x3 subgrid offset
        offsets = torch.stack(torch.meshgrid([
            torch.tensor([-0.25, 0, 0.25]),
            torch.tensor([-0.25, 0, 0.25]),
            torch.tensor([-0.25, 0, 0.25])
        ], indexing='ij'), dim=-1).view(-1, 3).to(self.device)
        # scale offsets to cell_size
        scaled_offsets = offsets * cell_size

        # flatten centers of all cells in order to batch
        flat_cell_centers = cell_centers.view(-1, 3)

        indices = torch.stack(torch.meshgrid(
            torch.arange(self.occ_grid_resolution[0], device = self.device),
            torch.arange(self.occ_grid_resolution[1], device = self.device),
            torch.arange(self.occ_grid_resolution[2], device = self.device),
            indexing='ij'
        ), dim=-1).view(-1, 3)

        for i in tqdm(range(0, flat_cell_centers.shape[0], self.chunk), desc="Evaluating Occupancy Grid"):
            # get centers of current block
            current_centers = flat_cell_centers[i:i + self.chunk]

            # generate 27 subsamples for current block
            # shape: (chunk_size, 27, 3)
            sample_points = current_centers.unsqueeze(1) + scaled_offsets.unsqueeze(0)

            # shape: (chunk_size * 27, 3)
            flat_sample_points = sample_points.view(-1, 3)

            # reshape to (1, N, 3)
            inputs = flat_sample_points.unsqueeze(0)

            # use teacher model to evaluate densities
            # densities is not related to viewdirs, so we use dummy viewdirs instead
            dummy_viewdirs = torch.zeros(1, 3, device = self.device)

            # shape: (1, N, 4)
            raw_output = teacher_model(inputs, dummy_viewdirs, "fine")

            if verbose is True:
                print("shape of raw_output: ", raw_output.shape)

            densities = raw_output.view(-1, 4)[..., -1]

            if verbose is True:
                print("densities of teacher model: ", densities)

            chunk_densities = densities.view(-1, 27)
            
            # if any densities of a subgrid > threshold, we consider this grid to be occupied
            occupied_mask = torch.any(chunk_densities > self.occ_threshold, dim = -1)

            # write results back to grids
            # get current indexes of the original grids
            current_indices = indices[i:i + self.chunk]

            occupied_indices = current_indices[occupied_mask]

            self.occ_grid[occupied_indices[:, 0], occupied_indices[:, 1], occupied_indices[:, 2]] = True

            if verbose is True:
                occupied_ratio = self.occ_grid.sum() / self.occ_grid.numel()
                print(f"Occupied ratio: {occupied_ratio:.4f}")