import torch
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
        self.register_buffer('occ_grid_resolution_tensor', torch.tensor(self.occ_grid_resolution, device = self.device))

        # grid path
        res_str = f"{self.occ_grid_resolution[0]}_{self.occ_grid_resolution[1]}_{self.occ_grid_resolution[2]}"
        self.grid_path = os.path.join(self.result_dir, f'occ_grid_res{res_str}_thresh{self.occ_threshold}.pth')
        self.occ_grid = None
        if os.path.exists(self.grid_path):
            self._load_grid()
        else:
            self._evaluate_grid()
            self._save_grid()
        
        self.estimator = nerfacc.OccGridEstimator(
            roi_aabb = self.scene_aabb,
            resolution = self.occ_grid.shape,
            levels = 1
        ).to(self.device)

        self.estimator.binaries = self.occ_grid.flatten()
        self.render_step_size = cfg.sampler.render_step_size
        self.early_stop_eps = cfg.sampler.ert_threshold


    def forward(self, batch, is_training: bool = True):
        rays_o, viewdirs = batch['xyz'], batch['viewdirs']
        batch_size, N_rays, _ = rays_o.shape
        rays_o_flat, viewdirs_flat = rays_o.view(-1, 3), viewdirs.view(-1, 3)
        num_total_rays = batch_size * N_rays

        # Perform sampling with ESS
        ray_indices, t_starts, t_ends = self.estimator.sampling(
            rays_o = rays_o_flat,
            rays_d = viewdirs_flat,
            render_step_size = self.render_step_size,
            early_stop_eps = self.early_stop_eps,
        )

        # define callback function
        def rgb_alpha_fn(t_starts, t_ends, ray_indices):
            t_mid = (t_start + t_ends) / 2.0
            pts = rays_o_flat[ray_indices] + viewdirs_flat[ray_indices] * t_mid.unsqueeze(-1)
            # query network
            rgb_chunk, density_chunk = self.net(pts.unsqueeze(1), viewdirs_flat[ray_indices])
            rgb_chunk = torch.sigmoid(rgb_chunk)
            # calculate alpha
            delta = (t_ends - t_starts).unsqueeze(-1)
            alpha = 1.0 - torch.exp(-torch.relu(density_chunk) * delta)
            return rgb_chunk, alpha.squeeze(-1)

        # call rendering function of nerfacc
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

    # def add_noise(self, raw_density):
    #     """
    #     This function is responsible for adding noise to the raw density values.

    #     @param raw_density: The raw density values.
    #     @return: The raw density values with added noise.

    #     """
    #     noise = torch.randn(raw_density.shape, device = raw_density.device) * self.raw_noise_std
    #     raw_density = raw_density + noise
    #     return raw_density


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
        self.occ_grid = torch.load(self.grid_path, map_location = self.device).squeeze(0)
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