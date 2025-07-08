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
    """
    Renderer for KiloNeRF.

    Args:
        net: The net to represent this scene. It will be queried to generate rgb and sigma.
        white_bkgd: whether to use background or not.
        occ_thred: threshold of occupancy grid. Grids with sigma < occ_thred is False, which means no occupancy; True otherwise.
        result_dir: To which grid result will be written to.
    """
    def __init__(
        self,
        net,
        white_bkgd,
        occ_thred,
        result_dir
    ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = net
        self.white_bkgd = white_bkgd
        self.occ_threshold = occ_thred
        self.result_dir = result_dir

        grid_resolution = cfg.task_arg.grid_resolution
        occ_factor = cfg.sampler.occ_factor
        self.occ_grid_resolution = [grid * occ_factor for grid in grid_resolution] # e.g., [256, 256, 256]
        self.scene_aabb = torch.tensor(cfg.task_arg.aabb['min'] + cfg.task_arg.aabb['max'], dtype=torch.float32, device=self.device)
        self.register_buffer('aabb_min', torch.tensor(cfg.task_arg.aabb['min'], device = self.device))
        self.register_buffer('aabb_max', torch.tensor(cfg.task_arg.aabb['max'], device = self.device))

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


    def forward(self, batch, is_training: bool = True):
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
            _, sigmas = self.net(positions.unsqueeze(1), viewdirs_flat[ray_indices])
            return torch.relu(sigmas.view(-1))  # (n_samples,)

        # Perform sampling with ESS
        # ray_indices: (N_samples, )
        # t_starts: (N_samples, )
        # t_ends: (N_samples, )
        ray_indices, t_starts, t_ends = self.estimator.sampling(
            rays_o = rays_o_flat,
            rays_d = viewdirs_flat,
            sigma_fn = sigma_fn,
            render_step_size = cfg.sampler.render_step_size,
            early_stop_eps = cfg.sampler.ert_threshold,
        )

        # define callback function
        def rgb_alpha_fn(t_starts, t_ends, ray_indices):
            t_mid = (t_starts + t_ends) / 2.0
            # shape of pts: (N_samples, 3)
            pts = rays_o_flat[ray_indices] + viewdirs_flat[ray_indices] * t_mid.unsqueeze(-1)
            # query network
            rgb_chunk, density_chunk = self.net(pts, viewdirs_flat[ray_indices])
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
        colors, opacities, depths, _ = nerfacc.rendering(
            t_starts = t_starts,
            t_ends = t_ends,
            ray_indices = ray_indices,
            n_rays = batch_size * N_rays,
            rgb_alpha_fn = rgb_alpha_fn,
        )

        # deal with background
        colors += (1.0 - opacities) * (1.0 if self.white_bkgd else 0.0)

        image = {
            'rgb_map': colors.view(batch_size, N_rays, 3),
            'depth_map': depths.view(batch_size, N_rays, 1),
            'alpha_map': opacities.view(batch_size, N_rays, 1)
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
        occ_grid = torch.load(self.grid_path, map_location = self.device)
        print(f"Occupancy grid loaded from {self.grid_path}.\n Shape of occ_grid: ", self.occ_grid.shape)

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
        cell_size = (self.aabb_max - self.aabb_min) / torch.tensor(self.occ_grid_resolution, dtype = torch.float32, device = self.device)

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