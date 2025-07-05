import torch
import torch.nn as nn
from src.config import cfg, args
from src.models import make_network
from src.utils.net_utils import load_network
from tqdm import tqdm
import os


class Sampler(nn.Module):
    """
    This module is responsible for sampling pts in the grids which are occupied and intersects with given viewdirs. Also, it maintains a uniform grid with a higher grid resolution in the given aabb, and use a pre-trained nerf model as teacher model to evaluate whether a grid is occupied.
    """
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.result_dir = cfg.result_dir
        self.aabb = cfg.task_arg.aabb
        self.chunk = cfg.task_arg.chunk_size
        
        grid_resolution = cfg.task_arg.grid_resolution
        occ_factor = cfg.sampler.occ_factor
        self.occ_grid_resolution = [grid * occ_factor for grid in grid_resolution] # e.g., [256, 256, 256]
        self.occ_threshold = cfg.sampler.occ_threshold
        self.max_points = cfg.sampler.max_points
        self.near = cfg.sampler.near
        self.far = cfg.sampler.far

        # grid path
        res_str = f"{self.occ_grid_resolution[0]}_{self.occ_grid_resolution[1]}_{self.occ_grid_resolution[2]}"
        self.grid_path = os.path.join(self.result_dir, f'occ_grid_res{res_str}_thresh{self.occ_threshold}.pth')

        self.occ_grid = None
        self.register_buffer('aabb_min', torch.tensor(self.aabb['min'], device = self.device))
        self.register_buffer('aabb_max', torch.tensor(self.aabb['max'], device = self.device))
        self.register_buffer('occ_grid_resolution_tensor', torch.tensor(self.occ_grid_resolution, device = self.device))

        # if already exists, load directly, else evaluate and save
        if os.path.exists(self.grid_path):
            self._load_grid()
        else:
            print("Occupancy grid not found. Evaluating from NeRF...")
            self._evaluate_grid()
            self._save_grid()


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
        self.occ_grid = torch.load(self.grid_path, map_location = self.device)
        print(f"Occupancy grid loaded from {self.grid_path}")


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



    def forward(self, batch, is_training: bool = True):
        """
        sample points from the given rays, using Eearly Ray Termination (only during inference) and Empty Space Skipping

        @param batch: A batch from dataloader (rays, rgb) (Here we only need rays)
        @return: sampled pts (N_rays, N_samples, 3)
        @return: z_vals: sample pts indices (N_rays, N_samples)
        """
        rays_o, viewdirs = batch['xyz'].squeeze(0), batch['viewdirs'].squeeze(0)            # shape: (N_rays, 3)
        
        # 1. Find ray-AABB intersection to get initial near and far bounds
        N_rays = rays_o.shape[0]
        near = torch.full((N_rays, ), self.near, device = self.device, dtype = torch.float32)
        far = torch.full((N_rays, ), self.far, device = self.device, dtype = torch.float32)

        # 2. Use ray matching to skip empty space (ESS)
        with torch.no_grad():
            effective_near = self._ray_match(rays_o, viewdirs, near, far)

        # 3. Generate sample points between the new near and far
        t_vals = torch.linspace(0., 1., self.max_points, device = self.device)
        # Stretch t_vals to be between effective_near and far for each ray
        z_vals = effective_near.unsqueeze(-1) * (1. - t_vals) + far.unsqueeze(-1) * t_vals

        # Stratified sampling for training
        # If during training and use perturb, add perturbation
        if is_training and cfg.task_arg.perturb > 0:
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand(z_vals.shape, device=self.device)
            z_vals = lower + (upper - lower) * t_rand

        # Calculate smaple point coordinates
        # (N_rays, max_points, 3)
        pts = rays_o.unsqueeze(1) + viewdirs.unsqueeze(1) * z_vals.unsqueeze(-1)

        return pts, z_vals


    @torch.no_grad()
    def _ray_match(self, rays_o, rays_d, near, far):
        """
        Perform ray matching to find the intersection with an occupied cell.
        This implements Empty Space Skipping (ESS).
        @param rays_o: (N, 3) ray origins.
        @param rays_d: (N, 3) ray directions.
        @param near: (N, ) near plane for each ray.
        @param far: (N, ) far plane for each ray.
        @return: (N, ) new near plane for each ray after skipping empty space.
        """
        cell_diag_length = torch.norm((self.aabb_max - self.aabb_min) / self.occ_grid_resolution_tensor)
        step_size = cell_diag_length * 0.5  # march with half a cell diagonal

        t = near.clone()
        new_near = near.clone()

        active_rays_mask = torch.ones_like(near, dtype=torch.bool)

        # March until all rays hit something or exceeded their far plane
        while active_rays_mask.any():
            # 1. Calculate the t for the NEXT step for all currently active rays
            t_next = t[active_rays_mask] + step_size

            # 2. Check for hits using the next step's position
            current_points = rays_o[active_rays_mask] + rays_d[active_rays_mask] * t_next.unsqueeze(-1)
            indices = self._get_index(current_points)

            # Create default masks for the active set
            valid_mask = (indices >= 0).all(dim=-1) & (indices < self.occ_grid_resolution_tensor).all(dim=-1)
            hit_mask = torch.zeros(active_rays_mask.sum(), dtype=torch.bool, device=self.device)

            # If any points are in valid grid cells, check their occupancy
            if valid_mask.any():
                valid_indices = indices[valid_mask]
                # Get occupancy and place it into the hit_mask at the correct positions
                hit_mask[valid_mask] = self.occ_grid[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]]

            # 3. For rays that are past the far plane, they are considered misses for this step
            #    Their new_near will be set to far, and they will be deactivated.
            past_far_mask = t_next > far[active_rays_mask]
            
            # A ray is a "miss" in this step if it's past far OR it didn't hit anything
            # But we only deactivate it if it's past far. Hits are handled next.
            miss_and_past_far = past_far_mask & ~hit_mask
            
            # Get global indices for updating the main tensors
            active_indices = torch.where(active_rays_mask)[0]

            # 4. Handle hits: Update new_near and deactivate the ray
            if hit_mask.any():
                hit_indices_in_active = torch.where(hit_mask)[0]
                global_hit_indices = active_indices[hit_indices_in_active]
                
                # IMPORTANT: The new_near is the hit distance, but capped at 'far'
                new_near[global_hit_indices] = torch.min(t_next[hit_indices_in_active], far[global_hit_indices])
                active_rays_mask[global_hit_indices] = False

            # 5. Handle rays that passed 'far' without hitting: Update new_near to 'far' and deactivate
            if miss_and_past_far.any():
                miss_indices_in_active = torch.where(miss_and_past_far)[0]
                global_miss_indices = active_indices[miss_indices_in_active]

                new_near[global_miss_indices] = far[global_miss_indices]
                active_rays_mask[global_miss_indices] = False
            
            # 6. Update t for the next iteration for rays that are still active.
            # We must only update the 't' values for rays that are continuing the march.
            # A direct assignment like `t[active_rays_mask] = t_next[...]` is risky because
            # the number of elements on the left and right may mismatch after deactivation in steps 4 & 5.
            
            # Identify which of the currently active rays (from the start of the loop) will continue.
            continuing_mask_in_active = ~hit_mask & ~past_far_mask
            
            # If there are any rays to continue...
            if continuing_mask_in_active.any():
                # Get the global indices of the rays that are continuing.
                continuing_global_indices = active_indices[continuing_mask_in_active]
                
                # Get the corresponding t_next values for these rays.
                t_next_for_continuing = t_next[continuing_mask_in_active]

                # Update the main 't' tensor only at these specific global indices.
                # This is robust to in-loop deactivations.
                t[continuing_global_indices] = t_next_for_continuing
        
        return new_near.clamp_max(far)

    def _get_index(self, pts) -> torch.Tensor:
        """
        Get the index of the voxel grid.

        @param pts: input tensor of shape (N, 3)
        @return: index tensor of shape (N, 3)
        """
        index = ((pts - self.aabb_min) / ((self.aabb_max - self.aabb_min) / self.occ_grid_resolution_tensor)).long()
        return index