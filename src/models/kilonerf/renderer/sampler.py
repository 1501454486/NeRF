import torch
import torch.nn as nn
from src.config import cfg, args
from src.models import make_network
from src.utils.net_utils import load_network
from tqdm import tqdm
import os


class Sampler:
    """
    This module is responsible for sampling pts in the grids which are occupied and intersects with given viewdirs. Also, it maintains a uniform grid with a higher grid resolution in the given aabb, and use a pre-trained nerf model as teacher model to evaluate whether a grid is occupied.
    """
    def __init__(self):
        self.teacher_model = make_network(cfg).cuda()
        load_network(self.teacher_model, cfg.teacher_model_dir)
        self.device = self.teacher_model.device
        self.result_dir = cfg.result_dir
        self.aabb = cfg.task_arg.aabb
        self.chunk = cfg.task_arg.chunk_size
        
        grid_resolution = cfg.task_arg.grid_resolution
        occ_factor = cfg.sampler.occ_factor
        self.occ_grid_resolution = [grid * occ_factor for grid in grid_resolution] # e.g., [256, 256, 256]
        self.occ_threshold = cfg.sampler.occ_threshold

        # grid path
        res_str = f"{self.occ_grid_resolution[0]}_{self.occ_grid_resolution[1]}_{self.occ_grid_resolution[2]}"
        self.grid_path = os.path.join(self.result_dir, f'occ_grid_res{res_str}_thresh{self.occ_threshold}.pth')

        self.occ_grid = None

        # if already exists, load directly, else evaluate and save
        if os.path.exists(self.grid_path):
            self.load_grid()
        else:
            print("Occupancy grid not found. Evaluating from NeRF...")
            self.evaluate_grid()
            self.save_grid()


    def save_grid(self):
        """
        Save grid to the disk
        """
        os.makedirs(self.result_dir, exist_ok = True)
        torch.save(self.occ_grid, self.grid_path)
        print(f"Occupancy grid saved to {self.grid_path}")

    
    def load_grid(self):
        """
        Load grid from disk
        """
        self.occ_grid = torch.load(self.grid_path, map_location = self.device)
        print(f"Occupancy grid loaded from {self.grid_path}")


    @torch.no_grad()
    def evaluate_grid(self, verbose: bool = False):
        """
        Use a teacher model (NeRF) to evaluate and generate grid.
        This process will traverse all the grids, and in each grid, it will generate sub_grid^3 to evaluate whether this grid is occupird or not.
        """
        if verbose is True:
            print("teacher model: ", self.teacher_model)
        # initialize a bool grid with all False
        self.occ_grid = torch.zeros(self.occ_grid_resolution, dtype = torch.bool, device = self.device)

        # calculate size of every grid from aabb and resolution
        aabb_min = torch.tensor(self.aabb['min'], device = self.device)
        aabb_max = torch.tensor(self.aabb['max'], device = self.device)
        resolution_tensor = torch.tensor(self.occ_grid_resolution, device = self.device)
        cell_size = (aabb_max - aabb_min) / resolution_tensor

        # calculate central coordinates of every cell
        x = torch.linspace(aabb_min[0], aabb_max[0], self.occ_grid_resolution[0] + 1, device = self.device)[:-1]
        y = torch.linspace(aabb_min[1], aabb_max[1], self.occ_grid_resolution[1] + 1, device = self.device)[:-1]
        z = torch.linspace(aabb_min[2], aabb_max[2], self.occ_grid_resolution[2] + 1, device = self.device)[:-1]
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
            raw_output = self.teacher_model(inputs, dummy_viewdirs, "fine")

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



    def sample(self, viewdirs):
        pass