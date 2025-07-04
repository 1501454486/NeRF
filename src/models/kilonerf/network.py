import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from src.models.encoding import get_encoder
from src.config import cfg


class KiloNeRF(nn.Module):
    def __init__(
        self, D = 2, W = 32, input_ch = 3, input_ch_views = 3, use_viewdirs = True
    ):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.use_viewdirs = use_viewdirs
        self.output_ch = 4

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, self.W)] +
            [nn.Linear(self.W, self.W) for i in range(self.D - 1)]
        )

        self.view_linears = nn.ModuleList(
            [nn.Linear(self.input_ch_views + self.W, self.W)]
        )

        if self.use_viewdirs:
            # feature vector(32)
            self.feature_linear = nn.Linear(self.W, self.W)
            # sigma(1)
            self.sigma_linear = nn.Linear(self.W, 1)
            # rgb color(3)
            self.rgb_linear = nn.Linear(self.W, 3)
        else:
            # output channel(default: 4)
            self.output_linear = nn.Linear(self.W, self.output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views], dim=-1
        )
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)

        if self.use_viewdirs:
            sigma = self.sigma_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.view_linears):
                h = self.view_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, sigma], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # dictionary, aabb['min']: [...], aabb['max']: [...]
        self.aabb = cfg.task_arg.aabb
        # list of [16, 16, 16]
        grid_resolution = cfg.task_arg.grid_resolution
        
        # for each of networks, randomly sample a batch of 3D points inside the 3D grid cell that corresponds to the respective network
        # these batches are augmented with viewing directions which are drawn randomly from the unit sphere
        self.batch_size = cfg.task_arg.N_rays
        self.chunk = cfg.task_arg.chunk_size
        self.batch_size = cfg.task_arg.N_rays
        self.white_bkgd = cfg.task_arg.white_bkgd
        self.use_viewdirs = cfg.task_arg.use_viewdirs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.register_buffer('aabb_min', torch.tensor(self.aabb['min'], device=self.device, dtype = torch.float32))
        self.register_buffer('aabb_max', torch.tensor(self.aabb['max'], device=self.device, dtype = torch.float32))
        self.register_buffer('grid_resolution', torch.tensor(grid_resolution, device=self.device, dtype = torch.long))

        # encoder
        self.embed_fn, self.input_ch = get_encoder(cfg.network.xyz_encoder)
        self.embeddirs_fn, self.input_ch_views = get_encoder(cfg.network.dir_encoder)

        self.model = nn.ModuleList([        # x dim
            nn.ModuleList([                 # y dim
                nn.ModuleList([             # z dim
                    KiloNeRF(
                        D = cfg.network.kilonerf.D,
                        W = cfg.network.kilonerf.W,
                        input_ch = self.input_ch,
                        input_ch_views = self.input_ch_views,
                        use_viewdirs = self.use_viewdirs
                    ) for _ in range(self.grid_resolution[2])
                ]) for _ in range(self.grid_resolution[1])
            ]) for _ in range(self.grid_resolution[0])
        ])


    def forward(self, pts, viewdirs):
        """
        Prepares inputs and applies network corresponding to the input points

        @param pts: (N_rays, N_samples, 3)
        @param viewdirs: (N_rays, 3)

        @return: rgb of shape (N_rays, N_samples, 3)
        @return: density of shape (N_rays, N_samples, 1)
        """
        # 1. prepare for model inputs
        N_rays, N_samples, _ = pts.shape
        # flatten pts
        pts_flat = pts.view(-1, 3)
        # expand viewdirs to match all points
        viewdirs_expanded = viewdirs.unsqueeze(1).expand(-1, N_samples, -1).reshape(-1, 3)

        # 2. calculate model indices and flatten them
        model_indices = self._get_index(pts_flat)       # (N_rays * N_samples, 3)

        # map 3-dimensional index (i, j, k) into 1-dimensional index
        # flat_index = i * res_y * res_z + j * res_z + k
        res_y, res_z = self.grid_resolution[1], self.grid_resolution[2]
        flat_indices = model_indices[:, 0] * (res_y * res_z) + \
                       model_indices[:, 1] * res_z + \
                       model_indices[:, 2]

        # 3. process in groups
        output_dim = self.model[0][0][0].output_ch
        outputs_flat = torch.empty(pts_flat.shape[0], output_dim, device = self.device)

        # embed features
        embedded_pts = self.embed_fn(pts_flat)
        embedded_dirs = self.embeddirs_fn(viewdirs_expanded)
        
        # find all activated models in this batch
        unique_flat_indices = torch.unique(flat_indices)

        # traverse all the activated models
        for flat_idx in unique_flat_indices:
            # find all pts that belongs to the current model
            mask = (flat_indices == flat_idx)

            # Get 3-dimensional index of current model
            i = flat_idx // (res_y * res_z)
            j = (flat_idx % (res_y * res_z)) // res_z
            k = flat_idx % res_z

            fn = self.model[i][j][k]

            kilonerf_input = torch.cat([embedded_pts[mask], embedded_dirs[mask]], dim = -1)
            model_output = self.batchify(fn, self.chunk)(kilonerf_input)
            outputs_flat[mask] = model_output

        # 4. process outputs
        # reshape outputs back to original shape
        outputs = outputs_flat.view(N_rays, N_samples, output_dim)

        rgb = outputs[..., :3]
        sigma = outputs[..., 3:4]       # remain the last dim to be 1
        
        return rgb, sigma


    def _get_index(self, pts) -> torch.Tensor:
        """
        Get the index of the voxel grid.

        @param pts: input tensor of shape (N, 3)
        @return: index tensor of shape (N, 3)
        """
        # 1. Normalize points to [0, 1] range based on the bounding box
        normalized_pts = (pts - self.aabb_min) / (self.aabb_max - self.aabb_min)
        
        # 2. Scale points to the grid resolution
        scaled_pts = normalized_pts * self.grid_resolution
        
        # 3. Clamp the indices to the valid range [0, resolution-1]
        #    This is the key fix to prevent out-of-bounds errors.
        clamped_pts = torch.clamp(scaled_pts, min=0)
        max_bounds = self.grid_resolution.float() - 1.0
        clamped_pts = torch.min(clamped_pts, max_bounds)
        
        # 4. Convert to long integer for indexing
        return clamped_pts.long()

    def _get_model(self, index):
        """
        Get the model corresponding to the given index

        @param index: index of the voxel grid.
        @return: model of the given index.
        """
        return self.model[index[0]][index[1]][index[2]]


    def batchify(self, fn, chunk):
        """Constructs a version of 'fn' that applies to smaller batches."""

        def ret(inputs):
            return torch.cat(
                [fn(inputs[i : i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0
            )

        return ret


    def get_color_reg_loss(self):
        """
        Calculate the L2 loss for the color-dependent layers
        """
        reg_loss = 0.0
        # traverse all the KiloNeRF models of this network
        for i in range(self.grid_resolution[0]):
            for j in range(self.grid_resolution[1]):
                for k in range(self.grid_resolution[2]):
                    model = self.model[i][j][k]

                    # extract layers concerning viewdirs
                    color_layer_1 = model.view_linears[0]
                    color_layer_2 = model.rgb_linear

                    # calculate L2 norm of weights and biases
                    reg_loss += torch.sum(color_layer_1.weight ** 2)
                    reg_loss += torch.sum(color_layer_1.bias ** 2)
                    reg_loss += torch.sum(color_layer_2.weight ** 2)
                    reg_loss += torch.sum(color_layer_2.bias ** 2)

        return reg_loss