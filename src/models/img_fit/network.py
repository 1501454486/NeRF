import torch
import torch.nn as nn
from torch.nn import functional as F
from src.models.encoding import get_encoder
from src.config import cfg


class Network(nn.Module):
    def __init__(
        self,
    ):
        super(Network, self).__init__()
        net_cfg = cfg.network
        self.uv_encoder, input_ch = get_encoder(net_cfg.uv_encoder)
        D, W = net_cfg.D, net_cfg.W
        self.backbone_layer = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) for i in range(D - 1)]
        )
        self.output_layer = nn.Sequential(nn.Linear(W, 3), nn.Sigmoid())

    def forward_network(self, xyz, xyz_dir, network):
        N_rays, N_samples = xyz.shape[:2]
        xyz, xyz_dir = xyz.reshape(-1, xyz.shape[-1]), xyz_dir.reshape(
            -1, xyz_dir.shape[-1]
        )
        xyz_encoding = self.xyz_encoder(xyz)
        dir_encoding = self.dir_encoder(xyz_dir)
        net_output = network(torch.cat([xyz_encoding, dir_encoding], dim=-1))
        return net_output.reshape(N_rays, N_samples, -1)

    def render(self, uv, batch):
        uv_encoding = self.uv_encoder(uv)
        x = uv_encoding
        for i, l in enumerate(self.backbone_layer):
            x = self.backbone_layer[i](x)
            x = F.relu(x)
        rgb = self.output_layer(x)
        return {"rgb": rgb}

    def batchify(self, uv, batch):
        all_ret = {}
        chunk = cfg.task_arg.chunk_size
        for i in range(0, uv.shape[0], chunk):
            ret = self.render(uv[i : i + chunk], batch)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], dim=0) for k in all_ret}
        return all_ret

    def forward(self, batch):
        B, N_pixels, C = batch["uv"].shape
        ret = self.batchify(batch["uv"].reshape(-1, C), batch)
        return {k: ret[k].reshape(B, N_pixels, -1) for k in ret}
