import torch
import torch.nn as nn
from src.models.nerf.renderer.volume_renderer import Renderer


class NetworkWrapper(nn.Module):
    def __init__(self, net, train_loader = None):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.renderer = Renderer(self.net)
        self.train_loader = train_loader

        # add metrics here

    def forward(self, batch):
        """
        Write your codes here.
        """
        gt_rgb = batch['gt_rgb']
        is_training = batch['is_training']
        output = self.renderer.render(batch, is_training)
        coarse_rgb_map = output['coarse_rgb_map']
        fine_rgb_map = output['fine_rgb_map']

        loss_c = nn.functional.mse_loss(coarse_rgb_map, gt_rgb)
        loss_f = nn.functional.mse_loss(fine_rgb_map, gt_rgb)
        loss = loss_c + loss_f

        loss_stats = {
            'loss_c': loss_c,
            'loss_f': loss_f,
            'loss': loss
        }

        image_stats = {
            'coarse_rgb_map': coarse_rgb_map,
            'fine_rgb_map': fine_rgb_map,
            'gt_rgb': gt_rgb,
            'error_map': torch.abs(fine_rgb_map - gt_rgb)
        }

        return output, loss, loss_stats, image_stats
