import torch
import torch.nn as nn

from src.config import cfg
from src.models.kilonerf.renderer.volumn_renderer import VolumnRenderer


class NetworkWrapper(nn.Module):
    def __init__(self, net, train_loader = None):
        super().__init__()
        self.net = net
        self.renderer = VolumnRenderer(net)

        self.l2_reg = cfg.task_arg.get('l2_reg', 0.0)
        self.ep_dist = cfg.train.ep_dist
        self.ep_ft = cfg.train.ep_ft
        self.train_loader = train_loader

    def forward(self, batch):
        """
        Forward and calculate loss during a batch.

        @param batch
        @param stage: current stage of training, distillation or fine-tuning;
                      during distillation, its loss comprises of 2 parts: L2 loss between its and teacher's, and L2 regularization loss
                      during fine-tuning, its loss is mse between gt and predicted.
        @param teacher_rgb: rgb result of batch from teacher model
        @param teacher_alpha: alpha result of batch from teacher model
        """
        is_training = batch['is_training']
        epoch = batch.get('epoch', 0)
        stage = batch['stage']

        student_image = self.renderer(batch, is_training)
        # shape of student_rgb: (B, N_rays, 3)
        student_rgb = student_image['rgb_map']
        # shape of student_alpha: (B, N_rays, 1)
        student_alpha = student_image['alpha_map']

        loss_stats, image_stats = {}, {}

        if stage == 'distillation':
            teacher_rgb = batch['teacher_rgb']
            teacher_alpha = batch['teacher_alpha']
            loss_color = nn.functional.mse_loss(teacher_rgb.view(-1, 3), student_rgb.view(-1, 3))
            loss_alpha = nn.functional.mse_loss(teacher_alpha.view(-1, 1), student_alpha.view(-1, 1))
            loss = loss_color + loss_alpha

            loss_stats.update(loss_color = loss_color, loss_alpha = loss_alpha)

            if self.l2_reg > 0:
                loss_reg = self.l2_reg * self.net.get_color_reg_loss()
                loss_stats.update(loss_regularization = loss_reg)
                loss += loss_reg

        elif stage == 'fine-tuning':
            # shape: (B, H, W, 3)
            gt_rgb = batch['gt_rgb']
            loss = nn.functional.mse_loss(gt_rgb.view(-1, 3), student_rgb.view(-1, 3))
            loss_stats.update(loss_finetune_mse = loss)
            image_stats.update(rgb_map = student_rgb, gt_rgb = gt_rgb)
        else:
            # validation or test, only forward
            loss = torch.tensor(0.0, device = student_rgb.device)
            image_stats.update(rgb_map = student_rgb, teacher_rgb = teacher_rgb)

        loss_stats.update(loss = loss)

        return student_image, loss, loss_stats, image_stats
