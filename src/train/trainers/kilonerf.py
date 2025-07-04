import torch
import torch.nn as nn

from src.config import cfg
from src.models.kilonerf.renderer.volumn_renderer import VolumnRenderer


class NetworkWrapper(nn.Module):
    def __init__(self, net, train_loader, teacher_network = None):
        super().__init__()
        self.net = net
        self.renderer = VolumnRenderer(net)
        self.teacher_network = teacher_network

        if teacher_network is not None:
            self.teacher_network.eval()
        
        self.is_training = train_loader is not None and train_loader.dataset.split == 'train'

        self.l2_reg = cfg.task_arg.get('l2_reg', 0.0)
        self.ep_dist = cfg.train.ep_dist
        self.ep_ft = cfg.train.ep_ft


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
        epoch = batch.get('epoch', 0)
        stage = 'distillation' if epoch < self.ep_dist else 'fine-tuning'

        student_image = self.renderer(batch, self.is_training)
        student_rgb = student_image['rgb_map']
        student_alpha = student_image['alpha_map']

        loss_stats, image_stats = {}, {}

        if stage == 'distillation':
            if self.teacher_network is not None:
                raise ValueError("Teacher network is required for distillation")

            # Use teacher model for labels
            with torch.no_grad():
                teacher_output, _, _, _ = self.teacher_network(batch)
                teacher_rgb = teacher_output['fine_rgb_map']
                teacher_alpha = teacher_output['alpha_map']

            loss_color = nn.functional.mse_loss(teacher_rgb, student_rgb)
            loss_alpha = nn.functional.mse_loss(teacher_alpha, student_alpha)
            loss = loss_color + loss_alpha

            loss_stats.update(loss_color = loss_color, loss_alpha = loss_alpha)

            if self.l2_reg > 0:
                loss_reg = self.l2_reg * self.net.get_color_reg_loss()
                loss_stats.update(loss_regularization = loss_reg)
                loss += loss_reg

        elif stage == 'fine-tuning':
            gt_rgb = batch['gt_rgb']
            loss = nn.functional.mse_loss(gt_rgb, student_rgb)
            loss_stats.update(loss_finetune_mse = loss)
        else:
            # validation or test, only forward
            loss = torch.tensor(0.0, device = student_rgb.device)

        loss_stats.update(loss = loss)
        image_stats.update(rgb_map = student_rgb, gt_rgb = gt_rgb)

        return student_image, loss, loss_stats, image_stats