import numpy as np
from src.config import cfg
import os
import torch.nn.functional as F
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class Evaluator:
    def __init__(
        self,
    ):
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.imgs = []

    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt) ** 2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def ssim_metric(self, img_pred, img_gt, batch, id, num_imgs):
        result_dir = os.path.join(cfg.result_dir, "images")
        os.system("mkdir -p {}".format(result_dir))

        ##################### FIX #################################
        pred_to_save = (img_pred[..., [2, 1, 0]] * 255).astype(np.uint8)
        gt_to_save = (img_gt[..., [2, 1, 0]] * 255).astype(np.uint8)

        cv2.imwrite(
            "{}/view{:03d}_pred.png".format(result_dir, id),
            pred_to_save,
        )
        cv2.imwrite(
            "{}/view{:03d}_gt.png".format(result_dir, id),
            gt_to_save,
        )
        img_pred_uint8 = (img_pred * 255).astype(np.uint8)
        img_gt_uint8 = (img_gt * 255).astype(np.uint8)

        ssim, _ = compare_ssim(img_pred_uint8, img_gt_uint8, win_size=101, full=True, channel_axis = -1)
        return ssim

    def evaluate(self, output, batch):
        """
        Write your codes here.
        """
        image_stats = {}
        
        img_pred = output['fine_rgb_map'].cpu().numpy()
        img_pred = img_pred.reshape(batch['H'].item(), batch['W'].item(), 3)
        img_gt = batch['gt_rgb'].cpu().numpy()
        img_gt = img_gt.reshape(batch['H'].item(), batch['W'].item(), 3)

        psnr = self.psnr_metric(img_pred, img_gt)
        ssim = self.ssim_metric(img_pred, img_gt, batch, batch['id'].item(), batch['num_imgs'])
        img = img_pred
        
        self.psnr.append(psnr)
        self.ssim.append(ssim)
        self.imgs.append(img)

        return {}

    def summarize(self):
        """
        Write your codes here.
        """
        if not self.psnr:
            print("No evaluations to summarize.")
            return {}
        
        mean_psnr = np.mean(self.psnr)
        mean_ssim = np.mean(self.ssim)

        summary = {
            "mean_psnr": mean_psnr,
            "mean_ssim": mean_ssim,
        }

        print(f"Validation Summary: PSNR: {mean_psnr:.4f}, SSIM: {mean_ssim:.4f}")

        result_path = os.path.join(cfg.result_dir, "metrics.json")
        with open(result_path, 'w') as f:
            json.dump(summary, f, indent=4)

        self.psnr.clear()
        self.ssim.clear()

        return summary
