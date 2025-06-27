import torch.utils.data as data
import torch
import numpy as np
from src.config import cfg
import json
import os
from PIL import Image


class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        """
        Description:
            __init__ 函数负责从磁盘中 load 指定格式的文件，计算并存储为特定形式

        Input:
            @kwargs: 读取的参数
        Output:
            None
        """
        super(Dataset, self).__init__()
        """
        Write your codes here.
        """
        self.data_root = kwargs['data_root']
        self.split = kwargs['split'] # train / val / test
        self.input_ratio = kwargs['input_ratio'] #  Whether to downsample the image
        # height and width of images
        self.H = kwargs['H']
        self.W = kwargs['W']
        # start, end and step_szie of cameras
        cam_start, cam_end, cam_step = kwargs['cams']

        # Select train/val/test data
        self.cam_path = os.path.join(self.data_root, f"transforms_{self.split}.json")
        self.img_path = os.path.join(self.data_root, self.split)

        """
        cam_info:
        - camera_angle_x -> float
        - frames -> list(dict)
        e.g. frames[0]:
            - file_path -> string
            - rotation -> float
            - transform_matrix -> list[4], each is a list[4]
        """
        with open(self.cam_path, 'r') as f:
            cam_info = json.load(f)

        # slice cam_info
        self.camera_angle_x = cam_info['camera_angle_x']
        if cam_end == -1:
            self.frames = cam_info['frames'][cam_start::cam_step]
        else:
            self.frames = cam_info['frames'][cam_start:cam_end:cam_step]

        # default: 2 and 6
        self.near = kwargs.get('near', 2)
        self.far = kwargs.get('far', 6)


    def __getitem__(self, index):
        """
        Description:
            __getitem__ 函数负责在运行时提供给网络一次训练需要的输入，以及 ground truth 的输出
        对 NeRF 来说，分别是 1024 条光线以及 1024 个 RGB值

        Input:
            @index: 图像下标, 范围为 [0, len-1]
        Output:
            @ret: 包含所需数据的字典
        """
        """
        Write your codes here.
        """
        frame = self.frames[index]
        img_path = os.path.join(self.data_root, frame['file_path'])
        image = self.load_image(img_path)   # shape: (H, W, 3)
        H, W, _ = image.shape

        # Randomly sample 1024 rays
        if self.split == 'train':
            # Randomly select 1024 pixels from the image
            u = np.random.randint(0, W, size=1024)
            v = np.random.randint(0, H, size=1024)
        else:
            # For validation and testing, use the whole image
            v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            u = u.flatten() # shape: (H*W,)
            v = v.flatten() # shape: (H*W,)

        # normalized camera coordinates
        coords_uv = np.stack([u, v], axis = 1)
        rgb = image[v, u, :]
        
        # physical image plane coordinates:
        coords_img = np.stack([-(coords_uv[:, 0] - W / 2), coords_uv[:, 1] - H / 2], axis = 1)

        # normalized camera coordinates
        processed_coords = coords_img / self.camera_angle_x
        z_col = np.full((coords_img.shape[0], 1), -1, dtype=processed_coords.dtype)
        coords_cam = np.concatenate([processed_coords, z_col], axis = 1)

        # camera coordinates to world coordinates
        # 1. turn it into homogeneous coords
        ones_col = np.ones((coords_cam.shape[0], 1))
        coords_cam_h = np.concatenate([coords_cam, ones_col], axis = 1) # shape: (1024, 4)
        coords_world_h_transposed = frame['transform_matrix'] @ coords_cam_h.T  # shape: (4, 1024)
        coords_world = coords_world_h_transposed.T[:, :3]   # shape: (1024, 3)

        rotation_matrix = np.array(frame['transform_matrix'])[:3, :3]  # shape: (3, 3)
        rays_transposed = rotation_matrix @ coords_world.T  # shape: (3, 1024)
        rays = rays_transposed.T  # shape: (1024, 3)

        return rays, rgb
        

    def __len__(self):
        """
        Description:
            __len__ 函数返回训练或者测试的数量

        Input:
            None
        Output:
            @len: 训练或者测试的数量
        """
        """
        Write your codes here.
        """
        return len(self.frames)

    def load_image(self, base_img_path):
        """
        Description:
            Open an image file and return it as a numpy array.

        Input:
            @image_path: Path to the image file.
        Output:
            @image: Image as a numpy array.
        """
        extentions = ['.png', '.jpg', '.jpeg']
        for ext in extentions:
            full_path = base_img_path + ext
            if os.path.exists(full_path):
                image_path = full_path
                break

        image = Image.open(image_path)
        # If need to downsample the image
        if self.input_ratio < 1.0:
            image = image.resize((int(self.W * self.input_ratio), int(self.H * self.input_ratio)), Image.LANCZOS)
        else:
            image = image.resize((self.W, self.H), Image.LANCZOS)

        # Conveert RGBA image into Numpy array and normalize to [0, 1]
        image = np.array(image).astype(np.float32) / 255.0
        # issue1
        # Get RGB channels and alpha channel
        rgb = image[:, :, :3]   # shape: (H, W, 3)
        alpha = image[:, :, 3:] # shape: (H, W, 1)
        # Create a white background
        white_background = np.ones_like(rgb, dtype=np.float32)
        # Alpha blend the image with the white background
        image_rgb = rgb * alpha + white_background * (1 - alpha)
        return image_rgb