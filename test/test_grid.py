import torch
import numpy as np

# 文件路径
file_path = 'data/result/kilonerf_replication/lego/kilonerf/default/occ_grid_res256_256_256_thresh10.pth'

# 加载数据
# 假设文件是使用 torch.save(tensor) 保存的
try:
    occ_grid = torch.load(file_path)
except Exception as e:
    print(f"加载文件时出错: {e}")
    # 有些项目可能会把张量保存在一个字典里
    # data = torch.load(file_path)
    # occ_grid = data['occ_grid'] # 键名取决于具体的保存代码

# 检查数据的基本信息
print(f"数据类型: {occ_grid.dtype}")
print(f"数据形状: {occ_grid.shape}")

# 查看其中包含哪些值 (确认是 0/1 或 True/False)
print(f"包含的唯一值: {torch.unique(occ_grid)}")

# 确保数据在 CPU 上并且是布尔类型或整数类型
occ_grid_np = occ_grid.cpu().bool().numpy()