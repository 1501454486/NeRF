import torch
import numpy as np
from skimage import measure
import trimesh

# --- 加载数据的代码 (同上) ---
file_path = 'data/result/kilonerf_replication/lego/kilonerf/default/occ_grid_res256_256_256_thresh10.pth'
occ_grid = torch.load(file_path)
# Marching cubes 需要浮点数输入
occ_grid_np = occ_grid.cpu().float().numpy()
# --- 结束 ---

print("正在使用 Marching Cubes 算法生成网格...")
# level=0.5 表示以 0.5 为阈值来区分内外
# spacing 可以调整模型的尺寸比例，这里设为一致
verts, faces, normals, values = measure.marching_cubes(
    occ_grid_np, level=0.5, spacing=(1.0, 1.0, 1.0)
)
print("网格生成完毕。")

# 使用 trimesh 库来创建和保存网格对象
mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

# 可以进行一些平滑处理（可选）
# mesh.process(smooth=True)

# 保存为 .obj 文件
output_path = 'output_mesh.obj'
mesh.export(output_path)

print(f"网格已成功保存到: {output_path}")
print("现在请使用 MeshLab 或其他 3D 查看器打开此文件。")