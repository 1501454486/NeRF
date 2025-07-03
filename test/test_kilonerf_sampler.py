import os
import sys
import torch
import torch.nn as nn

# 将项目根目录添加到 Python 路径中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config import cfg, args
from src.models.kilonerf.renderer.sampler import Sampler

def setup_test_sampler(cfg):
    """
    创建一个用于测试的 Sampler 实例。
    这个函数会绕过 Sampler 原本的 __init__ 方法，以避免加载真实模型。
    取而代之的是，它会手动设置必要的属性，并注入一个合成的占用栅格。
    """
    # 1. 创建一个 Sampler 实例，但不调用其 __init__ 方法
    sampler = Sampler.__new__(Sampler)
    # 2. 因为 Sampler 继承了 nn.Module, 我们需要手动调用其构造函数
    super(Sampler, sampler).__init__()

    # 3. 手动设置必要的属性
    sampler.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler.aabb = cfg.task_arg.aabb
    sampler.occ_grid_resolution = [res * cfg.sampler.occ_factor for res in cfg.task_arg.grid_resolution]
    sampler.max_points = cfg.sampler.max_points
    sampler.near = cfg.sampler.near
    sampler.far = cfg.sampler.far
    
    # 4. 手动注册 buffer
    sampler.register_buffer('aabb_min', torch.tensor(sampler.aabb['min'], device=sampler.device, dtype=torch.float32))
    sampler.register_buffer('aabb_max', torch.tensor(sampler.aabb['max'], device=sampler.device, dtype=torch.float32))
    sampler.register_buffer('occ_grid_resolution_tensor', torch.tensor(sampler.occ_grid_resolution, device=sampler.device, dtype=torch.long))

    # 5. 创建并注入一个合成的占用栅格 (occ_grid)
    occ_grid = torch.zeros(sampler.occ_grid_resolution, dtype=torch.bool, device=sampler.device)
    res = sampler.occ_grid_resolution[0]
    start, end = res // 4, res - res // 4
    occ_grid[start:end, start:end, start:end] = True
    sampler.occ_grid = occ_grid

    print("[*] 测试专用的 Sampler 已设置完毕。")
    print(f"    合成的占用栅格尺寸: {sampler.occ_grid_resolution}")
    print(f"    被占用的区域索引范围: [{start}:{end}] on all axes.")
    
    return sampler

def test_ray_match(sampler):
    """
    测试 _ray_match 方法 (空洞空间跳过)。
    """
    print("\n" + "="*50)
    print("--- Running Test 1: _ray_match Functionality ---")
    print("="*50)

    device = sampler.device
    aabb_min, aabb_max = sampler.aabb_min, sampler.aabb_max
    center = (aabb_min + aabb_max) / 2.0

    # --- 场景 1: 光线完全错过占用区 ---
    print("[*] 场景 1: 光线错过占用区...")
    rays_o_miss = torch.tensor([[aabb_min[0], aabb_min[1], aabb_min[2] - 0.1]], device=device) # 稍微移出边界
    rays_d_miss = torch.tensor([[1.0, 1.0, 0.0]], device=device)
    rays_d_miss = rays_d_miss / torch.linalg.norm(rays_d_miss)
    near_miss = torch.tensor([0.0], device=device)
    far_miss = torch.tensor([4.0], device=device)

    new_near_miss = sampler._ray_match(rays_o_miss, rays_d_miss, near_miss, far_miss)
    assert torch.allclose(new_near_miss, far_miss), f"场景1失败: 期望 new_near={far_miss.item()}, 实际为 {new_near_miss.item()}"
    print("[+] 场景 1 通过!")

    # --- 场景 2: 光线命中占用区 ---
    print("\n[*] 场景 2: 光线命中占用区...")
    rays_o_hit = torch.tensor([[aabb_min[0] - 0.1, center[1], center[2]]], device=device)
    rays_d_hit = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    near_hit = torch.tensor([0.0], device=device)
    far_hit = torch.tensor([4.0], device=device)

    new_near_hit = sampler._ray_match(rays_o_hit, rays_d_hit, near_hit, far_hit)
    assert new_near_hit > near_hit, "场景2失败: new_near 应该大于原始的 near"
    assert new_near_hit < far_hit, "场景2失败: new_near 应该小于 far"
    print(f"[+] 场景 2 通过! (new_near 从 {near_hit.item()} 更新为 {new_near_hit.item():.4f})")

    # --- 场景 3: 光线从占用区内部开始 ---
    # 新逻辑：由于函数会先步进再检查，所以 new_near 会是第一个步长的距离，而不是 0.0
    # 因此，我们断言它大于等于 near 并且远小于 far 即可。
    print("\n[*] 场景 3: 光线从占用区内部开始...")
    rays_o_inside = center.unsqueeze(0)
    rays_d_inside = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    near_inside = torch.tensor([0.0], device=device)
    far_inside = torch.tensor([4.0], device=device)

    new_near_inside = sampler._ray_match(rays_o_inside, rays_d_inside, near_inside, far_inside)
    
    # 修正断言： new_near 应该是大于等于0的一个很小的值（一个步长的距离）
    assert new_near_inside >= near_inside, f"场景3失败: new_near ({new_near_inside.item()}) 不应小于 near ({near_inside.item()})"
    assert new_near_inside < 0.1, f"场景3失败: new_near ({new_near_inside.item()}) 对于内部起点来说太大了"
    print(f"[+] 场景 3 通过! (new_near is a small positive value {new_near_inside.item():.4f}, as expected with the new logic)")

    print("\n--- _ray_match Test Passed! ---")


def test_forward(sampler):
    """
    测试 forward 方法 (即原 _sample)。
    """
    print("\n" + "="*50)
    print("--- Running Test 2: forward() Functionality ---")
    print("="*50)

    N_rays = 10
    device = sampler.device

    # 创建一个批次数据
    rays_o = torch.rand(N_rays, 3, device=device) * (sampler.aabb_max - sampler.aabb_min) + sampler.aabb_min
    viewdirs = torch.randn(N_rays, 3, device=device)
    viewdirs = viewdirs / torch.linalg.norm(viewdirs, dim=-1, keepdim=True)
    batch = {'xyz': rays_o.unsqueeze(0), 'viewdirs': viewdirs.unsqueeze(0)}

    print(f"[*] 正在使用 {N_rays} 条光线调用 sampler(batch)...")
    pts, z_vals = sampler(batch, is_training=False)

    print("[*] 正在验证输出的形状...")
    expected_pts_shape = [N_rays, sampler.max_points, 3]
    expected_z_vals_shape = [N_rays, sampler.max_points]

    assert list(pts.shape) == expected_pts_shape, f"pts 形状错误! 期望: {expected_pts_shape}, 实际: {list(pts.shape)}"
    assert list(z_vals.shape) == expected_z_vals_shape, f"z_vals 形状错误! 期望: {expected_z_vals_shape}, 实际: {list(z_vals.shape)}"
    print("[+] 输出形状正确!")

    print("\n[*] 正在验证 z_vals 的数值...")
    print("z_vals: ", z_vals)
    print("sampler.far: ", sampler.far)
    assert torch.all(z_vals >= sampler.near), "z_vals 中的值小于 sampler.near"
    assert torch.all(z_vals <= sampler.far + 1e-6), "z_vals 中的值大于 sampler.far (plus tolerance)"
    
    diff = z_vals[:, 1:] - z_vals[:, :-1]
    sorted_check = torch.all(diff >= -1e-6) # 允许一个及其微小的负数容差
    assert sorted_check, "z_vals 不是单调递增的"
    print("[+] z_vals 数值范围和排序正确!")

    print("\n--- forward() Test Passed! ---")


if __name__ == "__main__":
    print("="*50)
    print("--- KiloNeRF Sampler Test Unit ---")
    print("="*50)

    # 1. 设置测试环境
    sampler = setup_test_sampler(cfg)

    # 2. 运行测试
    test_ray_match(sampler)
    test_forward(sampler)

    print("\n" + "="*50)
    print("--- ALL SAMPLER TESTS PASSED SUCCESSFULLY! ---")
    print("="*50)