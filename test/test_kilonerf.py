import yaml
import os
import sys
from pprint import pprint

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.config import cfg, args
from src.models.kilonerf.network import Network
from src.models.make_network import make_network


def test_batchify(network, cfg):
    """
    测试1: 验证 batchify 功能。
    创建一个大于 chunk_size 的输入，并确保所有点都落在同一个 voxel 中，
    以强制单个 KiloNeRF 模型通过 batchify 来处理这个大批量数据。
    """
    print("\n" + "="*50)
    print("--- Running Test 2: Batchify Functionality ---")
    print("="*50)

    chunk_size = cfg.task_arg.chunk_size
    # 创建一个比 chunk_size 大的数据批次
    N_rays = 1
    N_samples = chunk_size * 2
    
    print(f"[*] Chunk size is {chunk_size}. Creating a large input with {N_samples} points.")

    # 计算第一个 voxel 的边界
    grid_res = network.grid_resolution
    cell_size = (network.aabb_max - network.aabb_min) / network.grid_resolution
    
    # 生成所有点，确保它们都落在第一个 voxel 内
    # 这样可以保证只有一个子网络被激活，并接收所有 N_samples 个点
    pts_in_cell = torch.rand(N_rays, N_samples, 3, device=network.device) * cell_size
    pts = pts_in_cell + network.aabb_min
    
    viewdirs = torch.randn(N_rays, 3, device=network.device)
    viewdirs = viewdirs / torch.linalg.norm(viewdirs, dim=-1, keepdim=True)
    
    print("[*] Running forward pass on the large batch...")
    try:
        with torch.no_grad():
            rgb, sigma = network(pts, viewdirs)
        print("[+] Forward pass on large batch completed successfully!")
    except Exception as e:
        print(f"[-] An error occurred during batched forward pass: {e}")
        assert False, "Batchify test failed with a runtime error."

    # 验证输出维度是否正确
    expected_rgb_shape = [N_rays, N_samples, 3]
    expected_sigma_shape = [N_rays, N_samples, 1]
    
    assert list(rgb.shape) == expected_rgb_shape, "Batched RGB shape is incorrect!"
    assert list(sigma.shape) == expected_sigma_shape, "Batched Sigma shape is incorrect!"
    print("[+] Output dimensions are correct.")
    print("\n--- Batchify Test Passed! ---")


def test_indexing(network, cfg):
    """
    测试2: 验证模型索引功能。
    我们将在两个不同的 voxel 中创建点，并验证：
    1. 通过整个 Network 得到的结果与直接调用相应子网络得到的结果一致。
    2. 不同 voxel 中的点会由不同的子网络处理，得到不同的结果。
    """
    print("\n" + "="*50)
    print("--- Running Test 3: Model Indexing Correctness ---")
    print("="*50)

    device = network.device
    
    # 1. 选择两个不同的 voxel 索引
    idx1 = (0, 0, 0)
    idx2 = tuple(x - 1 for x in network.grid_resolution) # e.g., (15, 15, 15)
    print(f"[*] Testing indexing for voxels {idx1} and {idx2}.")
    assert idx1 != idx2, "Cannot test indexing with the same voxel index."

    # 2. 获取对应的子网络模型
    model1 = network.model[idx1[0]][idx1[1]][idx1[2]]
    model2 = network.model[idx2[0]][idx2[1]][idx2[2]]
    
    # 3. 在每个 voxel 的相同相对位置创建测试点
    cell_size = (network.aabb_max - network.aabb_min) / network.grid_resolution
    relative_pos = torch.tensor([0.5, 0.5, 0.5], device=device) * cell_size

    pts1_world = (network.aabb_min + torch.tensor(idx1, device=device) * cell_size + relative_pos).view(1, 1, 3)
    pts2_world = (network.aabb_min + torch.tensor(idx2, device=device) * cell_size + relative_pos).view(1, 1, 3)
    
    # 创建一个共享的视角方向
    viewdirs = torch.tensor([[0.0, 0.0, -1.0]], device=device)

    # 4. === 通过整个网络前向传播 ===
    print("[*] Path A: Running points through the full Network.forward()...")
    with torch.no_grad():
        rgb1_net, sigma1_net = network(pts1_world, viewdirs)
        rgb2_net, sigma2_net = network(pts2_world, viewdirs)
    
    # 5. === 直接调用子网络 ===
    print("[*] Path B: Calling sub-models directly...")
    with torch.no_grad():
        # 准备子网络输入
        embedded_pts1 = network.embed_fn(pts1_world.view(-1, 3))
        embedded_pts2 = network.embed_fn(pts2_world.view(-1, 3))
        embedded_dirs = network.embeddirs_fn(viewdirs.view(-1, 3))
        
        input1_direct = torch.cat([embedded_pts1, embedded_dirs], dim=-1)
        input2_direct = torch.cat([embedded_pts2, embedded_dirs], dim=-1)
        
        # 直接调用
        output1_direct = model1(input1_direct)
        output2_direct = model2(input2_direct)

        rgb1_direct = output1_direct[..., :3]
        rgb2_direct = output2_direct[..., :3]

    # 6. === 验证结果 ===
    print("[*] Verifying results...")
    
    # 验证1: 全网络结果应与直接调用子网络结果一致
    assert torch.allclose(rgb1_net.squeeze(), rgb1_direct.squeeze(), atol=1e-6), "Network output for point 1 does not match direct model call."
    print("[+] Test 1 Passed: Point in voxel {} was correctly routed.".format(idx1))

    assert torch.allclose(rgb2_net.squeeze(), rgb2_direct.squeeze(), atol=1e-6), "Network output for point 2 does not match direct model call."
    print("[+] Test 2 Passed: Point in voxel {} was correctly routed.".format(idx2))

    # 验证2: 不同子网络处理相同相对位置的点，结果应不同 (因为模型权重不同)
    assert not torch.allclose(rgb1_net.squeeze(), rgb2_net.squeeze()), "Outputs from different voxels are unexpectedly the same."
    print("[+] Test 3 Passed: Different sub-models produce different results as expected.")
    
    # 验证3: 交叉验证，证明点1没有被路由到模型2
    output1_from_model2 = model2(input1_direct)
    rgb1_from_model2 = output1_from_model2[..., :3]
    assert not torch.allclose(rgb1_net.squeeze(), rgb1_from_model2.squeeze()), "Point 1 was incorrectly routed to model 2."
    print("[+] Test 4 Passed: Cross-check successful, routing is not arbitrary.")
    
    print("\n--- Indexing Test Passed! ---")


def test_color_reg_loss(network, cfg):
    """
    测试4: 验证颜色正则化损失计算的正确性。
    """
    print("\n" + "="*50)
    print("--- Running Test 4: Color Regularization Loss ---")
    print("="*50)

    # 1. 检查 network 对象中是否有 get_color_reg_loss 方法
    assert hasattr(network, 'get_color_reg_loss') and callable(getattr(network, 'get_color_reg_loss')), \
        "Network class is missing the 'get_color_reg_loss' method."
    print("[*] 'get_color_reg_loss' method found.")

    with torch.no_grad():
        # 2. 计算初始损失
        loss_before = network.get_color_reg_loss()
        print(f"[*] Initial regularization loss: {loss_before.item():.6f}")
        assert loss_before >= 0, "Regularization loss should be non-negative."

        # 3. 选择一个参数进行修改
        # 我们选择第一个子网络的 rgb_linear 层的第一个权重
        target_model = network.model[0][0][0]
        param_to_change = target_model.rgb_linear.weight
        original_value = param_to_change.data[0, 0].clone()
        
        # 4. 手动修改该参数
        delta = 1.0
        print(f"[*] Modifying a weight in model (0,0,0)'s rgb_linear layer from {original_value.item():.6f} by {delta}.")
        param_to_change.data[0, 0] += delta

        # 5. 计算修改后的损失
        loss_after = network.get_color_reg_loss()
        print(f"[*] Loss after modification: {loss_after.item():.6f}")

        # 6. 验证损失变化是否符合预期
        # 损失的变化量应为 (w+delta)^2 - w^2
        expected_change = (original_value + delta)**2 - original_value**2
        actual_change = loss_after - loss_before
        
        print(f"[*] Expected loss change: {expected_change.item():.6f}")
        print(f"[*] Actual loss change:   {actual_change.item():.6f}")
        
        assert torch.allclose(actual_change, expected_change, atol=1e-2), "The change in regularization loss is not as expected."
        print("[+] Loss change is correct.")

        # 7. 恢复原始权重，避免影响其他测试
        param_to_change.data[0, 0] = original_value
        loss_restored = network.get_color_reg_loss()
        assert torch.allclose(loss_restored, loss_before, atol=1e-5), "Failed to restore the original weight."
        print("[*] Original weight restored successfully.")

    print("\n--- Color Regularization Loss Test Passed! ---")


if __name__ == "__main__":
    # ==================================
    # ===      基本功能测试 (SMOKE TEST)
    # ==================================
    print("="*50)
    print("--- Running Test 1: Basic Functionality (Smoke Test) ---")
    print("="*50)

    # 设置测试参数
    N_rays = cfg.task_arg.N_rays
    N_samples = cfg.sampler.max_points
    
    # 实例化网络
    print("1. Initializing network...")
    network = make_network(cfg, "kilonerf")
    network.to(network.device)
    print(f"   Network initialized on device: {network.device}")
    
    # 生成测试数据
    print("\n2. Generating test data...")
    pts = torch.rand(N_rays, N_samples, 3, device=network.device) * \
          (network.aabb_max - network.aabb_min) + network.aabb_min
    viewdirs = torch.randn(N_rays, 3, device=network.device)
    viewdirs = viewdirs / torch.linalg.norm(viewdirs, dim=-1, keepdim=True)
    
    print(f"   Input 'pts' shape:      {list(pts.shape)}")
    print(f"   Input 'viewdirs' shape: {list(viewdirs.shape)}")
    
    # 运行前向传播
    print("\n3. Running forward pass...")
    rgb, sigma = None, None
    try:
        with torch.no_grad():
            rgb, sigma = network(pts, viewdirs)
        print("   Forward pass completed successfully!")
    except Exception as e:
        print(f"   An error occurred during forward pass: {e}")
        import traceback
        traceback.print_exc()
        
    # 检查输出维度
    if rgb is not None and sigma is not None:
        print("\n4. Verifying output dimensions...")
        print(f"   Output 'rgb' shape:   {list(rgb.shape)}")
        print(f"   Output 'sigma' shape: {list(sigma.shape)}")
        
        expected_rgb_shape = [N_rays, N_samples, 3]
        expected_sigma_shape = [N_rays, N_samples, 1]
        
        assert list(rgb.shape) == expected_rgb_shape, "RGB shape is incorrect!"
        assert list(sigma.shape) == expected_sigma_shape, "Sigma shape is incorrect!"
        
        print("\n--- Basic Test Passed! ---")

        # ==================================
        # ===      高级功能测试
        # ==================================
        # 如果基础测试通过，则进行更详细的测试
        test_batchify(network, cfg)
        test_indexing(network, cfg)
        test_color_reg_loss(network, cfg)
        
        print("\n" + "="*50)
        print("--- ALL TESTS PASSED SUCCESSFULLY! ---")
        print("="*50)