import torch
import torch.nn.functional as F
import nerfacc
import os

# --- 脚本配置 ---
# 如果你的GPU显存较小，可以把分辨率改小一点，例如 [128, 128, 128]
RESOLUTION = [256, 256, 256]
# 模拟你的项目中的场景边界
SCENE_AABB = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
# 模拟你的项目中的光线步长
RENDER_STEP_SIZE = 1e-3

def run_test():
    """
    主测试函数
    """
    # 1. 初始化环境和设备
    if not torch.cuda.is_available():
        print("❌ 错误：未找到CUDA设备。此测试必须在GPU上运行。")
        return
    device = torch.device("cuda")
    
    print("--- 环境检查 ---")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"Nerfacc 版本: {nerfacc.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    print(f"当前设备: {torch.cuda.get_device_name(device)}")
    print("-" * 40)

    # 2. 初始化 OccGridEstimator
    scene_aabb_tensor = torch.tensor(SCENE_AABB, dtype=torch.float32, device=device)
    
    # 我们将手动创建一个干净的、可复现的占用栅格
    # 而不是从你的文件中加载，以确保测试的纯净性
    print("正在初始化 Estimator 并创建虚拟占用栅格...")
    estimator = nerfacc.OccGridEstimator(
        roi_aabb=scene_aabb_tensor,
        resolution=RESOLUTION
    ).to(device)

    # 创建一个简单的球体作为占用区域
    grid_coords = torch.stack(torch.meshgrid(
        torch.linspace(SCENE_AABB[0], SCENE_AABB[3], RESOLUTION[0]),
        torch.linspace(SCENE_AABB[1], SCENE_AABB[4], RESOLUTION[1]),
        torch.linspace(SCENE_AABB[2], SCENE_AABB[5], RESOLUTION[2]),
        indexing='ij'
    ), dim=-1).to(device)
    
    # 将球体内部标记为占用 (True)
    binaries = (grid_coords.norm(dim=-1) < 0.75).flatten()
    estimator.binaries = binaries
    
    print("Estimator 初始化完毕。")
    print(f"Binaries (占用栅格) 形状: {estimator.binaries.shape}, 类型: {estimator.binaries.dtype}")
    print("-" * 40)

    # 3. 定义不同形状的测试用例
    # 根据 nerfacc 文档, `rays_o` 和 `rays_d` 都应该是 (n_rays, 3) 的2D张量
    
    # 你的项目中 batch_size * N_rays 是 2048
    N_RAYS = 2048
    
    test_cases = {
        "情况1: 标准形状 (2048, 3)": {
            "rays_o": torch.randn(N_RAYS, 3, device=device),
            "rays_d": F.normalize(torch.randn(N_RAYS, 3, device=device), dim=-1)
        },
        "情况2: 你的代码逻辑 (1, 2048, 3) -> view(-1, 3)": {
            "rays_o": torch.randn(1, N_RAYS, 3, device=device).view(-1, 3),
            "rays_d": F.normalize(torch.randn(1, N_RAYS, 3, device=device), dim=-1).view(-1, 3)
        },
        "情况3: 错误形状 (3D张量)": {
            "rays_o": torch.randn(1, N_RAYS, 3, device=device), # 未被压平
            "rays_d": F.normalize(torch.randn(1, N_RAYS, 3, device=device), dim=-1)
        },
        "情况4: 错误形状 (最后一个维度不是3)": {
            "rays_o": torch.randn(N_RAYS, 4, device=device),
            "rays_d": F.normalize(torch.randn(N_RAYS, 4, device=device), dim=-1)
        },
    }

    # 4. 执行测试
    for name, data in test_cases.items():
        print(f"\n--- 正在测试: {name} ---")
        rays_o_test = data["rays_o"]
        rays_d_test = data["rays_d"]
        
        # 确保数据是内存连续的，就像你的代码一样
        rays_o_test = rays_o_test.contiguous()
        rays_d_test = rays_d_test.contiguous()

        print(f"输入 rays_o 形状: {rays_o_test.shape}, 是否连续: {rays_o_test.is_contiguous()}")
        print(f"输入 rays_d 形状: {rays_d_test.shape}, 是否连续: {rays_d_test.is_contiguous()}")

        try:
            # 调用 nerfacc 的核心函数
            ray_indices, t_starts, t_ends = estimator.sampling(
                rays_o=rays_o_test,
                rays_d=rays_d_test,
                render_step_size=RENDER_STEP_SIZE,
            )
            print("✅ 成功: `estimator.sampling` 函数执行完毕，没有报错。")
            print(f"  输出 ray_indices 形状: {ray_indices.shape}, 类型: {ray_indices.dtype}")
            print(f"  输出 t_starts 形状: {t_starts.shape}, 类型: {t_starts.dtype}")
            print(f"  输出 t_ends 形状: {t_ends.shape}, 类型: {t_ends.dtype}")
            
            if ray_indices.numel() > 0:
                print(f"  输出样本 (前5个采样点):")
                print(f"    ray_indices: {ray_indices[:5].tolist()}")
                print(f"    t_starts: {[f'{t:.4f}' for t in t_starts[:5]]}")

        except Exception as e:
            print(f"❌ 失败: `estimator.sampling` 抛出异常！")
            print(f"  异常类型: {type(e).__name__}")
            print(f"  异常信息: {e}")
            
if __name__ == "__main__":
    run_test()