import yaml
import os
import sys
from pprint import pprint

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from src.datasets.nerf.blender import Dataset
from src.models.nerf.renderer.volume_renderer import Renderer


def test_renderer_render(net):
    print("\n--- 开始测试 Renderer 的 render 方法 ---")
    
    # 这里假设 net 已经被正确初始化
    # 创建一个模拟的 batch 数据
    batch = {
        'rays_o': torch.randn(10, 3),  # 模拟光线原点
        'rays_d': torch.randn(10, 3),  # 模拟光线方向
        'viewdirs': torch.randn(10, 3)  # 模拟视角方向
    }
    
    print("正在调用 render 方法...")
    rendered_output = net.render(batch)
    print("render 方法调用成功！")
    
    print("渲染输出:", rendered_output)

if __name__ == "__main__":
    # 测试 Renderer 的初始化
    network = make_network(cfg).cuda()
    load_network(network, cfg.trained_model_dir, epoch=cfg.test.epoch)
    renderer = make_renderer(cfg, network)
    
    # 测试 Renderer 的 render 方法
    test_renderer_render(network)
    
    print("\n--- 所有测试完成 ---")