import torch
import os
import numpy as np
from tqdm import tqdm

from src.config import cfg, args
from src.models import make_network
from src.datasets import make_data_loader
from src.utils.net_utils import load_network
from src.models.nerf.renderer.volume_renderer import Renderer

# 注意：您可能需要根据项目结构调整 volume_renderer 的导入路径
# 如果NeRF的渲染器和KiloNeRF的不一样，请导入正确的NeRF渲染器
# 例如 from src.models.nerf.renderer import Renderer as NeRFRenderer

def generate_distillation_data():
    """
    Loads a pre-trained teacher (NeRF) model and generates rendered outputs 
    (RGB and alpha) for the entire training dataset.
    """
    print("==================================================")
    print("Starting distillation data generation...")
    print(f"Scene: {cfg.scene}")
    print(f"Teacher model directory: {cfg.teacher_model_path}")
    print("==================================================")

    # --- 1. Setup Teacher Model and Renderer ---
    # 我们需要加载的是教师模型，其配置在 cfg.teacher 中
    # 我们临时将主 cfg 的网络部分替换为 teacher 的配置 
    teacher_network = make_network(cfg, "nerf")
    
    # 从配置文件指定的目录加载教师模型权重
    epoch = load_network(teacher_network, cfg.teacher_model_dir, resume=True)
    if epoch == 0:
        print(f"Error: No trained teacher model found in {cfg.teacher.trained_model_dir}")
        return
    print(f"Successfully loaded teacher model from epoch {epoch}.")
    
    teacher_network.to(torch.device("cuda:{}".format(cfg.local_rank)))
    teacher_network.eval()

    # 使用 volume_renderer.py 中的 Renderer
    # 注意：这里的 Renderer 必须是原始 NeRF 模型的渲染器
    renderer = Renderer(teacher_network)


    # --- 2. Setup DataLoader ---
    # 创建一个特殊的数据加载器，强制它加载完整的图像
    cfg.train_dataset.force_full_image = True # 使用我们之前添加的参数
    data_loader = make_data_loader(
        cfg, is_train=True, is_distributed=False
    )
    
    # --- 3. Define Output Directory ---
    # distilled 数据保存路径，建议在配置文件中定义
    # 例如在 lego.yaml 中增加:
    # dist_dataset:
    #   data_root: "data/lego/distilled"
    dist_data_root = cfg.dist_dataset.data_root
    os.makedirs(dist_data_root, exist_ok=True)
    print(f"Distilled data will be saved to: {dist_data_root}")

    # --- 4. Generation Loop ---
    for i, batch in enumerate(tqdm(data_loader, desc="Generating Distilled Data")):
        # 将数据移动到GPU
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(torch.device("cuda:{}".format(cfg.local_rank)))

        with torch.no_grad():
            # 使用 NeRF 的渲染器进行渲染，不启用训练模式下的扰动
            render_result = renderer.render(batch, is_training=False)
        
        # 提取需要的数据 (光线原点/方向, 教师模型的输出)
        # 确保这些键名与 renderer 的返回一致
        output_data = {
            'xyz': batch['xyz'].cpu(),
            'viewdirs': batch['viewdirs'].cpu(),
            'teacher_rgb': render_result['fine_rgb_map'].cpu(),
            'teacher_alpha': render_result['alpha_map'].cpu()
        }
        
        # 保存为 .pt 文件
        output_path = os.path.join(dist_data_root, f"dist_{i:04d}.pt")
        torch.save(output_data, output_path)

    print("==================================================")
    print("Distillation data generation finished successfully!")
    print(f"Total {len(data_loader)} files saved in {dist_data_root}")
    print("==================================================")


def main():
    if cfg.distributed:
        cfg.local_rank = int(os.environ["RANK"]) % torch.cuda.device_count()
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    generate_distillation_data()


if __name__ == "__main__":
    main()
