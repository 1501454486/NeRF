import yaml
import os
import sys
from pprint import pprint

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from src.datasets.nerf.blender import Dataset


def test_dataset_initialization(split_type='train'):
    print("--- 开始测试 Dataset 初始化 ---")

    # --- 关键步骤2：加载 YAML 配置文件 ---
    config_path = 'configs/nerf/lego.yaml'
    print(f"正在从 '{config_path}' 加载配置...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("配置加载成功！")

    # --- 关键步骤3：提取 Dataset 所需的配置 ---
    # 从总配置中，只取出 'train_dataset' 这一部分
    # 此时 train_dataset_config 是一个字典
    if split_type == 'train':
        dataset_config = config['train_dataset']
    elif split_type == 'val':
        dataset_config = config['val_dataset']
    elif split_type == 'test':
        dataset_config = config['test_dataset']
    else:
        raise ValueError(f"未知的 split_type: {split_type}")

    print("\n提取出的 'dataset' 配置信息:")
    pprint(dataset_config)

    # --- 关键步骤4：使用 **kwargs 语法进行初始化 ---
    print("\n正在使用提取的配置初始化 Dataset...")
    # 这里的 **dataset_config 是魔法发生的地方
    # 它会将字典中的所有键值对，解包成关键字参数传递给 __init__ 方法
    # 例如：等价于 Dataset(data_root="data/lego", split="train", ...)
    dataset = Dataset(**dataset_config)
    print("Dataset 初始化成功！")
    
    # --- 后续的验证步骤 ---
    print("\n--- 开始验证 Dataset ---")
    dataset_size = len(dataset)
    print(f"数据集报告的总长度为: {dataset_size}")

    print("\n获取并检查第一个样本...")
    first_sample = dataset.frames[0]
    print("第一个样本的内容键:", first_sample.keys())
    print("self.img_path:", dataset.img_path)
    print("img_path:", first_sample['file_path'])

    return dataset

def test_dataset_getitem(dataset):
    print("\n--- 开始测试 Dataset 的 __getitem__ 方法 ---")
    # 这里假设 dataset 已经被正确初始化
    # 直接调用 __getitem__ 方法获取第一个样本
    first_sample = dataset[0]
    print("第一个样本的内容:", first_sample)
    print("第一个样本的类型:", type(first_sample))
    print("第一个样本的形状:", first_sample.shape if hasattr(first_sample, 'shape') else 'N/A')
    print("第一个样本的数据类型:", first_sample.dtype if hasattr(first_sample, 'dtype') else 'N/A')
    print("第一个样本的键:", first_sample.keys() if isinstance(first_sample, dict) else 'N/A')
    print("第一个样本的图像数据:", first_sample['image'] if isinstance(first_sample, dict) else 'N/A')




if __name__ == "__main__":
    dataset = test_dataset_initialization('train')
    test_dataset_getitem(dataset)