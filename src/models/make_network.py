import imp

def make_network(cfg, model_name=None):
    """
    一个更健壮的 make_network 函数，按以下顺序决定加载哪个模型：
    1. 优先使用函数调用时直接传入的 model_name 参数。
    2. 如果没有传入，则查找 cfg.test.model_name 配置。
    3. 如果配置中也没有，则使用一个默认值。
    """
    final_model_name = ""

    # 优先级 1: 检查是否直接传入了参数
    if model_name is not None:
        final_model_name = model_name
        print(f"INFO: Using explicitly passed model_name: '{final_model_name}'")
    else:
        # 优先级 2: 从配置文件中查找
        # 优先级 3: 如果配置文件里也没有，则使用默认值 'kilonerf'
        default_name = 'kilonerf' # 你可以根据需要修改默认值
        final_model_name = cfg.get('test', {}).get('model_name', default_name)
        print(f"INFO: Using model_name from config (or default): '{final_model_name}'")

    # 根据最终确定的模型名称选择对应的模块和路径
    if final_model_name == "nerf":
        module = cfg.teacher_model_module
        path = cfg.teacher_model_path
    else:
        # 其他所有情况 (包括 "kilonerf") 都加载主网络
        module = cfg.network_module
        path = cfg.network_path

    # 加载并返回网络实例
    network = imp.load_source(module, path).Network()
    return network