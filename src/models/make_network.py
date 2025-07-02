import imp


def make_network(cfg, model_name = "nerf"):
    if model_name == "nerf":
        module = cfg.teacher_model_module
        path = cfg.teacher_model_path
    else:
        module = cfg.network_module
        path = cfg.network_path

    network = imp.load_source(module, path).Network()
    return network