from .trainer import Trainer
import imp
import os

from src.models import make_network
from src.utils.net_utils import load_network
from src.train.trainers.nerf import NetworkWrapper as TeacherNetworkWrapper


def _wrapper_factory(cfg, network, train_loader=None):
    if cfg.task == "kilonerf_replication":
        if not cfg.teacher_model_path:
            raise ValueError("teacher_model_path is a must for kilonerf!")
        teacher_network = make_network(cfg, "nerf")
        print("teacher model dir: ", cfg.teacher_model_dir)
        load_network(teacher_network, cfg.teacher_model_dir)
        teacher_model = TeacherNetworkWrapper(teacher_network)
        network_wrapper = imp.load_source(cfg.loss_module, cfg.loss_path).NetworkWrapper(network, teacher_network = teacher_model)
        
    else:
        module = cfg.loss_module
        path = cfg.loss_path
        network_wrapper = imp.load_source(module, path).NetworkWrapper(network)

    return network_wrapper


def make_trainer(cfg, network, train_loader=None):
    network = _wrapper_factory(cfg, network, train_loader)
    return Trainer(network)
