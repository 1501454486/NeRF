from .trainer import Trainer
import imp
import os

from src.models import make_network
from src.utils.net_utils import load_network
from src.train.trainers.nerf import NetworkWrapper as TeacherNetworkWrapper


def _wrapper_factory(cfg, network, train_loader=None):
    module = cfg.loss_module
    path = cfg.loss_path
    network_wrapper = imp.load_source(module, path).NetworkWrapper(network, train_loader = None)
    return network_wrapper


def make_trainer(cfg, network, train_loader=None):
    network = _wrapper_factory(cfg, network, train_loader)
    return Trainer(network)
