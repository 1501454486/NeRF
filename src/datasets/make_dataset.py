from . import samplers
from .dataset_catalog import DatasetCatalog
import torch
import torch.utils.data
import imp
from .collate_batch import make_collator
import numpy as np
import time
from src.config.config import cfg
from torch.utils.data import DataLoader, ConcatDataset


torch.multiprocessing.set_sharing_strategy("file_system")


def _dataset_factory(is_train, is_val=False, is_dist=False):
    if is_dist:
        module = cfg.dist_dataset_module
        path = cfg.dist_dataset_path
        args = cfg.dist_dataset
        dataset = imp.load_source(module, path).DistDataset(**args)
        return dataset
    elif is_val:
        module = cfg.val_dataset_module
        path = cfg.val_dataset_path
    elif is_train:
        module = cfg.train_dataset_module
        path = cfg.train_dataset_path
        args = cfg.train_dataset
    else:
        module = cfg.test_dataset_module
        path = cfg.test_dataset_path
        args = cfg.test_dataset
    dataset = imp.load_source(module, path).Dataset(**args)
    return dataset


def make_dataset(cfg, is_train=True, is_dist=False):
    dataset = _dataset_factory(is_train, is_dist=is_dist)
    return dataset


def make_data_sampler(dataset, shuffle, is_distributed):
    if is_distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter, is_train):
    if is_train:
        batch_sampler = cfg.train.batch_sampler
        sampler_meta = cfg.train.sampler_meta
    else:
        batch_sampler = cfg.test.batch_sampler
        sampler_meta = cfg.test.sampler_meta

    if batch_sampler == "default":
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last
        )
    elif batch_sampler == "image_size":
        batch_sampler = samplers.ImageSizeBatchSampler(
            sampler, batch_size, drop_last, sampler_meta
        )

    if max_iter != -1:
        batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, max_iter)
    return batch_sampler


def worker_init_fn(worker_id):
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2**16))))


def make_data_loader(cfg, is_train=True, is_distributed=False, max_iter=-1, is_dist=False):
    if is_train:
        batch_size = cfg.train.batch_size
        # shuffle = True
        shuffle = cfg.train.shuffle
        drop_last = False
    else:
        batch_size = cfg.test.batch_size
        shuffle = True if is_distributed else False
        drop_last = False

    dataset = make_dataset(cfg, is_train, is_dist)
    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(
        cfg, sampler, batch_size, drop_last, max_iter, is_train
    )
    num_workers = cfg.train.num_workers
    collator = make_collator(cfg, is_train)
    data_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collator,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        persistent_workers=True
    )

    return data_loader
