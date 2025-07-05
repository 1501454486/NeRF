from src.config import cfg, args
from src.models import make_network
from src.train import (
    make_trainer,
    make_optimizer,
    make_lr_scheduler,
    make_recorder,
    set_lr_scheduler,
)
from src.datasets import make_data_loader
from src.utils.net_utils import (
    load_model,
    save_model,
    load_network,
    save_trained_config,
    load_pretrain,
)
from src.evaluators import make_evaluator
import torch
import torch.distributed as dist
import os

torch.autograd.set_detect_anomaly(True)

if cfg.fix_random:
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(cfg, network):

    save_trained_config(cfg)
    # train_loader = make_data_loader(
    #     cfg, is_train=True, is_distributed=cfg.distributed, max_iter=cfg.ep_iter
    # )
    val_loader = make_data_loader(cfg, is_train=False)

    # dataloader for distillation
    if cfg.train.ep_dist > 0:
        dist_loader = make_data_loader(
            cfg, is_train = True, is_distributed = cfg.is_distributed, max_iter = cfg.ep_iter, is_dist = True
        )
    # dataloader for fine-tuning
    if cfg.train.epoch > cfg.train.ep_dist:
        ft_loader = make_data_loader(
            cfg, is_train = True, is_distributed = cfg.is_distributed, max_iter = cfg.ep_iter, is_dist = False
        )

    trainer = make_trainer(cfg, network, None)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)
    evaluator = make_evaluator(cfg)

    begin_epoch = load_model(
        network,
        optimizer,
        scheduler,
        recorder,
        cfg.trained_model_dir,
        resume=cfg.resume,
    )
    if begin_epoch == 0 and cfg.pretrain != "":
        load_pretrain(network, cfg.pretrain)

    set_lr_scheduler(cfg, scheduler)

    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch

        # decide which loader to use according to current epoch
        if epoch < cfg.train.ep_dist:
            stage = 'distillation'
            train_loader = dist_loader
        else:
            stage = 'fine-tuning'
            if epoch == cfg.train.ep_dist:
                print("\n" + "=" * 30)
                print("Switching from Distillation to Fine-tuning")
                print("\n" + "=" * 30)
            train_loader = ft_loader

        if cfg.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)

        if hasattr(train_loader.dataset, 'epoch'):
            train_loader.dataset.epoch = epoch

        trainer.train(epoch, train_loader, optimizer, recorder)

        scheduler.step()

        if (epoch + 1) % cfg.save_ep == 0 and cfg.local_rank == 0:
            save_model(
                network, optimizer, scheduler, recorder, cfg.trained_model_dir, epoch
            )

        if (epoch + 1) % cfg.save_latest_ep == 0 and cfg.local_rank == 0:
            save_model(
                network,
                optimizer,
                scheduler,
                recorder,
                cfg.trained_model_dir,
                epoch,
                last=True,
            )

        if (epoch + 1) % cfg.eval_ep == 0 and cfg.local_rank == 0:
            trainer.val(epoch, val_loader, evaluator, recorder)

    return network


def test(cfg, network):
    trainer = make_trainer(cfg, network)
    val_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    epoch = load_network(
        network, cfg.trained_model_dir, resume=cfg.resume, epoch=cfg.test.epoch
    )
    trainer.val(epoch, val_loader, evaluator)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def main():
    if cfg.distributed:
        cfg.local_rank = int(os.environ["RANK"]) % torch.cuda.device_count()
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    network = make_network(cfg, "kilonerf")
    if args.test:
        test(cfg, network)
    else:
        train(cfg, network)
    if cfg.local_rank == 0:
        print("Success!")
        print("=" * 80)
    # os.system("kill -9 {}".format(os.getpid()))


if __name__ == "__main__":
    main()
