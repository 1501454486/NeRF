import time
import datetime
import torch
import tqdm
from torch.nn.parallel import DistributedDataParallel
from src.config import cfg
from src.utils.data_utils import to_cuda


class Trainer(object):
    def __init__(self, network):
        device = torch.device("cuda:{}".format(cfg.local_rank))
        network = network.to(device)
        if cfg.distributed:
            network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(network)
            network = DistributedDataParallel(
                network,
                device_ids=[cfg.local_rank],
                output_device=cfg.local_rank,
                find_unused_parameters=True,
            )
        self.network = network
        self.local_rank = cfg.local_rank
        self.device = device
        self.global_step = 0

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):
        for k in batch:
            if isinstance(batch[k], tuple) or isinstance(batch[k], list):
                # batch[k] = [b.cuda() for b in batch[k]]
                batch[k] = [b.to(self.device) for b in batch[k]]
            elif isinstance(batch[k], dict):
                batch[k] = {key: self.to_cuda(batch[k][key]) for key in batch[k]}
            else:
                # batch[k] = batch[k].cuda()
                batch[k] = batch[k].to(self.device)
        return batch

    def train(self, epoch, data_loader, optimizer, recorder, stage = None):
        max_iter = len(data_loader)
        self.network.train()
        end = time.time()
        for iteration, batch in enumerate(tqdm.tqdm(data_loader, desc=f"Epoch {epoch} Training")):
            data_time = time.time() - end
            iteration = iteration + 1

            batch = to_cuda(batch, self.device)
            batch["step"] = self.global_step
            batch["epoch"] = epoch
            batch['is_training'] = True
            if stage is not None:
                batch['stage'] = stage

            def occ_eval_fn(x):
                dummy_viewdirs = torch.zeros_like(x)
                _, sigma = self.network.net(x, dummy_viewdirs)
                return sigma.squeeze()

            self.network.renderer.estimator.update_every_n_steps(
                step = self.global_step,
                occ_eval_fn = occ_eval_fn,
            )

            output, loss, loss_stats, image_stats = self.network(batch)

            # training stage: loss; optimizer; scheduler
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
            optimizer.step()

            if cfg.local_rank > 0:
                continue

            # data recording stage: loss_stats, time, image_stats
            recorder.step += 1

            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            self.global_step += 1
            if iteration % cfg.log_interval == 0 or iteration == (max_iter - 1):
                # print training state
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]["lr"]
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = "  ".join(
                    ["eta: {}", "{}", "lr: {:.6f}", "max_mem: {:.0f}"]
                )
                training_state = training_state.format(
                    eta_string, str(recorder), lr, memory
                )
                print(training_state)

                # record loss_stats and image_dict
                recorder.update_image_stats(image_stats)
                recorder.record("train")

    def val(self, epoch, data_loader, evaluator=None, recorder=None, stage = None):
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        image_stats = {}
        data_size = len(data_loader)
        for batch in tqdm.tqdm(data_loader):
            batch = to_cuda(batch, self.device)
            batch["step"] = recorder.step
            batch['epoch'] = epoch
            batch['is_training'] = False
            if stage is not None:
                batch['stage'] = stage
            with torch.no_grad():
                output, loss, loss_stats, _ = self.network(batch)
                if evaluator is not None:
                    image_stats_ = evaluator.evaluate(output, batch)
                    if image_stats_ is not None:
                        image_stats.update(image_stats_)

            loss_stats = self.reduce_loss_stats(loss_stats)
            for k, v in loss_stats.items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v

        loss_state = []
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append("{}: {:.4f}".format(k, val_loss_stats[k]))
        print(loss_state)

        if evaluator is not None:
            result = evaluator.summarize()
            val_loss_stats.update(result)

        if recorder:
            recorder.record("val", epoch, val_loss_stats, image_stats)
