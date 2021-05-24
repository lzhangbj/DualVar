import os
import sys
import argparse
import time, re
import builtins
import numpy as np
import random
import pickle
import socket
import math
from tqdm import tqdm
from backbone.select_backbone import select_backbone
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils import data
from torchvision import transforms
import torchvision.utils as vutils

import utils.augmentation as A
import utils.transforms as T
import utils.tensorboard_utils as TB
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

plt.switch_backend('agg')

from utils.utils import AverageMeter, calc_topk_accuracy,\
    ProgressMeter, neq_load_customized, save_checkpoint, FastDataLoader
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler
from model import *
from dataset.local_dataset import *
from utils.logging import get_root_logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_flop_stats(model):
    """
    Compute the gflops for the current model given the config.
    Args:
        model (model): model to compute the flop counts.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        is_train (bool): if True, compute flops for training. Otherwise,
            compute flops for testing.
    Returns:
        float: the total number of gflops of the given model.
    """
    from fvcore.nn.flop_count import flop_count
    inputs = (torch.rand(1, 2, 3, 16, 112, 112).cuda().float(),)
    gflop_dict, _ = flop_count(model, inputs)
    gflops = sum(gflop_dict.values())
    return gflops

def get_model(args):
    if args.model == 'moco_naked':
        model = MoCo_Naked(args.net, args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.distributed)
    elif args.model == 'moco_timeseriesv4':
        model = MoCo_TimeSeriesV4(args.net, args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.distributed,
                              n_series=args.n_series, series_dim=args.series_dim,
                              series_T=args.series_T, aligned_T=args.aligned_T, mode=args.mode, args=args)
    elif args.model == 'simclr_naked':
        model = SimCLR_Naked(args.net, args.moco_dim, args.moco_t, args.distributed)
    elif args.model == 'simclr_timeseriesv4':
        model = SimCLR_TimeSeriesV4(args.net, args.moco_dim, args.moco_t, args.distributed,
                              n_series=args.n_series, series_dim=args.series_dim,
                              series_T=args.series_T, aligned_T=args.aligned_T, mode=args.mode, args=args)
    else:
        raise NotImplementedError

    return model

class DistLogger():
    def __init__(self, log_file, print=True):
        self.print = print
        if print:
            self.logger = get_root_logger(log_file)

    def info(self, content):
        if self.print:
            self.logger.info(content)


def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--net', default='r21d', type=str)
    parser.add_argument('--model', default='simclr_timeseriesv4', type=str)
    # time series model
    parser.add_argument('--series_dim', default=64, type=int)
    parser.add_argument('--n_series', default=2, type=int)
    parser.add_argument('--shufflerank_theta', default=0.05, type=float)
    parser.add_argument('--series_T', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--aligned_T', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--mode', default='clip-sr-tc', type=str,
                        choices=['clip-sr-tc', 'clip-sr'])
    # moco specific configs:
    parser.add_argument('--moco-dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--moco-k', default=2048, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    # dataset
    parser.add_argument('--dataset', default='ucf101-2clip-stage-prototype', type=str)
    parser.add_argument('--seq_len', default=16, type=int, help='number of frames in each video block')
    parser.add_argument('--num_seq', default=2, type=int, help='number of video blocks')
    parser.add_argument('--ds', default=4, type=int, help='frame down sampling rate')
    parser.add_argument('--img_dim', default=112, type=int)
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('-j', '--workers', default=16, type=int)
    parser.add_argument('--seed', default=0, type=int)
    # augmentation
    parser.add_argument("--aug_temp_consist", action='store_true')
    parser.add_argument("--aug_series", action='store_true')
    parser.add_argument("--rand_flip", action='store_true')
    # optimizer
    parser.add_argument('--optim', default='sgd', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.03, type=float, help='learning rate')
    parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    # log
    parser.add_argument('--print_freq', default=20, type=int, help='frequency of printing output during training')
    parser.add_argument('--eval_freq', default=5, type=int, help='frequency of eval')
    parser.add_argument('--save_freq', default=5, type=int, help='frequency of saving')
    # mode
    parser.add_argument('--resume', default='', type=str, help='path of model to resume')
    parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
    # exp save directory
    parser.add_argument('--prefix', default='pretrain', type=str)
    parser.add_argument('--name_prefix', default='', type=str)
    # parallel configs:
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')

    args = parser.parse_args()
    return args


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.dist_url == "env://" and args.world_size == -1:
        try:
            args.world_size = int(os.environ["WORLD_SIZE"])
        except:
            args.world_size = 1

    if args.local_rank != -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    args.num_gpu = args.world_size if args.distributed else 1
    args.real_batch_size = args.batch_size * args.num_gpu

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        assert args.local_rank == -1
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    best_acc = 0
    args.gpu = gpu

    if args.distributed:
        if args.local_rank != -1:  # torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ:  # slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        elif args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
            if args.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url)
                                # world_size=args.world_size, rank=args.rank)
        args.rank = torch.distributed.get_rank()
        print(args.rank)
    else:
        args.rank = 0

    args.print = args.gpu == 0 or not args.distributed
    # suppress printing if not master
    if args.rank != 0:
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    args.img_path, args.model_path, args.exp_path, args.log_file = set_path(args)
    args.logger = DistLogger(log_file=args.log_file, print=args.print)
    # args.logger.info('Re-write num_seq to %d' % args.num_seq)

    ### model ###
    args.logger.info("=> creating {} model with '{}' backbone".format(args.model, args.net))
    model = get_model(args)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
            model_without_ddp = model.module
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            model_without_ddp = model.module
    else:
        if args.gpu is None:
            args.gpu = 0
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        model_without_ddp = model

    ### optimizer ###
    params = []
    for name, param in model.named_parameters():
        params.append({'params': param})

    args.logger.info('\n===========Check Grad============')
    for name, param in model.named_parameters():
        if not param.requires_grad:
            args.logger.info((name, param.requires_grad))
    args.logger.info('=================================\n')

    optimizer = optim.SGD(params, lr=args.lr, weight_decay=args.wd, momentum=0.9)

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    args.iteration = 1

    ### data ###
    transform_train = get_transform('train', args)
    train_loader = get_dataloader(get_data(transform_train, 'train', args), 'train', args)
    transform_train_cuda = transforms.Compose([
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225], channel=1)])

    args.logger.info('===================================')

    ### restart training ###
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch'] + 1
            args.iteration = checkpoint['iteration']
            best_acc = checkpoint['best_acc']
            state_dict = checkpoint['state_dict']
            try:
                model_without_ddp.load_state_dict(state_dict)
            except:
                args.logger.info('[WARNING] resuming training with different weights')
                neq_load_customized(model_without_ddp, state_dict, verbose=True, args=args)
            args.logger.info("=> load resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                args.logger.info('[WARNING] failed to load optimizer state, initialize optimizer')
        else:
            args.logger.info("[Warning] no checkpoint found at '{}', use random init".format(args.resume))
    elif args.pretrain:
        if os.path.isfile(args.pretrain):
            checkpoint = torch.load(args.pretrain, map_location=torch.device('cpu'))
            state_dict = checkpoint['state_dict']
            try:
                model_without_ddp.load_state_dict(state_dict)
            except:
                neq_load_customized(model_without_ddp, state_dict, verbose=True, args=args)
            args.logger.info(
                "=> loaded pretrained checkpoint '{}' (epoch {})".format(args.pretrain, checkpoint['epoch']))
        else:
            args.logger.info("=> no checkpoint found at '{}', use random init".format(args.pretrain))
    else:
        args.logger.info("=> train from scratch")

    torch.backends.cudnn.benchmark = True

    # tensorboard plot tools
    if args.print:
        writer_train = SummaryWriter(logdir=os.path.join(args.img_path, 'pretrain'))
        args.train_plotter = TB.PlotterThread(writer_train)

    scheduler = lr_scheduler.MultiStepLR(optimizer, args.schedule, gamma=0.1, last_epoch=args.start_epoch - 1)

    assert args.save_freq % args.eval_freq == 0
    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs): # 5 warmup epochs
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        _, train_acc = train_one_epoch(train_loader, model, criterion, optimizer, scheduler, transform_train_cuda,
                                       epoch, args)

        if ((epoch + 1) % args.eval_freq == 0) or (epoch == args.epochs - 1):
            # save check_point on rank==0 worker
            if not args.distributed \
                    or (not args.multiprocessing_distributed and args.rank == 0) \
                    or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):

                is_best = train_acc > best_acc
                best_acc = max(train_acc, best_acc)
                state_dict = model_without_ddp.state_dict()

                save_dict = {
                    'epoch': epoch,
                    'state_dict': state_dict,
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'iteration': args.iteration}
                save_checkpoint(save_dict, is_best, gap=0,
                                filename=os.path.join(args.model_path, 'epoch%d.pth.tar' % epoch),
                                keep_all='k400' in args.dataset, is_save=((epoch + 1) % args.save_freq == 0), save_latest=True)

    args.logger.info('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))
    time.sleep(5)
    sys.exit(0)


def train_one_epoch(data_loader, model, optimizer, scheduler, transforms_cuda, epoch, args,
                    model_head=None, head_optimizer=None, head_scheduler=None):
    batch_time_meter = AverageMeter('Time', ':.2f')
    data_time_meter = AverageMeter('Data', ':.2f')
    losses_meter = AverageMeter('VLoss', ':.4f')
    top1_meter = AverageMeter('Vacc@1', ':.4f')

    losses_meters_dict = OrderedDict()
    losses_meters_dict['clip'] = losses_meter
    acc_meters_dict = OrderedDict()
    acc_meters_dict['clip'] = top1_meter

    progress_meters_list = [batch_time_meter, data_time_meter] + \
                            list(losses_meters_dict.values()) + list(acc_meters_dict.values())

    progress = ProgressMeter(
        len(data_loader),
        progress_meters_list,
        prefix='Epoch:[{}/{}] lr:{} '.format(epoch, args.epochs, args.lr), logger=args.logger)

    model.train()

    def tr(x):
        B = x.size(0)
        return transforms_cuda(x).view(B, 3, args.num_seq*args.n_proto, args.seq_len, args.img_dim, args.img_dim) \
            .transpose(1, 2).contiguous()

    tic = time.time()
    end = time.time()

    for idx, batch in tqdm(enumerate(data_loader), total=len(data_loader), disable=True):
        data_time_meter.update(time.time() - end)
        input_seq = batch['seq']
        orig_input_seq = input_seq

        B = input_seq.size(0)
        input_seq = tr(input_seq.cuda(non_blocking=True))
        ret = model(input_seq)
        loss = 0

        if 'clip_contrast_loss' in ret:
            output = ret['clip_logits']
            target = ret['clip_labels']
            top1, top5 = calc_topk_accuracy(output, target, (1, 5))
            loss = ret['clip_contrast_loss']

            top1_meter.update(top1.item(), B)
            losses_meter.update(loss.item(), B)

        keys = list(ret.keys())
        extra_loss_keys = [key for key in keys if 'loss' in key and 'clip' not in key ]
        contrast_extra_loss_keys = [key for key in extra_loss_keys if 'contrast_loss' in key]
        misc_extra_loss_keys = [key for key in extra_loss_keys if key not in contrast_extra_loss_keys]

        for key in contrast_extra_loss_keys:
            # calculate loss
            prefix = key.replace("_contrast_loss", "")
            output = ret[f'{prefix}_logits']
            target = ret[f'{prefix}_labels']
            top1 = calc_topk_accuracy(output, target, (1,))[0]
            Loss = ret[f'{prefix}_contrast_loss']
            if prefix not in losses_meters_dict:
                losses_meters_dict[prefix] = AverageMeter(f'{prefix}_loss', ':.3f')
                acc_meters_dict[prefix] = AverageMeter(f'{prefix}_acc', ':.3f')
                progress_meters_list = [batch_time_meter, data_time_meter] + \
                                       list(losses_meters_dict.values()) + list(acc_meters_dict.values())
                progress.meters = progress_meters_list
            losses_meters_dict[prefix].update(Loss.item(), B)
            acc_meters_dict[prefix].update(top1.item(), B)
            loss += Loss

        for key in misc_extra_loss_keys:
            # calculate loss
            prefix = key.replace("_loss", "")
            Loss = ret[key]
            if prefix not in losses_meters_dict:
                losses_meters_dict[prefix] = AverageMeter(f'{prefix}_loss', ':.3f')
                progress_meters_list = [batch_time_meter, data_time_meter] + \
                                       list(losses_meters_dict.values()) + list(acc_meters_dict.values())
                progress.meters = progress_meters_list
            losses_meters_dict[prefix].update(Loss.item(), B)
            loss += Loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        batch_time_meter.update(time.time() - end)
        end = time.time()

        if (idx + 1) % args.print_freq == 0:
            if args.print:
                # display logs
                progress.display(idx)
                for key, meter in losses_meters_dict.items():
                    args.train_plotter.add_data(f'local/{key}_loss', meter.local_avg,
                                                args.iteration)
                for key, meter in acc_meters_dict.items():
                    args.train_plotter.add_data(f'local/{key}_acc', meter.local_avg,
                                                args.iteration)
        args.iteration += 1

    avg_loss = sum(t.avg for t in losses_meters_dict.values())

    args.logger.info('Epoch: [{0}/{1}]\t'
                     'T-epoch:{t:.2f}\t'
                     'Loss:{loss:.4f}'.format(epoch, args.epochs,
                                                t=time.time() - tic,
                                                loss=avg_loss))

    if args.print:
        for key, meter in losses_meters_dict.items():
            args.train_plotter.add_data(f'global/{key}_loss', meter.avg,
                                        epoch)
        for key, meter in acc_meters_dict.items():
            args.train_plotter.add_data(f'local/{key}_acc', meter.avg,
                                        epoch)

    scheduler.step()
    args.lr = optimizer.param_groups[0]['lr']

    # only return contrastive loss result
    return losses_meter.avg, top1_meter.avg


def get_transform(mode, args):
    # null transform
    null_transform = transforms.Compose([
        A.Scale((128, 171)),
        A.RandomCrop(size=args.img_dim),
        A.ToTensor(),
    ])

    # basic transform
    base_transform_list = [
        A.Scale((128, 171)),
        A.RandomCrop(size=args.img_dim),
        A.ToTensor(),
        transforms.RandomApply([
            A.ColorJitter(0.8, 0.8, 0.8, 0.2, p=0.8, consistent=args.aug_temp_consist, seq_len=args.seq_len,
                          block=args.n_block, grad_consistent=args.aug_temp_grad_consist)], p=0.8),
        transforms.RandomApply([A.GaussianBlur([.1, 2.], seq_len=args.seq_len)], p=0.5)
    ]
    base_transform = transforms.Compose(base_transform_list)

    # same series extra transform
    same_series_transform_list = [
        A.Scale((128, 171)),
        A.RandomCrop(size=args.img_dim),
        A.ToTensor(),
        transforms.RandomApply([
            A.ColorJitter(0.8, 0.8, 0.8, 0.2, p=0.8, consistent=args.aug_temp_consist, seq_len=args.seq_len,
                          block=args.n_block, grad_consistent=args.aug_temp_grad_consist)], p=0.8),
        transforms.RandomApply([A.GaussianBlur([.1, 2.], seq_len=args.seq_len)], p=0.5)
    ]
    same_series_transform = transforms.Compose(same_series_transform_list)

    weights = [
        [0.2, 0.8, 0],
        [0, 1.0, 0],
        [0, 0., 1.0]
    ]
    transform = A.MultiRandomizedTransform([null_transform, base_transform, same_series_transform], args.seq_len,
                                           weights=weights)

    args.logger.info(f"finishing {mode}'s transform")
    return transform


def get_data(transform, mode, args):
    args.logger.info('Loading data for "%s" mode' % mode)

    if args.dataset == 'ucf101-2clip-stage-prototype':
        dataset = UCF101LMDB_2CLIP_Stage_Prototype(aug_series=args.aug_series, rand_flip=args.rand_flip, mode=mode, transform=transform,
                                   num_frames=args.seq_len, ds=args.ds, return_label=True)
    elif args.dataset == 'k400-2clip-stage-prototype':
        dataset = K400LMDB_2CLIP_Stage_Prototype(aug_series=args.aug_series, rand_flip=args.rand_flip, mode=mode, transform=transform,
                                   num_frames=args.seq_len, ds=args.ds, return_label=True)
    else:
        raise NotImplementedError()

    return dataset


def get_dataloader(dataset, mode, args):
    args.logger.info('Creating data loaders for "%s" mode' % mode)
    train_sampler = data.distributed.DistributedSampler(dataset, shuffle=True) if args.distributed else None
    if mode == 'train':
        data_loader = FastDataLoader(
            dataset, batch_size=args.batch_size, shuffle=(train_sampler is None and not args.visualize),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    elif mode == 'val':
        data_loader = FastDataLoader(
            dataset, batch_size=8, shuffle=False,
            num_workers=8, pin_memory=True, sampler=None, drop_last=False)
    else:
        raise NotImplementedError
    args.logger.info('"%s" dataset has size: %d' % (mode, len(dataset)))
    return data_loader


def set_path(args):
    if args.resume:
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    elif args.test:
        exp_path = os.path.dirname(os.path.dirname(args.test))
    else:
        exp_path = 'log/{args.prefix}/pretrain/{args.name_prefix}'.format(
            args=args)
    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    log_file = os.path.join(exp_path, 'log')

    print(f"<<<<<<< exp_path {exp_path}")

    if args.visualize:
        img_path = os.path.join(exp_path, 'vis_img')
        log_file = os.path.join(exp_path, 'vis_log')

    if not os.path.exists(img_path):
        if args.rank == 0 or not args.distributed:
            os.makedirs(img_path)
    if not os.path.exists(model_path):
        if args.rank == 0 or not args.distributed:
            os.makedirs(model_path)
    return img_path, model_path, exp_path, log_file


if __name__ == '__main__':
    '''
    Three ways to run (recommend first one for simplicity):
    1. CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
       --nproc_per_node=2 main_nce.py (do not use multiprocessing-distributed) ...

       This mode overwrites WORLD_SIZE, overwrites rank with local_rank

    2. CUDA_VISIBLE_DEVICES=0,1 python main_nce.py \
       --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 ...

       Official methods from fb/moco repo
       However, lmdb is NOT supported in this mode, because ENV cannot be pickled in mp.spawn

    3. using SLURM scheduler
    '''
    args = parse_args()
    main(args)