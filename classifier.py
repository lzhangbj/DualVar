import os
import sys

sys.path.append('../CoCLR/')
import argparse
import time
import numpy as np
import random
import pickle
from tqdm import tqdm
from PIL import Image
import json
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch.distributed as dist
import builtins

plt.switch_backend('agg')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchvision import transforms
import torch.nn.functional as F

from model.classifier import LinearClassifier
from dataset.local_dataset import *
from utils.utils import AverageMeter, save_checkpoint, calc_topk_accuracy, \
    ProgressMeter, neq_load_customized, worker_init_fn, FastDataLoader
from utils.logging import get_root_logger
import utils.augmentation as A
import utils.transforms as T
import utils.tensorboard_utils as TB


def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--net', default='myrealr21d', type=str)
    parser.add_argument('--model', default='linclr', type=str)
    parser.add_argument('--num_fc', default=1, type=int, help='number of fc')
    parser.add_argument('--train_what', default='ft', type=str)
    parser.add_argument('--use_dropout', action='store_true', help='use dropout')
    parser.add_argument('--use_norm', action='store_true', help='use dropout')
    parser.add_argument('--use_bn', action='store_true', help='use dropout')
    parser.add_argument('--dropout', default=1., type=float, help='dropout')
    parser.add_argument('--ft-mode', action='store_true', help='use ft mode')
    parser.add_argument('--with_color_jitter', action='store_true', help='use color jittering')
    # dataset
    parser.add_argument('--dataset', default='ucf101', type=str)
    parser.add_argument('--which_split', default=1, type=int)
    parser.add_argument('--seq_len', default=16, type=int, help='number of frames in each video block')
    parser.add_argument('--num_seq', default=1, type=int, help='number of video blocks')
    parser.add_argument('--ds', default=4, type=int, help='frame down sampling rate')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size per GPU')
    parser.add_argument('--img_resize_dim', default=128, type=int)
    parser.add_argument('--img_dim', default=112, type=int)
    # optimizer
    parser.add_argument('--optim', default='sgd', type=str)
    parser.add_argument('--lr', default=5e-2, type=float, help='learning rate')
    parser.add_argument('--schedule', default=[10,20,30,40], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--gpu', default=None, type=int)
    # log
    parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
    parser.add_argument('--eval_freq', default=1, type=int)
    parser.add_argument('--save_freq', default=10, type=int)
    # exp settings
    parser.add_argument('--prefix', default='linclr', type=str)
    parser.add_argument('--name_prefix', default='', type=str)
    parser.add_argument('-j', '--workers', default=8, type=int)
    parser.add_argument('--dirname', default=None, type=str, help='dirname for feature')
    # mode
    parser.add_argument('--resume', default='', type=str, help='path of model to resume')
    parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
    parser.add_argument('--test', default='', type=str, help='path of model to load and pause')
    parser.add_argument('--retrieval', action='store_true', help='path of model to ucf retrieval')
    parser.add_argument('--center_crop', action='store_true')
    parser.add_argument('--five_crop', action='store_true')
    parser.add_argument('--ten_crop', action='store_true')
    parser.add_argument('--temporal_ten_clip', action='store_true')
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

    parser.add_argument('--aug_crop', action='store_true')
    parser.add_argument('--rand_flip', action='store_true')

    args = parser.parse_args()
    return args

def main(args):

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


class DistLogger():
    def __init__(self, log_file, print=True):
        self.print = print
        if print:
            self.logger = get_root_logger(log_file)

    def info(self, content):
        if self.print:
            self.logger.info(content)


def main_worker(gpu, ngpus_per_node, args):
    best_acc = 0
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
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
        args.rank = torch.distributed.get_rank()
        print(args.rank)
    else:
        args.rank = 0

    args.print = args.gpu == 0 or not args.distributed
    print(args.print)
    # suppress printing if not master
    if args.rank != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    args.img_path, args.model_path, args.exp_path, args.log_file = set_path(args)
    args.logger = DistLogger(log_file=args.log_file, print=args.print)

    args.logger.info('=> Effective BatchSize = %d' % args.real_batch_size)

    ### classifier model ###
    num_class_dict = {'ucf101': 101, 'ucf101-10clip': 101, 'hmdb51': 51, 'hmdb51-10clip': 51}
    args.num_class = num_class_dict[args.dataset]

    assert args.save_freq % args.eval_freq == 0, "save freq must be divided by eval freq "

    # if args.ft_mode:
    #     assert not args.use_bn and not args.use_dropout and not args.use_norm

    if args.model == 'linclr':
        model = LinearClassifier(
            network=args.net,
            num_class=args.num_class,
            dropout=args.dropout,
            use_dropout=args.use_dropout,
            use_final_bn=args.use_bn,
            use_l2_norm=args.use_norm)
        message = 'Classifier to %d classes with %s backbone;' % (args.num_class, args.net)
        if args.use_norm: message += ' + L2Norm'
        if args.use_bn: message += ' + final BN'
        if args.use_dropout: message += ' + dropout %f' % args.dropout

        args.logger.info(message)
    else:
        raise NotImplementedError

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
            model_without_dp = model.module
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            model_without_dp = model.module
    else:
        if args.gpu is None:
            args.gpu = 0
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        model_without_dp = model
    device = torch.device(f'cuda:{args.gpu}')

    ### optimizer ###
    if args.train_what == 'last':
        args.logger.info('=> [optimizer] only train last layer')
        params = []
        for name, param in model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
            else:
                params.append({'params': param})
    else:  # train all
        params = []
        args.logger.info('=> [optimizer] finetune all layer')
        for name, param in model.named_parameters():
            params.append({'params': param})

    if args.train_what == 'last':
        args.logger.info('\n===========Check Grad============')
        for name, param in model.named_parameters():
            if param.requires_grad:
                args.logger.info(name, param.requires_grad)
        args.logger.info('=================================\n')

    if args.optim == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, weight_decay=args.wd, momentum=0.9)
    else:
        raise NotImplementedError

    args.logger.info(f" => use {args.optim} optimizer")

    ce_loss = nn.CrossEntropyLoss()
    args.iteration = 1

    ### test: higher priority ###
    if args.test:
        if os.path.isfile(args.test):
            args.logger.info("=> loading testing checkpoint '{}'".format(args.test))
            checkpoint = torch.load(args.test, map_location=torch.device('cpu'))
            epoch = checkpoint['epoch']
            state_dict = checkpoint['state_dict']

            if args.retrieval:  # if directly test on pretrained network
                new_dict = {}
                for k, v in state_dict.items():
                    k = k.replace('encoder_q.0.', 'backbone.').replace('linear_fc', 'pretrain_fc')
                    new_dict[k] = v
                state_dict = new_dict
            try:
                model_without_dp.load_state_dict(state_dict)
            except:
                neq_load_customized(model_without_dp, state_dict, verbose=True, args=args)

        else:
            args.logger.info("[Warning] no checkpoint found at '{}'".format(args.test))
            epoch = 0
            args.logger.info("[Warning] if test random init weights, press c to continue")
            # import ipdb; ipdb.set_trace()

        args.logger.info('args=\n\t\t' + '\n\t\t'.join(['%s:%s' % (str(k), str(v)) for k, v in vars(args).items()]))

        transform_test_cuda = transforms.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], channel=1)])

        if args.retrieval:
            assert not args.distributed
            test_retrieval(model, ce_loss, transform_test_cuda, device, epoch, args)
        elif args.center_crop or args.five_crop or args.ten_crop:
            assert not args.distributed
            transform = get_transform('test', args)
            test_dataset = get_data(transform, 'test', args)
            test_10crop(test_dataset, model, ce_loss, transform_test_cuda, device, epoch, args)
        elif args.temporal_ten_clip:
            assert not args.distributed
            transform = get_transform('test', args)
            test_dataset = get_data(transform, 'test', args)
            temporal_test_10clip(test_dataset, model, ce_loss, transform_test_cuda, device, epoch, args)
        else:
            raise NotImplementedError

        sys.exit(0)

    ### data ###
    transform_train = get_transform('train', args)
    train_loader = get_dataloader(get_data(transform_train, 'train', args), 'train', args)
    transform_val = get_transform('val', args)
    val_loader = get_dataloader(get_data(transform_val, 'val', args), 'val', args)

    transform_train_cuda = transforms.Compose([
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], channel=1)])  # ImageNet
    transform_val_cuda = transforms.Compose([
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], channel=1)])  # ImageNet

    args.logger.info('===================================')

    ### restart training ###
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            args.iteration = checkpoint['iteration']
            best_acc = checkpoint['best_acc']
            state_dict = checkpoint['state_dict']

            try:
                model_without_dp.load_state_dict(state_dict)
            except:
                args.logger.info('[WARNING] resuming training with different weights')
                neq_load_customized(model_without_dp, state_dict, verbose=True, args=args)
            args.logger.info("=> load resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                args.logger.info('[WARNING] failed to load optimizer state, initialize optimizer')
        else:
            args.logger.info("[Warning] no checkpoint found at '{}', use random init".format(args.resume))

    elif args.pretrain:
        if os.path.isfile(args.pretrain):
            checkpoint = torch.load(args.pretrain, map_location='cpu')
            state_dict = checkpoint['state_dict']

            new_dict = {}
            for k, v in state_dict.items():
                k = k.replace('encoder_q.0.', 'backbone.').replace('final_fc', 'pretrain_fc')
                new_dict[k] = v
            state_dict = new_dict

            try:
                model_without_dp.load_state_dict(state_dict)
            except:
                neq_load_customized(model_without_dp, state_dict, verbose=True, args=args)
            args.logger.info(
                "=> loaded pretrained checkpoint '{}' (epoch {})".format(args.pretrain, checkpoint['epoch']))
        else:
            args.logger.info("[Warning] no checkpoint found at '{}', use random init".format(args.pretrain))
            raise NotImplementedError

    else:
        args.logger.info("=> train from scratch")

    torch.backends.cudnn.benchmark = True

    # plot tools
    writer_val = SummaryWriter(logdir=os.path.join(args.img_path, 'val'))
    writer_train = SummaryWriter(logdir=os.path.join(args.img_path, 'train'))
    args.val_plotter = TB.PlotterThread(writer_val)
    args.train_plotter = TB.PlotterThread(writer_train)

    args.logger.info('args=\n\t\t' + '\n\t\t'.join(['%s:%s' % (str(k), str(v)) for k, v in vars(args).items()]))

    # main loop
    for epoch in range(args.start_epoch, args.epochs):
        np.random.seed(epoch)
        random.seed(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        train_one_epoch(train_loader, model, ce_loss, optimizer, transform_train_cuda, device, epoch, args)

        if (epoch + 1) % args.eval_freq == 0:
            _, val_acc = validate(val_loader, model, ce_loss, transform_val_cuda, device, epoch, args)

            # save check_point
            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            state_dict = model_without_dp.state_dict()
            save_dict = {
                'epoch': epoch,
                'state_dict': state_dict,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'iteration': args.iteration}
            save_checkpoint(save_dict, is_best, 0,  # make gap = 0, prevent delete previous saved dict
                            filename=os.path.join(args.model_path, 'epoch%d.pth.tar' % epoch),
                            keep_all=False, is_save=((epoch + 1) % args.save_freq == 0))

    args.logger.info('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))
    time.sleep(5)
    sys.exit(0)


def train_one_epoch(data_loader, model, criterion, optimizer, transforms_cuda, device, epoch, args):
    batch_time = AverageMeter('Time', ':.2f')
    data_time = AverageMeter('Data', ':.2f')
    losses = AverageMeter('Loss', ':.4f')
    top1_meter = AverageMeter('acc@1', ':.4f')
    top5_meter = AverageMeter('acc@5', ':.4f')

    args.lr = optimizer.param_groups[0]['lr']
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses, top1_meter, top5_meter],
        prefix='Epoch:[{}/{}] lr:{} '.format(epoch, args.epochs, args.lr), logger=args.logger)

    if args.train_what == 'last':
        model.eval()  # totally freeze BN in backbone
    else:
        model.train()

    if args.use_bn:
        if args.distributed:
            model.module.final_bn.train()
        else:
            model.final_bn.train()

    end = time.time()
    tic = time.time()

    def tr(x):  # transformation on tensor
        B = x.size(0)
        return transforms_cuda(x).view(B, 3, args.num_seq, args.seq_len, args.img_dim, args.img_dim) \
            .transpose(1, 2).contiguous()

    for idx, batch in enumerate(data_loader):
        data_time.update(time.time() - end)

        input_seq = batch['seq']
        target = batch['vid']
        B = input_seq.size(0)
        input_seq = tr(input_seq.to(device, non_blocking=True))
        target = target.to(device, non_blocking=True)

        input_seq = input_seq.squeeze(1)  # num_seq is always 1, seqeeze it
        logit, _ = model(input_seq)
        loss = criterion(logit, target)
        top1, top5 = calc_topk_accuracy(logit, target, (1, 5))

        losses.update(loss.item(), B)
        top1_meter.update(top1.item(), B)
        top5_meter.update(top5.item(), B)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % args.print_freq == 0:
            progress.display(idx)

            args.train_plotter.add_data('local/loss', losses.local_avg, args.iteration)
            args.train_plotter.add_data('local/top1', top1_meter.local_avg, args.iteration)
            args.train_plotter.add_data('local/top5', top5_meter.local_avg, args.iteration)

        args.iteration += 1

    args.logger.info('Epoch: [{0}][{1}/{2}]\t'
                     'T-epoch:{t:.2f}\t'.format(epoch, idx, len(data_loader), t=time.time() - tic))

    args.train_plotter.add_data('global/loss', losses.avg, epoch)
    args.train_plotter.add_data('global/top1', top1_meter.avg, epoch)
    args.train_plotter.add_data('global/top5', top5_meter.avg, epoch)

    args.logger.info('train Epoch: [{0}][{1}/{2}]\t'
                     'T-epoch:{t:.2f}\t'.format(epoch, idx, len(data_loader), t=time.time() - tic))

    return losses.avg, top1_meter.avg


def validate(data_loader, model, criterion, transforms_cuda, device, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_meter = AverageMeter('acc@1', ':.4f')
    top5_meter = AverageMeter('acc@5', ':.4f')

    model.eval()

    def tr(x):
        B = x.size(0)
        return transforms_cuda(x).view(B, 3, args.num_seq, args.seq_len, args.img_dim, args.img_dim) \
            .transpose(1, 2).contiguous()

    with torch.no_grad():
        end = time.time()
        for idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_seq = batch['seq']
            target = batch['vid']
            B = input_seq.size(0)
            input_seq = tr(input_seq.to(device, non_blocking=True))
            target = target.to(device, non_blocking=True)

            input_seq = input_seq.squeeze(1)  # num_seq is always 1, seqeeze it
            logit, _ = model(input_seq)
            loss = criterion(logit, target)
            top1, top5 = calc_topk_accuracy(logit, target, (1, 5))

            losses.update(loss.item(), B)
            top1_meter.update(top1.item(), B)
            top5_meter.update(top5.item(), B)
            batch_time.update(time.time() - end)
            end = time.time()

    args.val_plotter.add_data('global/loss', losses.avg, epoch)
    args.val_plotter.add_data('global/top1', top1_meter.avg, epoch)
    args.val_plotter.add_data('global/top5', top5_meter.avg, epoch)

    args.logger.info('val Epoch: [{0}]\t'
                     'Loss: {loss.avg:.4f} Acc@1: {top1.avg:.4f} Acc@5: {top5.avg:.4f}\t'
                     .format(epoch, loss=losses, top1=top1_meter, top5=top5_meter))

    return losses.avg, top1_meter.avg


def test_10crop(dataset, model, criterion, transforms_cuda, device, epoch, args):
    prob_dict = {}
    model.eval()

    # aug_list: 1,2,3,4,5 = topleft, topright, bottomleft, bottomright, center
    # flip_list: 0,1 = raw, flip
    if args.center_crop:
        args.logger.info('Test using center crop')
        aug_list = [5];
        flip_list = [0];
        title = 'center'
    if args.five_crop:
        args.logger.info('Test using 5 crop')
        aug_list = [5, 1, 2, 3, 4];
        flip_list = [0];
        title = 'five'
    if args.ten_crop:
        args.logger.info('Test using 10 crop')
        aug_list = [5, 1, 2, 3, 4];
        flip_list = [0, 1];
        title = 'ten'

    # def tr(x):
    #     B = x.size(0); assert B == 1
    #     num_test_sample = x.size(2)//(args.seq_len*args.num_seq)
    #     return transforms_cuda(x)\
    #     .view(3,num_test_sample,args.num_seq,args.seq_len,args.img_dim,args.img_dim).permute(1,2,0,3,4,5)
    #     # (n_test_sample, num_seq/1, c, seq_len, H, W)

    def tr(x):
        B = x.size(0)
        num_test_sample = x.size(2) // (args.seq_len * args.num_seq)
        return transforms_cuda(x) \
            .view(B, 3, num_test_sample, args.num_seq, args.seq_len, args.img_dim, args.img_dim) \
            .permute(0, 2, 3, 1, 4, 5, 6) \
            .view(B * num_test_sample, args.num_seq, 3, args.seq_len, args.img_dim, args.img_dim)
        # (B*n_test_sample, num_seq/1, c, seq_len, H, W)

    with torch.no_grad():
        end = time.time()
        # for loop through 10 types of augmentations, then average the probability
        for flip_idx in flip_list:
            for aug_idx in aug_list:
                args.logger.info('Aug type: %d; flip: %d' % (aug_idx, flip_idx))
                if flip_idx == 0:
                    transform = transforms.Compose([
                        A.RandomHorizontalFlip(command='left'),
                        A.Scale(size=args.img_resize_dim),
                        A.FiveCrop(size=args.img_dim, where=aug_idx),
                        A.ToTensor()])
                else:
                    transform = transforms.Compose([
                        A.RandomHorizontalFlip(command='right'),
                        A.Scale(size=args.img_resize_dim),
                        A.FiveCrop(size=args.img_dim, where=aug_idx),
                        A.ToTensor()])

                dataset.transform = transform
                dataset.return_path = True
                dataset.return_label = True
                test_sampler = data.SequentialSampler(dataset)
                data_loader = data.DataLoader(dataset,
                                              batch_size=1,
                                              sampler=test_sampler,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              worker_init_fn=worker_init_fn,
                                              pin_memory=True)

                for idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
                    input_seq = batch['seq']
                    input_seq = tr(input_seq.to(device, non_blocking=True))
                    input_seq = input_seq.squeeze(1)  # num_seq is always 1, seqeeze it
                    logit, _ = model(input_seq)
                    # average probability along the temporal window
                    prob_mean = F.softmax(logit, dim=-1).mean(0, keepdim=True)  # (1, num_class)

                    target = batch['vid']
                    vname = batch['vpath']
                    vname = vname[0]
                    if vname not in prob_dict.keys():
                        prob_dict[vname] = {'mean_prob': [], }
                    prob_dict[vname]['mean_prob'].append(prob_mean[0])

                if (title == 'ten') and (flip_idx == 0) and (aug_idx == 5):
                    args.logger.info('center-crop result:')
                    acc_1 = summarize_probability(prob_dict,
                                                  data_loader.dataset.encode_action, 'center', args)
                    args.logger.info('center-crop:')
                    args.logger.info('test Epoch: [{0}]\t'
                                     'Mean: Acc@1: {acc[0].avg:.4f} Acc@5: {acc[1].avg:.4f}'
                                     .format(epoch, acc=acc_1))

            if (title == 'ten') and (flip_idx == 0):
                args.logger.info('five-crop result:')
                acc_5 = summarize_probability(prob_dict,
                                              data_loader.dataset.encode_action, 'five', args)
                args.logger.info('five-crop:')
                args.logger.info('test Epoch: [{0}]\t'
                                 'Mean: Acc@1: {acc[0].avg:.4f} Acc@5: {acc[1].avg:.4f}'
                                 .format(epoch, acc=acc_5))

    args.logger.info('%s-crop result:' % title)
    acc_final = summarize_probability(prob_dict,
                                      data_loader.dataset.encode_action, 'ten', args)
    args.logger.info('%s-crop:' % title)
    args.logger.info('test Epoch: [{0}]\t'
                     'Mean: Acc@1: {acc[0].avg:.4f} Acc@5: {acc[1].avg:.4f}'
                     .format(epoch, acc=acc_final))
    sys.exit(0)


def temporal_test_10clip(dataset, model, criterion, transforms_cuda, device, epoch, args):
    prob_dict = {}
    per_prob_list = []
    cls_prob_dict = {}
    model.eval()

    assert args.num_seq == 10

    # aug_list: 1,2,3,4,5 = topleft, topright, bottomleft, bottomright, center
    # flip_list: 0,1 = raw, flip
    args.logger.info('Test using temporal 10 center clip crop')
    title = 'temporal_10_clip'

    def tr(x):
        B = x.size(0)
        num_test_sample = x.size(2) // (args.seq_len * args.num_seq)
        assert num_test_sample == 1
        return transforms_cuda(x) \
            .view(B, 3, args.num_seq, args.seq_len, args.img_dim, args.img_dim) \
            .permute(0, 2, 1, 3, 4, 5).contiguous() \
            .view(B * args.num_seq, 3, args.seq_len, args.img_dim, args.img_dim)
        # (B*10, c, seq_len, H, W)

    with torch.no_grad():
        end = time.time()
        # temporally uniform sample 10 clips, then average result
        base_transform = transforms.Compose([
            # A.RandomHorizontalFlip(command='left'),
            A.Scale(size=args.img_resize_dim),
            A.CenterCrop(size=args.img_dim),
            A.ToTensor()])
        if args.aug_crop and args.img_dim==112:
            base_transform = transforms.Compose([
                # A.RandomHorizontalFlip(command='left'),
                A.Scale(size=(128, 171)),
                A.CenterCrop(size=args.img_dim),
                A.ToTensor()])

        transform = A.MultipleClipTransform([base_transform, ] * 10, seq_len=args.seq_len)

        dataset.transform = transform
        dataset.return_path = True
        dataset.return_label = True
        test_sampler = data.SequentialSampler(dataset)
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=test_sampler,
                                      shuffle=False,
                                      num_workers=args.workers,
                                      worker_init_fn=worker_init_fn,
                                      pin_memory=True)


        for idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_seq = batch['seq']  # (B, C, 10, seq_len, )
            B = input_seq.size(0)
            input_seq = tr(input_seq.to(device, non_blocking=True))
            logit, _ = model(input_seq)
            # average probability along the temporal window
            prob_per = F.softmax(logit, dim=-1).view(B, 10, -1)  # (B, num_class)
            per_prob_list.append(prob_per)
            prob_mean = prob_per.mean(1, keepdim=False)  # (B, num_class)

            target = batch['vid']
            vnames = batch['vpath']
            for i, vname in enumerate(vnames):
                if vname not in prob_dict.keys():
                    prob_dict[vname] = {'mean_prob': []}
                prob_dict[vname]['mean_prob'].append(prob_mean[i])

                label = data_loader.dataset.decode_action(target[i].item())
                if label not in cls_prob_dict.keys():
                    cls_prob_dict[label] = {'mean_prob': [],}
                cls_prob_dict[label]['mean_prob'].append(prob_mean[i])

    args.logger.info('<<<<<< temporal uniform 10 crop result: >>>>>>>>> ')
    acc_1 = summarize_probability(prob_dict,
                                data_loader.dataset.encode_action, title, args)
    args.logger.info('######## temporal uniform 10 crop classwise result: #########')
    acc_1 = summarize_classwise_probability(cls_prob_dict,
                                  data_loader.dataset.encode_action, title, args)
    sys.exit(0)


def summarize_classwise_probability(prob_dict, action_to_idx, title, args):
    acc = [AverageMeter(), AverageMeter()]
    stat = {}
    for action_name, item in tqdm(prob_dict.items(), total=len(prob_dict)):
        target = action_to_idx(action_name)
        mean_prob = torch.stack(item['mean_prob'], 0)  # .mean(0) (n, num_class)
        mean_top1, mean_top5 = calc_topk_accuracy(mean_prob,
                                                  torch.LongTensor([target, ] * len(item['mean_prob'])).cuda(), (1, 5))
        stat[action_name] = {'mean_prob': mean_prob.tolist()}
        acc[0].update(mean_top1.item(), 1)
        acc[1].update(mean_top5.item(), 1)

        args.logger.info('{action_name}Mean: Acc@1: {acc[0].avg:.4f} Acc@5: {acc[1].avg:.4f}'
                        .format(action_name=action_name, acc=acc))

    with open(os.path.join(os.path.dirname(args.test),
                           '%s-classwise_prob-%s.json' % (os.path.basename(args.test), title)), 'w') as fp:
        json.dump(stat, fp)
    return acc


def summarize_probability(prob_dict, action_to_idx, title, args):
    acc = [AverageMeter(), AverageMeter()]
    stat = {}
    for vname, item in tqdm(prob_dict.items(), total=len(prob_dict)):
        try:
            action_name = vname.split('/')[-3]
        except:
            action_name = vname.split('/')[-2]
        target = action_to_idx(action_name)
        mean_prob = torch.stack(item['mean_prob'], 0)  # .mean(0) (n, num_class)
        mean_top1, mean_top5 = calc_topk_accuracy(mean_prob,
                                                  torch.LongTensor([target, ] * len(item['mean_prob'])).cuda(), (1, 5))
        stat[vname] = {'mean_prob': mean_prob.tolist()}
        acc[0].update(mean_top1.item(), 1)
        acc[1].update(mean_top5.item(), 1)

    args.logger.info('Mean: Acc@1: {acc[0].avg:.4f} Acc@5: {acc[1].avg:.4f}'
                     .format(acc=acc))

    with open(os.path.join(os.path.dirname(args.test),
                           '%s-prob-%s.json' % (os.path.basename(args.test), title)), 'w') as fp:
        json.dump(stat, fp)
    return acc


def test_retrieval(model, criterion, transforms_cuda, device, epoch, args):
    model.eval()
    # (temporarily sample 10 clips)
    assert args.num_seq == 10

    def tr(x):
        B = x.size(0)
        test_sample = x.size(2) // (args.seq_len * args.num_seq)
        assert test_sample == 1
        return transforms_cuda(x) \
            .view(B, 3, args.num_seq, args.seq_len, args.img_dim, args.img_dim) \
            .permute(0, 2, 1, 3, 4, 5).contiguous() \
            .view(B * args.num_seq, 3, args.seq_len, args.img_dim, args.img_dim) \
            .contiguous()

    with torch.no_grad():
        # transform = transforms.Compose([
        #     A.CenterCrop(size=(224, 224)),
        #     A.Scale(size=(args.img_dim, args.img_dim)),
        #     A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.3, consistent=True),
        #     A.ToTensor()])

        transform = transforms.Compose([
            A.Scale(size=args.img_resize_dim),
            A.CenterCrop(size=args.img_dim),
            A.ToTensor()])
        if args.aug_crop and args.img_dim==112:
            transform = transforms.Compose([
                A.Scale((128, 171)),
                A.CenterCrop(size=args.img_dim),
                A.ToTensor()])

        if args.dataset == 'ucf101':
            d_class = UCF101_10CLIP
        elif args.dataset == 'hmdb51':
            d_class = HMDB51_10CLIP
        else:
            raise NotImplementedError()

        train_dataset = d_class(mode='train',
                                transform=transform,
                                num_frames=args.seq_len,
                                ds=args.ds,
                                which_split=1,
                                return_label=True,
                                return_path=True)
        args.logger.info('train dataset size: %d' % len(train_dataset))

        test_dataset = d_class(mode='test',
                               transform=transform,
                               num_frames=args.seq_len,
                               ds=args.ds,
                               which_split=1,
                               return_label=True,
                               return_path=True)
        args.logger.info('test dataset size: %d' % len(test_dataset))

        train_sampler = data.SequentialSampler(train_dataset)
        test_sampler = data.SequentialSampler(test_dataset)

        train_loader = data.DataLoader(train_dataset,
                                       batch_size=args.batch_size,
                                       sampler=train_sampler,
                                       shuffle=False,
                                       num_workers=args.workers,
                                       worker_init_fn=worker_init_fn,
                                       pin_memory=True)
        test_loader = data.DataLoader(test_dataset,
                                      batch_size=args.batch_size,
                                      sampler=test_sampler,
                                      shuffle=False,
                                      num_workers=args.workers,
                                      worker_init_fn=worker_init_fn,
                                      pin_memory=True)
        if args.dirname is None:
            dirname = 'feature'
        else:
            dirname = args.dirname

        try:
            os.makedirs(os.path.join(os.path.dirname(args.test), dirname))
        except:
            pass

        ############# test features
        args.logger.info('Computing test set feature ... ')
        test_feature = []
        test_per_feature = []
        test_label = []
        test_vname = []
        sample_id = 0
        for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            input_seq = batch['seq'].to(device, non_blocking=True)
            B = input_seq.size(0)
            input_seq = tr(input_seq)
            current_target = batch['vid']
            vnames = batch['vname']
            current_target = current_target.to(device, non_blocking=True)

            test_sample = input_seq.size(0)
            # input_seq = input_seq.squeeze(1)
            logit, feature = model(input_seq)
            per_feature = feature.view(B, args.num_seq, feature.size(-1))
            feature = per_feature.mean(dim=1)

            test_feature.append(feature)
            test_per_feature.append(per_feature)

            for i in range(B):
                sample_id += 1
                test_label.append(current_target[i])
                test_vname.append(vnames[i])

        test_feature = torch.cat(test_feature, dim=0)
        test_label = torch.tensor(test_label).long().to(device)

        args.logger.info(test_feature.size())
        # test_label = torch.cat(test_label, dim=0)
        torch.save(test_feature,
                   os.path.join(os.path.dirname(args.test), dirname, '%s_test_feature.pth.tar' % args.dataset))
        torch.save(test_label,
                   os.path.join(os.path.dirname(args.test), dirname, '%s_test_label.pth.tar' % args.dataset))
        with open(os.path.join(os.path.dirname(args.test), dirname, '%s_test_vname.pkl' % args.dataset),
                  'wb') as fp:
            pickle.dump(test_vname, fp)

        test_per_feature = torch.cat(test_per_feature, dim=0)
        torch.save(test_per_feature,
                   os.path.join(os.path.dirname(args.test), dirname, '%s_test_per_feature.pth.tar' % args.dataset))

        ############## train features
        args.logger.info('Computing train set feature ... ')
        train_feature = []
        train_per_feature = []
        train_label = []
        train_vname = []
        sample_id = 0
        for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            input_seq = batch['seq']
            B = input_seq.size(0)
            input_seq = input_seq.to(device, non_blocking=True)
            input_seq = tr(input_seq)
            current_target = batch['vid']
            vnames = batch['vname']
            current_target = current_target.to(device, non_blocking=True)

            test_sample = input_seq.size(0)
            # input_seq = input_seq.squeeze(1)
            logit, feature = model(input_seq)
            per_feature = feature.view(B, args.num_seq, feature.size(-1))
            feature = per_feature.mean(dim=1)

            train_feature.append(feature)
            train_per_feature.append(per_feature)

            for i in range(B):
                train_label.append(current_target[i])
                train_vname.append(vnames[i])
                sample_id += 1

        train_feature = torch.cat(train_feature)
        train_label = torch.tensor(train_label).long().to(device)
        args.logger.info(train_feature.size())
        # train_label = torch.cat(train_label, dim=0)
        torch.save(train_feature,
                   os.path.join(os.path.dirname(args.test), dirname, '%s_train_feature.pth.tar' % args.dataset))
        torch.save(train_label,
                   os.path.join(os.path.dirname(args.test), dirname, '%s_train_label.pth.tar' % args.dataset))
        with open(os.path.join(os.path.dirname(args.test), dirname, '%s_train_vname.pkl' % args.dataset),
                  'wb') as fp:
            pickle.dump(train_vname, fp)

        train_per_feature = torch.cat(train_per_feature, dim=0)
        torch.save(train_per_feature,
                   os.path.join(os.path.dirname(args.test), dirname, '%s_train_per_feature.pth.tar' % args.dataset))

        ks = [1, 5, 10, 20, 50]
        NN_acc = []

        # centering
        test_feature = test_feature - test_feature.mean(dim=0, keepdim=True)
        train_feature = train_feature - train_feature.mean(dim=0, keepdim=True)

        # normalize
        test_feature = F.normalize(test_feature, p=2, dim=1)
        train_feature = F.normalize(train_feature, p=2, dim=1)

        # dot product
        sim = test_feature.matmul(train_feature.t())

        torch.save(sim, os.path.join(os.path.dirname(args.test), dirname, '%s_sim.pth.tar' % args.dataset))

        for k in ks:
            topkval, topkidx = torch.topk(sim, k, dim=1)
            acc = torch.any(train_label[topkidx] == test_label.unsqueeze(1), dim=1).float().mean().item()
            NN_acc.append(acc)
            args.logger.info('%dNN acc = %.4f' % (k, acc))

        args.logger.info('NN-Retrieval on %s:' % args.dataset)
        for k, acc in zip(ks, NN_acc):
            args.logger.info('\t%dNN acc = %.4f' % (k, acc))

        with open(os.path.join(os.path.dirname(args.test), dirname, '%s_test_vname.pkl' % args.dataset), 'rb') as fp:
            test_vname = pickle.load(fp)

        with open(os.path.join(os.path.dirname(args.test), dirname, '%s_train_vname.pkl' % args.dataset), 'rb') as fp:
            train_vname = pickle.load(fp)

        sys.exit(0)


def adjust_learning_rate(optimizer, epoch, args):
    '''Decay the learning rate based on schedule'''
    # stepwise lr schedule
    ratio = 0.1 if epoch in args.schedule else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * ratio


def get_transform(mode, args):
    if mode == 'train':
        transform_list = [
            A.Scale(args.img_resize_dim),
            A.RandomCrop(args.img_dim),
            A.ToTensor()
        ]
        if args.with_color_jitter:
            transform_list.append(A.ColorJitter(0.8, 0.8, 0.8, 0.2, p=0.8, consistent=True, seq_len=args.seq_len, block=1))
        if args.rand_flip:
            transform_list.insert(2, A.RandomHorizontalFlip(consistent=False, seq_len=args.seq_len))
        if args.aug_crop and args.img_dim==112:
            transform_list[0] = A.Scale(size=(128, 171))
            # transform_list[1] = A.RandomSizedCrop(consistent=True, size=112, p=1.0)


    elif mode == 'val' or mode == 'test':
        transform_list = [
            A.Scale(args.img_resize_dim),
            A.CenterCrop(size=args.img_dim),
            A.ToTensor()
        ]
        if args.aug_crop and args.img_dim==112:
            transform_list[0] = A.Scale(size=(128, 171))

    transform = transforms.Compose(transform_list)

    return transform


def get_data(transform, mode, args):
    args.logger.info('Loading data for "%s" mode' % mode)
    if args.dataset == 'ucf101':
        dataset = UCF101LMDB(mode=mode, transform=transform,
                             num_frames=args.seq_len * args.num_seq, ds=args.ds, which_split=args.which_split,
                             ft_mode=args.ft_mode,
                             return_label=True)
    elif args.dataset == 'ucf101-10clip':
        dataset = UCF101_10CLIP(mode=mode, transform=transform,
                                num_frames=args.seq_len, ds=args.ds, which_split=args.which_split,
                                return_label=True)
    elif args.dataset == 'hmdb51':
        dataset = HMDB51LMDB(mode=mode, transform=transform,
                             num_frames=args.seq_len * args.num_seq, ds=args.ds, which_split=args.which_split,
                             ft_mode=args.ft_mode,
                             return_label=True)
    elif args.dataset == 'hmdb51-10clip':
        dataset = HMDB51_10CLIP(mode=mode, transform=transform,
                                num_frames=args.seq_len, ds=args.ds, which_split=args.which_split,
                                return_label=True)
    else:
        raise NotImplementedError
    return dataset


def get_dataloader(dataset, mode, args):
    args.logger.info("Creating data loaders")
    train_sampler = data.distributed.DistributedSampler(dataset, shuffle=True) if args.distributed else data.RandomSampler(dataset)
    val_sampler = None

    if mode == 'train':
        data_loader = FastDataLoader(
            dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            worker_init_fn=worker_init_fn,
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    elif mode == 'val':
        data_loader = FastDataLoader(
            dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
            worker_init_fn=worker_init_fn,
            num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True)

    elif mode == 'test':
        data_loader = FastDataLoader(
            dataset, batch_size=1, shuffle=True,
            worker_init_fn=worker_init_fn,
            num_workers=args.workers, pin_memory=True)
    args.logger.info('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader


def set_path(args):
    if args.resume:
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    elif args.test:
        exp_path = os.path.dirname(os.path.dirname(args.test))
    else:
        exp_path = 'log/{args.prefix}/ft/{args.name_prefix}'.format(
            args=args)
    if 'ucf' in args.dataset:
        dataset_fold = 'ucf'
    elif 'hmdb' in args.dataset:
        dataset_fold = 'hmdb'

    img_path = os.path.join(exp_path, dataset_fold, 'img')
    model_path = os.path.join(exp_path, dataset_fold, 'model')
    if not args.test:
        log_file_name = 'log'
    elif args.retrieval:
        log_file_name = 'test_retrieval_log'
    elif not args.temporal_ten_clip:
        log_file_name = 'test_log'
    else:
        log_file_name = 'temporal_10_test_log'
    log_file = os.path.join(exp_path, dataset_fold, log_file_name)

    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    return img_path, model_path, exp_path, log_file


if __name__ == '__main__':
    args = parse_args()
    main(args)
