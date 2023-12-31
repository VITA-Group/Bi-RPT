'''
iterative pruning for supervised task
with lottery tickets or pretrain tickets
support datasets: cifar10, Fashionmnist, cifar100, svhn
'''

import os
import pdb
from sched import scheduler
import time
import pickle
import random
import shutil
import argparse
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
#import torchvision.models as models
from models.resnet import resnet18, MaskedConv2d
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import *
from utils import train_with_imagenet_gdp, test_with_imagenet
from pruning_utils import check_sparsity,extract_mask,prune_model_custom
import copy
import torch.nn.utils.prune as prune


def pruning_model(model, px=0.2):

    print('start unstructured pruning for all conv layers')
    parameters_to_prune =[]
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m,'weight'))



    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

parser = argparse.ArgumentParser(description='PyTorch Iterative Pruning')

##################################### data setting #################################################
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--imagenet_train_data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--imagenet_val_data', type=str, default='../data', help='location of the data corpus')

parser.add_argument('--dataset', type=str, default='cifar10', help='dataset[cifar10&100, svhn, fmnist')

##################################### model setting #################################################

##################################### basic setting #################################################
parser.add_argument('--gpu', type=int, default=None, help='gpu device id')
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--random', action="store_true", help="using random-init model")
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')
parser.add_argument('--resume', type=str, default=None, help='checkpoint file')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default=None, type=str)

##################################### training setting #################################################
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--imagenet_batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=95, type=int, help='number of total epochs to run')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

##################################### Pruning setting #################################################
parser.add_argument('--pruning_times', default=19, type=int, help='overall times of pruning')
parser.add_argument('--rate', default=0.2, type=float, help='pruning rate')
parser.add_argument('--prune_type', default='lt', type=str, help='IMP type (lt, pt, rewind_lt or pt_trans)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:35505', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument("--warmup", default=0, type=int)

parser.add_argument("--alpha-init", default=5, type=int)
parser.add_argument("--sparsity-pen", default=1e-9, type=float)
parser.add_argument('--l1-reg-beta', type=float, default=1e-6)
parser.add_argument('--reg-lr', type=float, default=10)
parser.add_argument('--lamb', type=float, default=1)


def main():
    best_sa = 0
    args = parser.parse_args()
    print(args)

    print('*'*50)
    print('Dataset: {}'.format(args.dataset))
    print('*'*50)
    print('Pruning type: {}'.format(args.prune_type))

    #torch.cuda.set_device(int(args.gpu))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    #args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    args.distributed = True
    args.multiprocessing_distributed=True

    ngpus_per_node = torch.cuda.device_count()
    if True:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    model = resnet18(pretrained=True, num_classes=1000, imagenet=True)
    if args.checkpoint and not args.resume:
        print(f"LOAD CHECKPOINT {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint,map_location="cpu")
        state_dict = checkpoint['state_dict']
        load_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                load_state_dict[key[7:]] = state_dict[key]
            else:
                load_state_dict[key] = state_dict[key]
        model.load_state_dict(load_state_dict)

    model.new_fc = nn.Linear(512, 200)
    from torch.nn import init
    init.kaiming_normal_(model.new_fc.weight.data)
    process_group = torch.distributed.new_group(list(range(args.world_size)))
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)

    # init pretrianed weight
    ticket_init_weight = deepcopy(model.state_dict())
    for m in model.modules():
        if isinstance(m, MaskedConv2d):
            m.set_incremental_weights()
            #m.set_lower()
    print('dataparallel mode')
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])#, find_unused_parameters=True)
    else:
        model.cuda()
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        model = torch.nn.parallel.DistributedDataParallel(model)#, find_unused_parameters=True)

    cudnn.benchmark = True
    
    checkpoint = torch.load("/usr/workspace/olivare/cub_unroll_3/0checkpoint.pth.tar", map_location=f"cpu")
    epoch = checkpoint['epoch']
    print(epoch)
    model.load_state_dict(checkpoint['state_dict'])

    for m in model.modules():
        if isinstance(m, MaskedConv2d):
            m.epsilon = 0.1 * (0.9) ** epoch
    print('######################################## Start Standard Training Iterative Pruning ########################################')
        # evaluate on validation set
    for name, m in model.module.named_modules():
        if isinstance(m, MaskedConv2d):
            m.set_upper()
            print(name)
            print(((m.mask_beta ** 2) / ((m.mask_beta ** 2) + m.epsilon)).mean())
            print(((m.mask_alpha ** 2) / ((m.mask_alpha ** 2) + m.epsilon)).mean())
            print(((m.mask_alpha ** 2) / ((m.mask_alpha ** 2) + m.epsilon) * (m.mask_beta ** 2) / ((m.mask_beta ** 2) + m.epsilon)).mean())
            

if __name__ == '__main__':
    main()