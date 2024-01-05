'''
iterative pruning for supervised task
with lottery tickets or pretrain tickets
support datasets: cifar10, Fashionmnist, cifar100, svhn
'''

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data

#import torchvision.models as models
from models.resnet import resnet18, MaskedConv2d
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import *
from gradient_unroll import train_with_imagenet_unroll
from utils import test_with_imagenet
from pruning_utils import extract_mask, prune_model_custom
import copy
import torch.nn.utils.prune as prune

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
parser.add_argument('--sign-lr', type=float, default=1e-5)
parser.add_argument('--lower-lr', type=float, default=1e-2)
parser.add_argument('--lamb', type=float, default=1)

parser.add_argument('--ten-shot', action="store_true")
parser.add_argument('--lower_steps', type=int, default=1)
parser.add_argument('--no-alpha', action="store_true")
parser.add_argument('--no-beta', action="store_true")
parser.add_argument('--imagenet-pretrained', action="store_true")

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

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    args.distributed = True
    args.multiprocessing_distributed=True

    main_worker(args.gpu, 1, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    '''
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    '''

    model = resnet18(pretrained=args.imagenet_pretrained, num_classes=1000, imagenet=True)
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
    # process_group = torch.distributed.new_group(list(range(args.world_size)))
    model_lower = copy.deepcopy(model)
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)
    # model_lower = nn.SyncBatchNorm.convert_sync_batchnorm(model_lower, process_group)

    # init pretrianed weight
    for m in model.modules():
        if isinstance(m, MaskedConv2d):
            m.set_incremental_weights()
    for m in model_lower.modules():
        if isinstance(m, MaskedConv2d):
            m.set_incremental_weights(beta=False)
    print('dataparallel mode')
    

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        model_lower.cuda(args.gpu)

        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.DataParallel(model)#, find_unused_parameters=True)
        model_lower = torch.nn.DataParallel(model_lower)#, find_unused_parameters=True)

    else:
        model.cuda()
        model_lower.cuda()
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        model = torch.nn.DataParallel(model)#, find_unused_parameters=True)
        model_lower = torch.nn.DataParallel(model_lower)#, find_unused_parameters=True)
        
    # Data loading code
    initialization = copy.deepcopy(model.module.state_dict())
    cudnn.benchmark = True
    from cub import cub200, cub200_10
    train_transform_list = [
        transforms.RandomResizedCrop(448),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ]
    test_transforms_list = [
            transforms.Resize(int(448/0.875)),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))

    ]
    if not args.ten_shot:
        train_dataset = cub200(args.data, True, transforms.Compose(train_transform_list))
        val_dataset = cub200(args.data, False, transforms.Compose(test_transforms_list))
    else:
        train_dataset = cub200_10(args.data, True, transforms.Compose(train_transform_list))
        val_dataset = cub200_10(args.data, False, transforms.Compose(test_transforms_list))


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    
    #imagenet_traindir = os.path.join(args.imagenet_data, 'imagenet-c.x-full','gaussian_noise','3')
    imagenet_traindir = args.imagenet_train_data
    imagenet_valdir = args.imagenet_val_data  
    imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    imagenet_train_dataset = datasets.ImageFolder(
        imagenet_traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            imagenet_normalize,
        ]))

    imagenet_train_loader = torch.utils.data.DataLoader(
        imagenet_train_dataset, batch_size=args.imagenet_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # imagenet_train_loader = None
    criterion = nn.CrossEntropyLoss()
    alpha_params = {}
    beta_params = {}


    optimizer = torch.optim.SGD([
                {'params': [p for name, p in model.named_parameters() if 'mask' not in name], "lr": args.lr}, # params 
                {'params': [p for name, p in model.named_parameters() if 'mask' in name], "lr": args.reg_lr, 'weight_decay': 0} # masks
            ], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            
    for m in model.modules():
        if isinstance(m, MaskedConv2d):
            pass

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            start_epoch=args.start_epoch
            all_result=checkpoint['result']
            best_sa = checkpoint['best_sa']
            start_state = checkpoint['state']
            if start_state:
                current_mask = extract_mask(checkpoint['state_dict'])
                prune_model_custom(model.module, current_mask, conv1=True)
                args.epochs = 45

            model.load_state_dict(checkpoint['state_dict'])

            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
            optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (state {}, epoch {})"
                .format(args.resume, checkpoint['state'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    else:
        all_result = {}
        all_result['train'] = []
        all_result['test_ta'] = []
        all_result['ta'] = []

        start_epoch = 0
        start_state = 0
        best_sa = 0
    print('######################################## Start Standard Training Iterative Pruning ########################################')
    for epoch in range(start_epoch, args.epochs):

        print(optimizer.state_dict()['param_groups'][0]['lr'])

        acc = train_with_imagenet_unroll(train_loader, imagenet_train_loader, model, model_lower, criterion, optimizer, epoch, args)
        
        scheduler.step()
        # evaluate on validation set
        tacc, all_outputs, all_targets = test_with_imagenet(val_loader, model, criterion, args, alpha_params, beta_params)
        # evaluate on test set
        all_result['train'].append(acc)
        all_result['ta'].append(tacc)

        # if epoch % 5 == 0:
        #     torch.save(model.state_dict(), os.path.join(args.save_dir, f"model_{epoch}.pth.tar"))

        # remember best prec@1 and save checkpoint
        is_best_sa = tacc  > best_sa
        best_sa = max(tacc, best_sa)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'result': all_result,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_sa': best_sa,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'alpha': alpha_params,
                'beta': beta_params,
                'outputs': all_outputs,
                'targets': all_targets,
            }, is_SA_best=is_best_sa, pruning=0, save_path=args.save_dir)

        plt.plot(all_result['train'], label='train_acc')
        plt.plot(all_result['ta'], label='val_acc')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'net_train.png'))
        plt.close()

    #report result
    print('* best SA={}'.format(all_result['ta'][np.argmax(np.array(all_result['ta']))]))


if __name__ == '__main__':
    main()