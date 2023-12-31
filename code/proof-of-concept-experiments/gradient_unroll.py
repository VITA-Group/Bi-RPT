import os
import time
import copy
import torch
import random
import shutil
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from dataset import *
from models.resnet import resnet18, resnet50, resnet152
from pruning_utils import *
from models.resnet import MaskedConv2d

__all__ = ['setup_model_dataset', 'setup_seed',
            'train', 'test',
            'save_checkpoint', 'load_weight_pt_trans', 'load_ticket']

def train_with_imagenet_unroll(train_loader, imagenet_train_loader, model, model_lower, criterion, optimizer, epoch, args):

    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    start = time.time()
    # imagenet_train_loader_iter = iter(imagenet_train_loader)
    for name, m in model.named_modules():
        if isinstance(m, MaskedConv2d):
            m.epsilon *= 0.9
            
    for i, (image, target) in enumerate(train_loader):

        if epoch < args.warmup:
            warmup_lr(epoch, i+1, optimizer, one_epoch_step=len(train_loader), args=args)

        image = image.cuda()
        target = target.cuda()
        if i % 10 == 0:
            for name, m in model.named_modules():
                if isinstance(m, MaskedConv2d):
                    m.set_lower()
            # decrease lr
            previous_lr = optimizer.param_groups[0]['lr']
            current_lr = 1e-3 # previous_lr / 10
            optimizer.param_groups[0]['lr'] = current_lr
            state_dict = model.state_dict()

            for key in list(state_dict.keys()):
                if 'mask_beta' in key: del state_dict[key]
            model_lower.load_state_dict(state_dict)
            weights = []
            alphas = []
            for _ in range(args.lower_steps):            
                try:
                    imagenet_image, imagenet_target = next(imagenet_train_loader_iter)
                except:
                    imagenet_train_loader_iter = iter(imagenet_train_loader)
                    imagenet_image, imagenet_target = next(imagenet_train_loader_iter)
                # compute output
                imagenet_image = imagenet_image.cuda()
                imagenet_target = imagenet_target.cuda()
                
                
                output_old, output_new = model(imagenet_image)
                loss = criterion(output_old, imagenet_target) + 0 * output_new.sum()
                loss.backward()
                
                for name, m in model.named_modules():
                    if isinstance(m, MaskedConv2d):
                        m.mask_alpha.grad = None
                        m.mask_beta.grad = None
                if _ > 0:
                    for name, m in model.named_modules():
                        if isinstance(m, MaskedConv2d):
                            m.weight.grad.data = torch.sign(m.weight.grad.data) * 1e-4
                
                optimizer.step()
                optimizer.zero_grad()
                if _ == 0:
                    output_old, output_new = model_lower(imagenet_image)
                    loss_lower = criterion(output_old, imagenet_target) + 0 * output_new.sum()
                    for name, m in model_lower.named_modules():
                        if isinstance(m, MaskedConv2d):
                            weights.append(m.weight)
                            alphas.append(m.mask_alpha)
                    grad_w = torch.autograd.grad(loss_lower, weights, create_graph=True, retain_graph=True)
            # restore lr 
            optimizer.param_groups[0]['lr'] = previous_lr

            for name, m in model.named_modules():
                if isinstance(m, MaskedConv2d):
                    m.set_upper()

        output_old, output_new = model(image)
        loss = criterion(output_new, target) + 0 * output_old.sum()
        optimizer.zero_grad()
        loss.backward()
        # remove weights grad
        grad_new_w = []
        for name, m in model.named_modules():
            if isinstance(m, MaskedConv2d):
                grad_new_w.append(m.weight.grad.data.clone())
        # start unrolling
        aux_loss = 0
        for go, gn in zip(grad_w, grad_new_w):
            aux_loss = aux_loss + torch.sum(go * gn)
        
        grads = torch.autograd.grad(aux_loss, alphas, retain_graph=True)
        idx = 0
        # alpha_lr = 1e-3
        if not args.no_alpha:
            for m in model.modules():
                if isinstance(m, MaskedConv2d):
                    # print(grads[idx].abs().mean())
                    # print(m.mask_alpha.grad.abs().mean())
                    # print("----------")
                    m.mask_alpha.grad.data.sub_(grads[idx] * 0.1)
                    idx += 1
        else:
            for m in model.modules():
                if isinstance(m, MaskedConv2d):
                    m.mask_alpha.grad = None
        
        if args.no_beta:
            for m in model.modules():
                if isinstance(m, MaskedConv2d):
                    m.mask_beta.grad = None

        optimizer.step()
        model.zero_grad()
        
        # end unrolling

        loss = loss.float()
        for name, m in model.named_modules():
           if isinstance(m, MaskedConv2d):
                if not args.no_beta:
                    beta = m.mask_beta.data.detach().clone()
                    lr = optimizer.param_groups[1]['lr']
                    # print(lr * args.lamb)
                    #print(beta.data.abs().mean())
                    
                    m1 = beta >= lr * args.lamb
                    m2 = beta <= -lr * args.lamb
                    m3 = (beta.abs() < lr * args.lamb)
                    m.mask_beta.data[m1] = m.mask_beta.data[m1] - lr * args.lamb
                    m.mask_beta.data[m2] = m.mask_beta.data[m2] + lr * args.lamb
                    m.mask_beta.data[m3] = 0
                if not args.no_alpha:
                    alpha = m.mask_alpha.data.detach().clone()
                    lr = optimizer.param_groups[1]['lr']
                    # print(lr * args.lamb)
                    #print(alpha.data.abs().mean())
                    m1 = alpha >= lr * args.lamb
                    m2 = alpha <= -lr * args.lamb
                    m3 = (alpha.abs() < lr * args.lamb)
                    m.mask_alpha.data[m1] = m.mask_alpha.data[m1] - lr * args.lamb
                    m.mask_alpha.data[m2] = m.mask_alpha.data[m2] + lr * args.lamb
                    m.mask_alpha.data[m3] = 0
        # measure accuracy and record loss
        prec1 = accuracy(output_new.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    if args.rank == 0:
        for name, p in model.named_parameters():
            if 'mask_alpha' in name or 'mask_beta' in name:
                print(name, (p.data.abs()).mean())
                

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def warmup_lr(epoch, step, optimizer, one_epoch_step, args):

    overall_steps = args.warmup*one_epoch_step
    current_steps = epoch*one_epoch_step + step

    lr = args.lr * current_steps/overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p['lr']=lr
