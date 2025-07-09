import math
import os
import torch
import torch.optim
from torch import nn
import sys
sys.path.append('.')
from data_loader import build_data_loader, trans_tif
import numpy as np
from util.util import prepare_training
import torch.nn.functional as F
import pandas as pd
from tabulate import tabulate
import torch.backends.cudnn as cudnn
import torch.backends.cuda
import argparse
from compare.lite_hcnnet.Lite_HCNet import Lite_HCNet
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

def args_parser():
    project_name = 'lite_hcnnet'
    parser = argparse.ArgumentParser()
    parser.add_argument('-results', type=str, default='./results/')
    parser.add_argument('-checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('-project_name', type=str, default=project_name)
    parser.add_argument('-dataset', type=str, default='PaviaU',
                        choices=['PaviaU', 'Houston'])

    # learning setting
    parser.add_argument('--epochs', type=int, default=20,
                        help='end epoch for training')
    # parser.add_argument('--lr', type=float, default=2e-4,
    #                     help='learning rate')
    parser.add_argument('--lr_scheduler', default='multisteplr', type=str) #cosinewarm poly
    parser.add_argument('--lr_start', default=1e-2, type=int)
    parser.add_argument('--lr_decay', default=0.99, type=float)
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                    help='weight decay (default: 0.001)')
    parser.add_argument('--lr_min', default=2e-6, type=int)
    parser.add_argument('--T_0', default=20, type=int)
    parser.add_argument('--T_mult', default=2, type=int)
    parser.add_argument('--optim', default='adam', type=str)
    parser.add_argument('--milestones', default=[40], type=list)
    # SGD
    parser.add_argument('--momentum', default=0.98, type=float)
    # Adam & AdamW
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--num', default=0, type=int)

    # dataset setting
    parser.add_argument('--batch_size', type=int, default=16)
    # parser.add_argument('--train_ratio', type=list, default=[3,4,5,6,7,8,2,1,3,4])
    parser.add_argument('--train_ratio', type=float, default=0.01,
                        help='samples for training')
    # parser.add_argument('--train_ratio', type=float, default=0.8,
    #                     help='samples for training')
    parser.add_argument('--is_train', type=bool, default=True,
                        help='train or test')
    parser.add_argument('--is_outimg', type=bool, default=False,
                        help='output all image or not')
    parser.add_argument('--checkpointsmodelfile', type=str, default='./checkpoints/own/own.pth')
    parser.add_argument('--seed', type=int, default=400,
                        help='random seed') # 5,7 300 ; 10 200
    parser.add_argument('--PCA', type=int, default=None, help='PCA')

    args = parser.parse_args()
    return args

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

def calc_loss(outputs, labels):
    criterion = nn.CrossEntropyLoss()
    loss =  criterion(outputs, labels)
    return loss

def train(model, device, train_loader, optimizer, epoch, args):
    model.train()
    total_loss = 0
    
    # 添加时间统计
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    
    for i, (inputs_1, labels) in enumerate(train_loader):
        inputs_1 = inputs_1.to(device)
        labels = labels.to(device)
        if args.PCA is not None:
            inputs_1 = inputs_1.view(-1, args.PCA, args.patch_size, args.patch_size)
        else:
            inputs_1 = inputs_1.view(-1, args.hsi_bands, args.patch_size, args.patch_size)

        optimizer.zero_grad()
        outputs = model(inputs_1)
        loss = calc_loss(outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 计算时间
    end_time.record()
    torch.cuda.synchronize()
    epoch_time = start_time.elapsed_time(end_time) / 1000  # 转换为秒

    print(' epoch %d' % (epoch))
    print(' [loss avg: %.4f]' % (total_loss/(len(train_loader))))
    print(' [current loss: %.4f]' % (loss.item()))
    print(' [epoch time: %.2f s]' % (epoch_time))
    
    content = ' epoch %d' % (epoch) + \
             ' [loss avg: %.4f]' % (total_loss/(len(train_loader))) + \
             ' [current loss: %.4f]' % (loss.item()) + \
             ' [epoch time: %.2f s]' % (epoch_time)
    with open(args.log_file, 'a') as appender:
        appender.write(content + '\n')

def main():
    args = args_parser()
    print(args)
    
    # 添加总训练时间计时
    total_start_time = torch.cuda.Event(enable_timing=True)
    total_end_time = torch.cuda.Event(enable_timing=True)
    total_start_time.record()
    
    model_dir_path = os.path.join(args.results, args.project_name + '/', args.dataset+'/')
    log_file = os.path.join(args.results, args.project_name+'/', args.dataset+'/log.txt')

    os.makedirs(model_dir_path, exist_ok=True)
    os.makedirs(args.checkpoints+args.project_name+'/'+args.dataset+'/', exist_ok=True)
    args.log_file = log_file
    
    train_loader, val_loader = build_data_loader(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.PCA is None:
        model = Lite_HCNet(args.hsi_bands, args.num_class, patch_size=args.patch_size).to(device)
    else:    
        model = Lite_HCNet(args.PCA, args.num_class, patch_size=args.patch_size).to(device)

    optimizer, lr_scheduler = prepare_training(args, model)
            
    best_acc = 0
    for epoch in range(args.epochs):
        train(model, device, train_loader, optimizer, epoch, args)
        lr_scheduler.step()
        if (epoch+1)%2 == 0:
            acc = val(model, device, val_loader, epoch, args)
            if acc >= best_acc:
                  best_acc = acc
                  print("save model")
                  checkpointsmodelfile = os.path.join(args.checkpoints, args.project_name, args.dataset, 'model_%.2f.pth' % best_acc)
                  torch.save(model.state_dict(), checkpointsmodelfile)
    
    # 计算总训练时间
    total_end_time.record()
    torch.cuda.synchronize()
    total_time = total_start_time.elapsed_time(total_end_time) / 1000  # 转换为秒
    print('\n总训练时间: %.2f 秒 (%.2f 小时)' % (total_time, total_time/3600))
    with open(args.log_file, 'a') as appender:
        appender.write('\n总训练时间: %.2f 秒 (%.2f 小时)\n' % (total_time, total_time/3600))

def main_test():
    pass

if __name__ == '__main__':
     main()
