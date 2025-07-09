import math
import os
import torch
import torch.optim
from torch import nn
import sys
sys.path.append('.')
from compare.HybirdSN.HybirdSN import HybridSN
from data_loader import build_data_loader_f3d, trans_tif
import numpy as np
from util.util import prepare_training
import torch.nn.functional as F
import pandas as pd
from tabulate import tabulate
import torch.backends.cudnn as cudnn
import torch.backends.cuda
import argparse
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


def args_parser():
    project_name = 'hybirdsn'
    parser = argparse.ArgumentParser()
    parser.add_argument('-results', type=str, default='./results/')
    parser.add_argument('-checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('-project_name', type=str, default=project_name)
    parser.add_argument('-dataset', type=str, default='Houston', 
                        choices=['PaviaU', 'Houston','Indian','Salinas','WHU_Hi_LongKou'])

    # dataset setting
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_ratio', type=int, default=7,
                        help='samples for training')
    parser.add_argument('--is_train', type=bool, default=False,
                        help='train or test')
    parser.add_argument('--is_outimg', type=bool, default=False,
                        help='output all image or not')
    parser.add_argument('--checkpointsmodelfile', type=str, default='./checkpoints/hybirdsn/Houston/model_98.43.pth')
    parser.add_argument('--seed', type=int, default=345,
                        help='random seed')
    parser.add_argument('--PCA', type=int, default=None, help='PCA')
    parser.add_argument('--allimg', type=bool, default=False, help='allimg')

    args = parser.parse_args()
    return args

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr


def pred_allimg(model, device, X, y, args):
    model.eval()
    height = y.shape[0]
    width = y.shape[1]
    with torch.no_grad():
        outputs = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                if args.allimg:
                    image_patch = X[i:i + args.patch_size, j:j + args.patch_size, :]
                    # 使用与根目录pred_all.py相同的处理方式
                    image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2], 1)
                    X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2)).to(device)
                    prediction = model(X_test_image)
                    prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
                    outputs[i][j] = prediction + 1
                else:
                    if int(y[i, j]) == 0:
                        continue
                    image_patch = X[i:i + args.patch_size, j:j + args.patch_size, :]
                    # 使用与根目录pred_all.py相同的处理方式
                    image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2], 1)
                    X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2)).to(device)
                    prediction = model(X_test_image)
                    prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
                    outputs[i][j] = prediction + 1
            if i % 20 == 0:
                print('... ... row ', i, ' handling ... ...')
    
    if args.allimg:
        finalmodelfile = args.results + args.project_name+ '/' + args.dataset+ '/ALL_PRED.tif'
        trans_tif(outputs, finalmodelfile)
    else:
        finalmodelfile = args.results + args.project_name+ '/' + args.dataset+ '/Label_PRED.npy'
        y[y!=0] = 1
        outputs = outputs*y
        trans_tif(outputs, finalmodelfile)
    

def main():
    args = args_parser()
    print(args)
    
    # 检查权重文件路径
    if not os.path.exists(args.checkpointsmodelfile):
        print(f"错误：模型权重文件不存在: {args.checkpointsmodelfile}")
        print("请确保：")
        print("1. 模型已经训练完成")
        print("2. 权重文件路径正确")
        print(f"3. 检查目录 {os.path.dirname(args.checkpointsmodelfile)} 是否存在")
        return
    
    model_dir_path = os.path.join(args.results, args.project_name + '/')
    log_file = os.path.join(args.results, args.project_name +'/log.txt')

    os.makedirs(model_dir_path, exist_ok=True)
    os.makedirs(args.checkpoints+args.project_name+'/', exist_ok=True)
    args.log_file = log_file

    X, y = build_data_loader_f3d(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = HybridSN(class_num=args.num_class).to(device)
    
   
        # 加载checkpoint
    checkpoint = torch.load(args.checkpointsmodelfile, weights_only=True)
    model.load_state_dict(checkpoint)

    

    pred_allimg(model, device, X, y, args)

if __name__ == '__main__':
     main()