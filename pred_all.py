
import math  # 数学运算库
import os  # 操作系统接口
import torch  # PyTorch深度学习框架
import torch.optim  # PyTorch优化器
from torch import nn  # PyTorch神经网络模块
from models import baseNet  # 导入自定义的基础网络模型
from data_loader import build_data_loader, trans_tif  # 导入数据加载和TIF转换函数
import numpy as np  # 数值计算库
from util.util import prepare_training  # 导入训练准备函数
import torch.nn.functional as F  # PyTorch函数式接口
import pandas as pd  # 数据分析库
from tabulate import tabulate  # 表格格式化输出
import torch.backends.cudnn as cudnn  # CUDA深度神经网络库
import torch.backends.cuda  # CUDA后端
import argparse  # 命令行参数解析
# 启用TensorFloat32精度加速
torch.backends.cudnn.allow_tf32 = True  # 允许cudnn使用TF32
torch.backends.cuda.matmul.allow_tf32 = True  # 允许矩阵乘法使用TF32


def args_parser():
    """
    解析命令行参数的函数
    返回: 解析后的参数对象
    """
    project_name = 'own'  
    parser = argparse.ArgumentParser()  
  
    parser.add_argument('-results', type=str, default='./results/') 
    parser.add_argument('-checkpoints', type=str, default='./checkpoints/')  
    parser.add_argument('-project_name', type=str, default=project_name)  
    parser.add_argument('-dataset', type=str, default='Houston', 
                        choices=['PaviaU', 'Houston','Indian','Salinas','WHU_Hi_LongKou'])
    parser.add_argument('--epochs', type=int, default=200,  
                        help='end epoch for training')
                
  
    parser.add_argument('--hidden_size', type=int, default=512)  

   
    parser.add_argument('--batch_size', type=int, default=32)  
    # parser.add_argument('--train_ratio', type=list, default=[3,4,5,6,7,8,2,1,3,4])
    parser.add_argument('--train_ratio', type=int, default=7,  
                        help='samples for training')
    parser.add_argument('--is_train', type=bool, default=False,  
                        help='train or test')
    parser.add_argument('--is_outimg', type=bool, default=False,  
                        help='output all image or not')
    parser.add_argument('--checkpointsmodelfile', type=str, 
                       default='./checkpoints/own/Houston/model_99.35.pth')  # 修正路径格式
    parser.add_argument('--seed', type=int, default=300,  
                        help='random seed') # 5,7 300 ; 10 200
    parser.add_argument('--PCA', type=int, default=None, help='PCA')  # 
    parser.add_argument('--allimg', type=bool, default=False, help='allimg')  # 

    args = parser.parse_args()  # 
    return args

def custom_repr(self):
    """
    自定义张量的表示方法
    新格式：在原有输出前添加形状信息 {Tensor:(shape)}
    """
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

#   
original_repr = torch.Tensor.__repr__
#   
torch.Tensor.__repr__ = custom_repr


def pred_allimg(model, device, X, y, epoch, args):
    """
    预测整个图像的函数
    参数:
        model: 训练好的模型
        device: 计算设备(CPU/GPU)
        X: 输入图像数据
        y: 标签数据
        epoch: 训练轮次
        args: 参数
    """
    model.eval()  
    height = y.shape[0]   
    width = y.shape[1]   
    with torch.no_grad():  
        outputs = np.zeros((height, width))   
        for i in range(height):   
            for j in range(width):   
                if args.allimg:   
                    
                    image_patch = X[i:i + args.patch_size, j:j + args.patch_size, :]
                    
                    image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2],
                                                    1)
                    
                    X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2)).to(device)
                    
                    prediction = model(X_test_image)
                
                    prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
                    
                    outputs[i][j] = prediction + 1
                else:  
                    if int(y[i, j]) == 0:  
                        continue
                    
                    image_patch = X[i:i + args.patch_size, j:j + args.patch_size, :]
                    
                    image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2],
                                                    1)
                    
                    X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2)).to(device)
                    
                    prediction = model(X_test_image)
                    
                    prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
                    
                    outputs[i][j] = prediction + 1
            
            if i % 20 == 0:
                print('... ... row ', i, ' handling ... ...')
    
    
    if args.allimg:   
        
        finalmodelfile = args.results + args.project_name+ '/' + args.dataset+ '/All_PRED.tif'
        
        trans_tif(outputs, finalmodelfile)
    else: 
        
        finalmodelfile = args.results + args.project_name+ '/' + args.dataset+ '/Label_PRED.npy'
        
        y[y!=0] = 1
        
        outputs = outputs*y
        
        trans_tif(outputs, finalmodelfile)
    
def main():
    """
    主函数，程序的入口点
    """
    args = args_parser()  
    print(args)  
    
    
    model_dir_path = os.path.join(args.results, args.project_name + '/')
    log_file = os.path.join(args.results, args.project_name +'/log.txt')

    
    os.makedirs(model_dir_path, exist_ok=True)  
    os.makedirs(args.checkpoints+args.project_name+'/', exist_ok=True)  
    args.log_file = log_file  

    
    X, y = build_data_loader(args)  

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    if args.PCA is None:  
        model = baseNet(args.hsi_bands, args.num_class).to(device)  
    else:   
        model = baseNet(args.PCA, args.num_class).to(device)  
    
    
    model.load_state_dict(torch.load(args.checkpointsmodelfile, weights_only=True))
    
    pred_allimg(model, device, X, y, args.epochs, args)

def main_test():
    """
    测试函数，用于模型测试（当前为空）
    """
    pass



if __name__ == '__main__':
     main()
