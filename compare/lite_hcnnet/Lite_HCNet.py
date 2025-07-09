import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import spectral
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from operator import truediv
import warnings
import time
from torchsummary import summary
import math
warnings.filterwarnings("ignore")
from scipy.interpolate import make_interp_spline
import os
import matplotlib as mpl

# 模型代码
# -------------------------------------------------------------------------------------------- #
def bsm(n,d):
    a = [[0]*n for x in range(n)]
    p = 0
    q = n-1
    w = (n+1)/2
    w =int(w)
    #print(w)
    #w1 = 1 / w
    #print(w1)
    t = 0
    while p < d:
        for i in range(p,q):
            a[p][i] = t


        for i in range(p,q):
            a[i][q] = t


        for i in range(q,p,-1):
            a[q][i] = t


        for i in range(q,p,-1):
            a[i][p] = t

        p += 1
        q -= 1
        #t += w1

    while p==d or p>d and p<q:
        for i in range(p,q):
            a[p][i] = 1


        for i in range(p,q):
            a[i][q] = 1


        for i in range(q,p,-1):
            a[q][i] = 1


        for i in range(q,p,-1):
            a[i][p] = 1

        a[w-1][w-1] = 1
        p += 1
        q -= 1
    return np.array(a)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ScaleMaskModule(nn.Module):#尺度掩膜
    def __init__(self,d):
        # w是空间尺度，n是光谱维度，p是批次大小
        super(ScaleMaskModule, self).__init__()

        self.d = d
    def forward(self,x):
        w = x.shape[3]
        n = x.shape[2]
        o = x.shape[1]
        p = x.shape[0]
       # print(x.shape)
        out = bsm(w,self.d)
        #print(out.shape)
        out = torch.from_numpy(out)
        out = out.repeat(p, o, 1, 1)#out.repeat(p, o,n, 1, 1)
        #print(out.shape)
        out = out.type(torch.FloatTensor)
        out = out.to(device)
        #print(x * out)
        return x * out

class NCAM3D(nn.Module):# 3D NCAM
    def __init__(self, c, patch_size):
        super(NCAM3D, self).__init__()
        gamma = 2
        b = 3
        kernel_size_21 = int(abs((math.log(c, 2) + b) / gamma))
        kernel_size_21 = kernel_size_21 if kernel_size_21 % 2 else kernel_size_21 + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.ScaleMaskModule = ScaleMaskModule((patch_size-1)//2-1)

        self.conv1d = nn.Conv2d(1, 1, kernel_size=(2,kernel_size_21), padding=(0,(kernel_size_21 - 1) // 2), dilation=1)
        self.conv1d1 = nn.Conv2d(1, 1, kernel_size=(2, kernel_size_21), padding=(0, (kernel_size_21 - 1) // 2),
                               dilation=1)

    def forward(self, x):

        out =x
        #### 通道注意力
        out_1 = out.shape[1]
        out_2 = out.shape[2]
        out = out.reshape(out.shape[0], -1, out.shape[3], out.shape[4])
        ###中心像素的光谱
        out_x = self.ScaleMaskModule(out)
        out_x1 = self.avg_pool(out_x)
        out_x1 = out_x1.reshape(out_x1.shape[0], -1)
        out_x2 = reversed(out_x1.permute(1, 0)).permute(1, 0)  # 原地翻转，倒序:[1,2,3]->[3,2,1]

        out_x1 = out_x1.reshape(out_x1.shape[0], 1, 1, out_x1.shape[1])
        out_x2 = out_x2.reshape(out_x2.shape[0], 1, 1, out_x2.shape[1])

        out_xx = torch.cat([out_x1, out_x2], dim=2)
        #######
        ###全局空间的光谱
        out1 = self.avg_pool(out)
        out1 = out1.reshape(out1.shape[0], -1)

        out2 = reversed(out1.permute(1, 0)).permute(1, 0)  # 原地翻转，倒序:[1,2,3]->[3,2,1]

        out1 = out1.reshape(out1.shape[0], 1, 1, out1.shape[1])
        out2 = out2.reshape(out2.shape[0], 1, 1, out2.shape[1])

        outx = torch.cat([out1, out2], dim=2)
        #########
        at1 = F.sigmoid(self.conv1d(outx)).permute(0, 3, 1, 2) * F.sigmoid(self.conv1d1(out_xx)).permute(0, 3, 1, 2)
        at = F.sigmoid((at1-0.2)*2)
        out = out * at
        #####
        out = out.reshape(out.shape[0], out_1, out_2, out.shape[2], out.shape[3])

        return out


class NCAM2D(nn.Module):  # 2D NCAM
    def __init__(self, c, patch_size):
        super(NCAM2D, self).__init__()

        gamma = 2
        b = 3
        kernel_size_21 = int(abs((math.log(c , 2) + b) / gamma))
        kernel_size_21 = kernel_size_21 if kernel_size_21 % 2 else kernel_size_21 + 1#保证结果为奇数，当%2不为0时执行if语句，即奇数；

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.ScaleMaskModule = ScaleMaskModule((patch_size - 1) // 2-1)

        self.conv1d = nn.Conv2d(1, 1, kernel_size=(2, kernel_size_21), padding=(0, (kernel_size_21 - 1) // 2),
                                dilation=1)
        self.conv1d1 = nn.Conv2d(1, 1, kernel_size=(2, kernel_size_21), padding=(0, (kernel_size_21 - 1) // 2),
                                 dilation=1)


    def forward(self, x):
        out = x
        #### 通道注意力

        ###中心像素的光谱
        out_x = self.ScaleMaskModule(out)
        out_x1 = self.avg_pool(out_x)
        out_x1 = out_x1.reshape(out_x1.shape[0], -1)
        out_x2 = reversed(out_x1.permute(1, 0)).permute(1, 0)  # 原地翻转，倒序:[1,2,3]->[3,2,1]

        out_x1 = out_x1.reshape(out_x1.shape[0], 1, 1, out_x1.shape[1])
        out_x2 = out_x2.reshape(out_x2.shape[0], 1, 1, out_x2.shape[1])

        out_xx = torch.cat([out_x1, out_x2], dim=2)
        #######
        ###全局空间的光谱
        out1 = self.avg_pool(out)
        out1 = out1.reshape(out1.shape[0], -1)

        out2 = reversed(out1.permute(1, 0)).permute(1, 0)  # 原地翻转，倒序:[1,2,3]->[3,2,1]

        out1 = out1.reshape(out1.shape[0], 1, 1, out1.shape[1])
        out2 = out2.reshape(out2.shape[0], 1, 1, out2.shape[1])

        outx = torch.cat([out1, out2], dim=2)
        #########
        at1 = F.sigmoid(self.conv1d(outx)).permute(0, 3, 1, 2) * F.sigmoid(self.conv1d1(out_xx)).permute(0, 3, 1, 2)

        at = F.sigmoid((at1 - 0.2) * 2)
        #print(at)
        #at = F.sigmoid(self.conv1d(outx)).permute(0, 3, 1, 2)
        out = out * at

        return out

class LE_DSC3D(nn.Module):# 3D LE-DSC
    def __init__(self, nin, nout,kernel_size_c,kernel_size_h,kernel_size_w,pca_components, patch_size, padding=True):
        super(LE_DSC3D, self).__init__()
        self.nout = nout
        self.nin = nin
        self.at1 = NCAM3D(self.nin*pca_components, patch_size)
        self.at2 = NCAM3D(self.nout * pca_components, patch_size)

        if padding == True:

         self.depthwise = nn.Conv3d(nin, nin, kernel_size=(kernel_size_c, 1, kernel_size_w),
                                   padding=((kernel_size_c - 1) // 2, 0, (kernel_size_w - 1) // 2), groups=nin)
         self.depthwise1 = nn.Conv3d(nin, nin, kernel_size=(kernel_size_c, kernel_size_h, 1),
                                    padding=((kernel_size_c - 1) // 2, (kernel_size_h - 1) // 2, 0), groups=nin)
         self.depthwise2 = nn.Conv3d(nin, nin, kernel_size=(1,kernel_size_h,kernel_size_w),
                                    padding=(0, (kernel_size_h - 1) // 2, (kernel_size_w - 1) // 2), groups=nin)

        else:
         self.depthwise = nn.Conv3d(nin, nin, kernel_size=(kernel_size_c,1,kernel_size_w),  groups=nin)
         self.depthwise1 = nn.Conv3d(nin, nin, kernel_size=(kernel_size_c,kernel_size_h,1),  groups=nin)
         self.depthwise2 = nn.Conv3d(nin, nin, kernel_size=(1,kernel_size_h,kernel_size_w),  groups=nin)

        self.pointwise = nn.Conv3d(nin, nout, kernel_size=1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        out1 = self.depthwise(x)

        out2 = self.depthwise1(x)

        out3 = self.depthwise2(x)

        out3 = out1+out2+out3 #

        out =out3
        #### 通道注意力
        out = self.at1(out)

        out = self.pointwise(out)

        #### 通道注意力
        out = self.at2(out)
        ####

        return out
###################################################################
class LE_DSC2D(nn.Module):#深度可分离混合卷积 ---- 2D数据版本
    def __init__(self, nin, nout, kernel_size_h, kernel_size_w, patch_size, padding=True):
        super(LE_DSC2D, self).__init__()
        self.nout = nout
        self.nin = nin
        self.at1 = NCAM2D(self.nin, patch_size)
        self.at2 = NCAM2D(self.nout, patch_size)

        if padding == True:

         self.depthwise = nn.Conv2d(nin, nin, kernel_size=(kernel_size_h,1), padding=((kernel_size_h - 1) // 2,0), groups=nin)
         self.depthwise1 = nn.Conv2d(nin, nin, kernel_size=(1,kernel_size_w), padding=(0,(kernel_size_w - 1) // 2), groups=nin)
        else:

         self.depthwise = nn.Conv2d(nin, nin, kernel_size=(kernel_size_h,1),  groups=nin)
         self.depthwise1 = nn.Conv2d(nin, nin, kernel_size=(1,kernel_size_w),  groups=nin)

        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):

        out1 = self.depthwise(x)

        out2 = self.depthwise1(x)

        out3 = out1+out2 #

        out =out3
        #### 通道注意力
        out = self.at1(out)

        out = self.pointwise(out)
        #### 通道注意力
        out = self.at2(out)
        ####

        return out

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out
class LE_HCL(nn.Module):#输入默认为一个三维块，即三维通道数为1
    def __init__(self, ax, aa, c, pca_components, patch_size):#ax二维通道数，c卷积核大小，d为padding和dilation大小
        super(LE_HCL, self).__init__()
        self.conv3d = nn.Sequential(
            LE_DSC3D(1, ax, c, c, c, pca_components, patch_size),
            nn.BatchNorm3d(ax),
            hswish(),

        )
        self.conv2d = nn.Sequential(
            LE_DSC2D(aa, aa // ax, c, c, patch_size),
            nn.BatchNorm2d(aa // ax),
            hswish(),

        )

    def forward(self, x):

        out = self.conv3d(x)

        out = out.reshape(out.shape[0], -1, out.shape[3], out.shape[4])
        out = self.conv2d(out)

        # out = out.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        out = out + x

        return out
# 主模型
class Lite_HCNet(nn.Module):
    def __init__(self, in_channels, class_num, patch_size):
        super(Lite_HCNet, self).__init__()
        ########
        e = 3 # LE-HCL中的e参数
        self.unit1 = LE_HCL(e, e*in_channels, 3, in_channels, patch_size)
        self.unit2 = LE_HCL(e, e*in_channels, 7, in_channels, patch_size)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(in_channels, class_num)

    def forward(self, x):

        out1 = self.unit1(x)
        out2 = self.unit2(x)

        out = out1+out2

        # out = out.reshape(out.shape[0], -1, out.shape[3], out.shape[4])
        out = self.avg_pool(out)
        out = out.reshape(out.shape[0], -1)

        out = self.fc1(out)

        return out

if __name__ == '__main__':
    model = Lite_HCNet(15, 10, 9).cuda()
    x = torch.randn(32, 15, 9, 9).cuda()
    y = model(x)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型总参数量: {total_params:,} 个参数')
    
    # 计算可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'可训练参数量: {trainable_params:,} 个参数')
