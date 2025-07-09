import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

class Attention(nn.Module):
    def __init__(self, n, nin, in_channels):
        super(Attention, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.GAP = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv3d(1,1,kernel_size=(n, 1, 1),padding=((n-1)//2,0,0))

        self.AvgPool = nn.AvgPool3d((in_channels//2,1,1))# 黄河口设为（9,1,1），其余为（10,1,1）
        self.conv1 = nn.Conv3d(nin, nin, kernel_size=(1, n, n), stride=1, padding=(0, (n - 1) // 2, (n - 1) // 2))

        self.fc1 = nn.Linear(nin, nin // 8, bias=False)

        self.fc2 = nn.Linear(nin// 8, nin, bias=False)

    def forward(self, x):
        # 输入x的形状: [batch_size=32, channels=160, depth=8, height=7, width=7]
        n1,c,l,w,h = x.shape
        
        # SE (Squeeze-and-Excitation) 分支
        # 1. 调换维度使通道在第三维 [32, 8, 160, 7, 7]
        # 2. 全局平均池化得到 [32, 8, 160, 1, 1]
        # 3. 调回维度 [32, 160, 8, 1, 1]
        # 4. 1x1卷积后得到 [32, 1, 8, 1, 1]
        se = self.sigmoid(self.conv(self.GAP(x.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)))

        # SA (Spatial Attention) 分支
        # 1. 在通道维度上进行平均池化 [32, 160, 1, 7, 7]
        # 2. 空间卷积后保持形状 [32, 160, 1, 7, 7]
        sa = self.sigmoid(self.conv1(self.AvgPool(x)))

        # CA (Channel Attention) 分支
        # 1. 全局平均池化 [32, 160, 1, 1, 1]
        # 2. 展平 [32, 160]
        # 3. 全连接降维 [32, 20]
        # 4. 全连接升维 [32, 160]
        # 5. 重塑 [32, 160, 1, 1, 1]
        x1 = self.GAP(x)
        x1 = x1.reshape(n1, -1)
        f1 = self.fc1(x1)
        f2 = self.fc2(f1)
        ca = self.sigmoid(f2)
        ca = ca.reshape(n1, c, 1, 1, 1)

        # 融合三个注意力分支
        # 广播机制自动扩展维度
        w = se*sa*ca  # [32, 160, 8, 7, 7]
        out = x*w     # [32, 160, 8, 7, 7]

        return out

class Unit(nn.Module):
    def __init__(self, n):
        super(Unit, self).__init__()

        # 第一个子块：输入通道64，输出通道32
        self.bn1 = nn.BatchNorm3d(64)                    # 输入: [32, 64, 8, 7, 7]
        # 空间特征提取：nxn卷积，保持空间尺寸不变
        self.Conv3d_1 = nn.Conv3d(64, 32, kernel_size=(1, n, n), stride=1,
                                 padding=(0,(n-1)//2,(n-1)//2))  # 输出: [32, 32, 8, 7, 7]
        self.bn2 = nn.BatchNorm3d(32)
        # 通道特征提取：nx1x1卷积
        self.Conv3d_2 = nn.Conv3d(32, 32, kernel_size=(n, 1, 1), stride=1,
                                 padding=((n-1)//2,0,0))  # 输出: [32, 32, 8, 7, 7]

        # 第二个子块：输入通道96（32+64），输出通道32
        self.bn3 = nn.BatchNorm3d(96)                    # 输入: [32, 96, 8, 7, 7]
        self.Conv3d_3 = nn.Conv3d(96, 32, kernel_size=(1, n, n), stride=1,
                                 padding=(0, (n - 1) // 2, (n - 1) // 2))  # 输出: [32, 32, 8, 7, 7]
        self.bn4 = nn.BatchNorm3d(32)
        self.Conv3d_4 = nn.Conv3d(32, 32, kernel_size=(n, 1, 1), stride=1,
                                 padding=((n - 1) // 2, 0, 0))  # 输出: [32, 32, 8, 7, 7]

        # 第三个子块：输入通道128（32+64+32），输出通道32
        self.bn5 = nn.BatchNorm3d(128)                   # 输入: [32, 128, 8, 7, 7]
        self.Conv3d_5 = nn.Conv3d(128, 32, kernel_size=(1, n, n), stride=1,
                                 padding=(0, (n - 1) // 2, (n - 1) // 2))  # 输出: [32, 32, 8, 7, 7]
        self.bn6 = nn.BatchNorm3d(32)
        self.Conv3d_6 = nn.Conv3d(32, 32, kernel_size=(n, 1, 1), stride=1,
                                 padding=((n - 1) // 2, 0, 0))  # 输出: [32, 32, 8, 7, 7]

    def forward(self, x):
        # 第一个子块处理
        out1 = self.Conv3d_1(F.relu(self.bn1(x)))        # [32, 32, 8, 7, 7]
        x1 = self.Conv3d_2(F.relu(self.bn2(out1)))       # [32, 32, 8, 7, 7]
        out1 = torch.cat([x1, x], dim=1)                 # [32, 96, 8, 7, 7]

        # 第二个子块处理
        out2 = self.Conv3d_3(F.relu(self.bn3(out1)))     # [32, 32, 8, 7, 7]
        x2 = self.Conv3d_4(F.relu(self.bn4(out2)))       # [32, 32, 8, 7, 7]
        out2 = torch.cat([x2, x, x1], dim=1)             # [32, 128, 8, 7, 7]

        # 第三个子块处理
        out3 = self.Conv3d_5(F.relu(self.bn5(out2)))     # [32, 32, 8, 7, 7]
        x3 = self.Conv3d_6(F.relu(self.bn6(out3)))       # [32, 32, 8, 7, 7]
        out3 = torch.cat([x3, x, x1, x2], dim=1)         # [32, 160, 8, 7, 7]

        return out3

class baseNet(nn.Module):
    def __init__(self, in_channels, class_num):
        super(baseNet, self).__init__()

        self.conv3d = nn.Sequential(
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
            nn.Conv3d(1, 64, kernel_size=(3, 3, 3), stride=(2,1,1), padding=(1,1,1),)
        )
        self.bneck_1 = Unit(3)
        self.at1 = Attention(3, 160, in_channels)

        self.bneck_2 = Unit(5)
        self.at2 = Attention(5, 160, in_channels)

        self.bneck_3 = Unit(7)
        self.at3 = Attention(7, 160, in_channels)

        self.conv3d_2 = nn.Sequential(
            nn.BatchNorm3d(160),
            nn.ReLU(inplace=True),
            nn.Conv3d(160, 256, kernel_size=(in_channels//2, 1, 1), stride=1,)#黄河口设为（9,1,1），其余为（10,1,1）
        )
        self.conv3d_3 = nn.Sequential(
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
            nn.Conv3d(1, 64, kernel_size=(256, 1, 1), stride=1, )
        )

        self.MaxPool = nn.MaxPool3d((1, 7, 7))
        self.conv3d_4 = nn.Sequential(
            nn.BatchNorm3d(64),#黄河口时隐藏
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=1,padding=(0,1,1))
        )

        self.GAP = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Linear(64, class_num)


    def forward(self, x):
        # 输入x的形状: [32, 103, 7, 7]
        if len(x.shape) == 4:
            x = x.unsqueeze(1)                    # [32, 1, 103, 7, 7]
        x = self.conv3d(x)                        # [32, 64, 52, 7, 7]

        # 三个并行分支处理
        x1 = self.bneck_1(x)                      # [32, 160, 52, 7, 7]
        x1 = self.at1(x1)                         # [32, 160, 52, 7, 7]

        x2 = self.bneck_2(x)                      # [32, 160, 52, 7, 7]
        x2 = self.at2(x2)                         # [32, 160, 52, 7, 7]

        x3 = self.bneck_3(x)                      # [32, 160, 52, 7, 7]
        x3 = self.at3(x3)                         # [32, 160, 52, 7, 7]

        # 特征融合
        out = x1 + x2 + x3                        # [32, 160, 52, 7, 7]

        # 特征压缩和变换
        out = self.conv3d_2(out)                  # [32, 256, 1, 7, 7]
        out = out.permute(0, 2, 1, 3, 4)         # [32, 1, 256, 7, 7]
        out = self.conv3d_3(out)                  # [32, 64, 1, 7, 7]
        out = self.MaxPool(out)                   # [32, 64, 1, 1, 1]

        # 最终特征提取
        out = self.conv3d_4(out)                  # [32, 64, 1, 1, 1]
        out = self.GAP(out)                       # [32, 64, 1, 1, 1]

        # 分类
        out = out.reshape(out.shape[0], -1)       # [32, 64]
        out = self.fc(out)                        # [32, num_classes]
        
        return out

if __name__ == '__main__':
    model = baseNet(128, 10)
    x = torch.randn(2, 128, 9, 9)
    y = model(x)
    print(y.shape)
    
    # 计算模型总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型总参数量: {total_params:,} 个参数')
    
    # 计算可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'可训练参数量: {trainable_params:,} 个参数')

#初始参数量：
    # 模型总参数量: 5,636,352 个参数
    # 可训练参数量: 5,636,352 个参数


    #优化：
    #优化思路说明：
    # 1. 使用深度可分离卷积：将标准卷积分解为深度卷积和逐点卷积，大幅减少参数量
    # 2. 减少中间通道数：将Unit模块中的中间通道数从32减少到16
    # 3. 简化注意力机制：
    #    - 减小池化窗口大小
    #    - 减少通道压缩比例
    #    - 在卷积后添加BatchNorm提高稳定性
    # 4. 保持输入输出维度不变，只修改内部结构
    # 原始代码中的参数量主要集中在：

    # 1. 标准3D卷积层
    # 2. 多分支注意力机制
    # 3. Unit模块中的密集连接