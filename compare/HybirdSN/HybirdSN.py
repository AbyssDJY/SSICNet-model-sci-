import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

class HybridSN(nn.Module):
    def __init__(self,class_num):
        super(HybridSN, self).__init__()
        #self.extra = GloRe_Unit_3D(1, 64)
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), stride=1, padding=0),
            #nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            #nn.PReLU(inplace=True),
        )
        #self.extra_2 = GloRe_Unit_3D(8, 64)
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), stride=1, padding=0),
            #nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        #self.extra_4 = GloRe_Unit_3D(16, 64)
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=0),
            #nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        #self.extra = GloRe_Unit_3D(32, 64)
        #extra_1 = GloRe_Unit_2D(256, 64)
        self.conv2d_4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=1, padding=0),# HU-128 salinas-64  longkou-64
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #nn.ReLU(inplace=True),
        )
        #self.extra_3 = GloRe_Unit_2D(16, 64)


        #self.extra_1 = GloRe_Unit_2D(16, 48)
        self.fc1 = nn.Linear(64, 256)#HU-64 salinas-64  longkou-576
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, class_num)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        #out = self.extra(x)
        out = self.conv3d_1(x)
        #out = self.extra(out)
        out = self.conv3d_2(out)
        #out = self.extra(out)
        out = self.conv3d_3(out)
        #out = self.extra_2(out)
        out = out.reshape(out.shape[0], -1, out.shape[3], out.shape[4])
        #out = self.extra_1(out)
        out = self.conv2d_4(out)
        #out = self.conv2d_5(out)
        #out = self.extra_3(out)
        out = out.reshape(out.shape[0], -1)
        out = F.relu(self.dropout(self.fc1(out)))
        out = F.relu(self.dropout(self.fc2(out)))
        out = self.fc3(out)
        return out
    
    def print_model_params(self):
        """计算并打印模型参数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'HybridSN模型总参数量: {total_params:,} 个参数')
        print(f'HybridSN模型可训练参数量: {trainable_params:,} 个参数')
        
        # 打印各层参数量
        print("\n各层参数量详情:")
        for name, param in self.named_parameters():
            print(f"{name}: {param.numel():,} 个参数")

if __name__ == '__main__':
    model=HybridSN(15).cuda()
    x = torch.randn(32,1,15, 9, 9).cuda()
    y = model(x)