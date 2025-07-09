import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

class F3D(nn.Module):
    def __init__(self, class_num, dataset_name=None):
        super(F3D, self).__init__()

        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), stride=1, padding=0),
            nn.ReLU(inplace=True),

        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), stride=1, padding=0),
            nn.ReLU(inplace=True),

        )
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.ReLU(inplace=True),

        )
        self.conv3d_4 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 1, 1), stride=1, padding=0),
            nn.ReLU(inplace=True),

        )
        
        # 根据数据集名称动态设置fc1的输入维度
        fc1_input_dim = 1152 if dataset_name == 'WHU_Hi_LongKou' else 128
        
        self.fc1 = nn.Linear(fc1_input_dim, 256)  #input ：salinas：128， Huston：128 longkou:1152
        self.fc2 = nn.Linear(256, 128)  
        self.fc3 = nn.Linear(128, class_num)  
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):

        out = self.conv3d_1(x)

        out = self.conv3d_2(out)

        out = self.conv3d_3(out)

        out = self.conv3d_4(out)

        out = out.reshape(out.shape[0], -1)


        out = F.relu(self.dropout(self.fc1(out)))
        out = F.relu(self.dropout(self.fc2(out)))
        out = self.fc3(out)
        return out
    
    def print_model_params(self):
        """计算并打印模型参数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'F3D模型总参数量: {total_params:,} 个参数')
        print(f'F3D模型可训练参数量: {trainable_params:,} 个参数')
        
        # 打印各层参数量
        print("\n各层参数量详情:")
        for name, param in self.named_parameters():
            print(f"{name}: {param.numel():,} 个参数")


if __name__ == '__main__':
    # 测试代码

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    model.print_model_params()  # 新增：打印模型参数信息

