import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import math
warnings.filterwarnings("ignore")

class Attention(nn.Module):

    def __init__(self, n, nin, in_channels):
        super(Attention, self).__init__()
        # 简化SE分支
        self.sigmoid = nn.Sigmoid()
        self.GAP = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv3d(1,1,kernel_size=(n, 1, 1),padding=((n-1)//2,0,0))

        # 简化CA分支，减少中间层维度
        self.fc1 = nn.Linear(nin, nin // 16, bias=False)  # 降维更多
        self.fc2 = nn.Linear(nin// 16, nin, bias=False)

    def forward(self, x):
        n1,c,l,w,h = x.shape
        # 只保留SE和CA两个分支
        x_perm = x.permute(0, 2, 1, 3, 4)
        gap_out = self.GAP(x_perm)
        perm_back = gap_out.permute(0, 2, 1, 3, 4)
        se = self.sigmoid(self.conv(perm_back))

        x1 = self.GAP(x)
        x1 = x1.reshape(n1, -1)
        f1 = self.fc1(x1)
        f2 = self.fc2(f1)
        ca = self.sigmoid(f2)
        ca = ca.reshape(n1, c, 1, 1, 1)

        w = se*ca
        out = x*w

        return out



class SpatialSelfCorrelation(nn.Module):
    """空间自相关模块 (S-SC)，聚合空间信息"""
    def __init__(self, channels, reduction=2):
        super(SpatialSelfCorrelation, self).__init__()
        self.channels = channels
        self.reduction = reduction
        self.linear = nn.Conv3d(channels, channels//reduction, kernel_size=(1, 1, 1))
        
    def forward(self, x):
        b, c, d, h, w = x.shape
        # 降维
        x_reduced = self.linear(x)  # B, C/r, D, H, W
        
        # 重塑为矩阵乘法形式
        q = x_reduced.reshape(b, c//self.reduction, -1)  # B, C/r, D*H*W
        k = x_reduced.reshape(b, c//self.reduction, -1)  # B, C/r, D*H*W
        v = x.reshape(b, c, -1)  # B, C, D*H*W
        
        # 矩阵乘法计算注意力
        attn = torch.bmm(q.transpose(1, 2), k)  # B, D*H*W, D*H*W
        attn = F.softmax(attn / math.sqrt(c//self.reduction), dim=2)
        
        # 应用注意力
        out = torch.bmm(v, attn)  # B, C, D*H*W
        out = out.reshape(b, c, d, h, w)
        
        return out + x  # 残差连接

class ChannelSelfCorrelation(nn.Module):
    """通道自相关模块 (C-SC)，聚合通道信息"""
    def __init__(self, channels):
        super(ChannelSelfCorrelation, self).__init__()
        self.channels = channels
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(channels, channels // 8)
        self.fc2 = nn.Linear(channels // 8, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, d, h, w = x.shape
        
        # 全局平均池化
        y = self.gap(x).reshape(b, c)
        
        # 通道注意力
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).reshape(b, c, 1, 1, 1)
        
        return x * y  # 应用通道注意力

# 移除 DualFeatureExtraction 类

# 添加新的空间-通道交叉注意力模块
class SpatialChannelCrossAttention(nn.Module):
    """空间-通道交叉注意力模块"""
    def __init__(self, channels):
        super(SpatialChannelCrossAttention, self).__init__()
        self.channels = channels
        
        # 空间到通道的投影
        self.spatial_proj = nn.Conv3d(channels, channels, kernel_size=1)
        # 通道到空间的投影
        self.channel_proj = nn.Conv3d(channels, channels, kernel_size=1)
        
        # 注意力计算的缩放因子
        self.scale = channels ** -0.5
        
    def forward(self, x):
        b, c, d, h, w = x.shape
        
        # 空间特征
        spatial_feat = self.spatial_proj(x)
        spatial_feat_flat = spatial_feat.reshape(b, c, -1)  # B, C, D*H*W
        
        # 通道特征
        channel_feat = self.channel_proj(x)
        channel_feat_flat = channel_feat.reshape(b, c, -1)  # B, C, D*H*W
        
        # 计算空间到通道的注意力
        attn_s2c = torch.bmm(spatial_feat_flat, channel_feat_flat.transpose(1, 2))  # B, C, C
        attn_s2c = F.softmax(attn_s2c * self.scale, dim=-1)
        out_s2c = torch.bmm(attn_s2c, channel_feat_flat)  # B, C, D*H*W
        
        # 计算通道到空间的注意力
        attn_c2s = torch.bmm(channel_feat_flat.transpose(1, 2), spatial_feat_flat)  # B, D*H*W, D*H*W
        attn_c2s = F.softmax(attn_c2s * self.scale, dim=-1)
        out_c2s = torch.bmm(spatial_feat_flat, attn_c2s)  # B, C, D*H*W
        
        # 融合两种注意力的结果
        out = out_s2c + out_c2s
        out = out.reshape(b, c, d, h, w)
        
        return out + x  # 残差连接

class SCCModule(nn.Module):
    """结合S-SC、C-SC和交叉注意力"""
    def __init__(self, channels):
        super(SCCModule, self).__init__()

        self.s_sc = SpatialSelfCorrelation(channels)
        self.c_sc = ChannelSelfCorrelation(channels)
        # 添加交叉注意力
        self.cross_attn = SpatialChannelCrossAttention(channels)
        
    def forward(self, x):
        # 空间自相关
        spatial_feat = self.s_sc(x)
        
        # 通道自相关
        channel_feat = self.c_sc(spatial_feat)
        
        # 空间-通道交叉注意力
        out = self.cross_attn(channel_feat)
        
        return out

#GLMIX
class ConjugateModule(nn.Module):
    """共轭模块设计，包含聚类、多头自注意力和分配模块"""
    def __init__(self, channels, num_slots=8, num_heads=4):
        super(ConjugateModule, self).__init__()
        self.channels = channels
        self.num_slots = num_slots
        self.num_heads = num_heads
        
        # 聚类模块
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.slot_init = nn.Parameter(torch.randn(1, channels, num_slots))
        self.slot_proj = nn.Linear(channels, channels)
        
        # 多头自注意力机制 (MHSA)
        self.norm = nn.LayerNorm(channels)
        self.mhsa = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        
        # 分配模块
        self.feat_proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.slot_to_feat = nn.Linear(channels, channels)
        
    def forward(self, x):
        b, c, d, h, w = x.shape
        
        # 聚类模块：特征网格转化为语义槽
        # 平均池化初始化
        x_flat = x.reshape(b, c, d*h*w)  # B, C, D*H*W
        
        # 初始化语义槽
        slots = self.slot_init.expand(b, -1, -1)  # B, C, num_slots
        
        # 计算余弦相似度
        x_norm = F.normalize(x_flat, dim=1)
        slots_norm = F.normalize(slots, dim=1)
        sim = torch.bmm(slots_norm.transpose(1, 2), x_norm)  # B, num_slots, D*H*W
        
        # 软分配
        attn = F.softmax(sim, dim=2)  # B, num_slots, D*H*W
        
        # 更新语义槽
        updates = torch.bmm(x_flat, attn.transpose(1, 2))  # B, C, num_slots
        slots = self.slot_proj(updates.transpose(1, 2)).transpose(1, 2)  # B, C, num_slots
        
        # 添加多头自注意力处理语义槽
        slots_trans = slots.transpose(1, 2)  # B, num_slots, C
        slots_norm = self.norm(slots_trans)
        slots_attn, _ = self.mhsa(slots_norm, slots_norm, slots_norm)
        slots_trans = slots_trans + slots_attn  # 残差连接
        slots = slots_trans.transpose(1, 2)  # B, C, num_slots
        
        # 分配模块：语义槽转换后分配到空间位置
        feat_proj = self.feat_proj(x)  # B, C, D, H, W
        feat_flat = feat_proj.reshape(b, c, d*h*w)  # B, C, D*H*W
        
        # 计算语义槽与特征的相似度
        slots_feat = self.slot_to_feat(slots.transpose(1, 2)).transpose(1, 2)  # B, C, num_slots
        sim = torch.bmm(slots_feat.transpose(1, 2), feat_flat)  # B, num_slots, D*H*W
        attn = F.softmax(sim, dim=1)  # B, num_slots, D*H*W
        
        # 分配回特征
        out = torch.bmm(slots, attn)  # B, C, D*H*W
        out = out.reshape(b, c, d, h, w)
        
        # 融合原始特征和语义槽特征
        return out + x  # 残差连接

class Unit(nn.Module):
    """
    密集连接单元，包含三个子块，每个子块包含:
    1. 空间卷积 (n×n)
    2. 通道卷积 (n×1×1)
    3. 密集连接：将前面所有层的输出连接起来
    """
    def __init__(self, n):
        super(Unit, self).__init__()
        # 第一个子块
        self.bn1 = nn.BatchNorm3d(32)
        self.conv1 = nn.Conv3d(32, 64, kernel_size=(1, n, n), stride=1, padding=(0, (n-1)//2, (n-1)//2))
        self.gelu = nn.GELU()
        self.bn2 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(n, 1, 1), stride=1, padding=((n-1)//2, 0, 0))

    def forward(self, x):
        # 输入 x: [32, 32, 8, 7, 7]
        out1 = self.conv1(self.gelu(self.bn1(x)))
        x1 = self.conv2(self.gelu(self.bn2(out1)))
        return x1


class baseNet(nn.Module):
    """
    主网络架构，包含以下关键组件：
    1. 初始特征提取
    2. 三个不同感受野(3,5,7)的特征提取分支
    3. 特征融合和降维
    4. 分类头
    """
    def __init__(self, in_channels, class_num):
        super(baseNet, self).__init__()
        # 初始特征提取
        self.conv3d = nn.Sequential(
            nn.BatchNorm3d(1),
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.GELU(),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),
            nn.GELU(),
        )

        # 三个并行分支，使用不同大小的感受野
        self.bneck_1 = Unit(3)  # 3×3感受野
        # self.at1 = Attention(3, 128, in_channels)  # 对应的注意力机制

        # 添加 HiT-SR 的 SCC 模块
        self.scc_module = SCCModule(128)
        
        # 添加 GLMix 的共轭模块
        self.conjugate = ConjugateModule(128, num_slots=8)
        # 新增简单卷积层用于enhanced_features_2
        self.enhance_conv = nn.Conv3d(128, 128, kernel_size=1)
        


        # 特征融合和降维模块
        # 减少中间通道数
        self.conv3d_2 = nn.Sequential(
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=(in_channels//2, 1, 1), stride=1,)
        )
        self.conv3d_3 = nn.Sequential(
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
            nn.Conv3d(1, 32, kernel_size=(128, 1, 1), stride=1, )
        )

        self.MaxPool = nn.MaxPool3d((1, 7, 7))
        self.conv3d_4 = nn.Sequential(
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=1,padding=(0,1,1))
        )
        self.GAP = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, class_num)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        x = self.conv3d(x)

        # 基础特征提取
        x1 = self.bneck_1(x)
        
        # 应用 SCC 模块和共轭模块增强特征
        enhanced_features = self.scc_module(x1)
        enhanced_features = self.conjugate(enhanced_features)
        # 新增：对enhanced_features做简单卷积
        # enhanced_features_2 = self.enhance_conv(enhanced_features)
        # 融合基础特征和增强特征 (残差连接)
        out = x1 + enhanced_features 
        # + enhanced_features_2

        out = self.conv3d_2(out)
        out = out.permute(0, 2, 1, 3, 4)
        out = self.conv3d_3(out)
        out = self.MaxPool(out)

        out = self.conv3d_4(out)
        out = self.GAP(out)

        out = out.reshape(out.shape[0], -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

if __name__ == '__main__':
    model = baseNet(12, 10)
    x = torch.randn(2, 12, 9, 9)
    y = model(x)
    print(y.shape)

    # 计算模型总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型总参数量: {total_params:,} 个参数')
    
    # 计算可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'可训练参数量: {trainable_params:,} 个参数')
    
    # 计算模型的FLOPs
    def count_flops(model, input_size=(2, 12, 9, 9)):
        from thop import profile
        input = torch.randn(input_size)
        flops, params = profile(model, inputs=(input, ))
        print(f'模型FLOPs: {flops/1e9:.2f} G')
        return flops, params
    
    try:
        count_flops(model)
    except:
        print("无法计算FLOPs，请安装thop库: pip install thop")