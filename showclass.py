import scipy.io as sio
import numpy as np
import argparse

class Args:
    def __init__(self):
        self.dataset = None
        self.hsi_bands = None
        self.num_class = None
        self.patch_size = None
        self.PCA = None

def build_data_loader(args):
    if args.dataset == 'PaviaU':
        X = sio.loadmat('./oridata/PaviaU/PaviaU.mat')['paviaU']
        y = sio.loadmat('./oridata/PaviaU/PaviaU_gt.mat')['paviaU_gt']
        args.hsi_bands = X.shape[2]
        args.num_class = 9
        args.patch_size = 7
        args.PCA = 12
    if args.dataset == 'Houston':
        X = sio.loadmat('./oridata/Houston/Houston.mat')['Houston']
        y = sio.loadmat('./oridata/Houston/Houston_gt.mat')['Houston_gt']
        args.hsi_bands = X.shape[2]
        args.num_class = 15
        args.patch_size = 13  # 要是奇数
        args.PCA = 16  # pca要是偶数
    if args.dataset == 'Indian':
        X = sio.loadmat('./oridata/Indian/Indian_pines_corrected.mat')['indian_pines_corrected']
        y = sio.loadmat('./oridata/Indian/Indian_pines_gt.mat')['indian_pines_gt']
        args.hsi_bands = X.shape[2]
        args.num_class = 16
        args.patch_size = 19
        args.PCA = 16
    if args.dataset == 'Salinas':
        X = sio.loadmat('./oridata/salinas/Salinas.mat')['salinas']
        y = sio.loadmat('./oridata/salinas/Salinas_gt.mat')['salinas_gt']
        args.hsi_bands = X.shape[2]  # 224 bands
        args.num_class = 16  # 修改为16
        args.patch_size = 9
        args.PCA = 16
    if args.dataset == 'WHU_Hi_LongKou':
        X = sio.loadmat('./oridata/WHU_Hi_LongKou/WHU_Hi_LongKou.mat')['WHU_Hi_LongKou']
        y = sio.loadmat('./oridata/WHU_Hi_LongKou/WHU_Hi_LongKou_gt.mat')['WHU_Hi_LongKou_gt']
        args.hsi_bands = X.shape[2]
        args.num_class = 9  # WHU_Hi_LongKou数据集有9个类别
        args.patch_size = 13  # 设置patch大小为7
        args.PCA = 16  # PCA降维到16维
    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)
    return X, y

    color_map = {
        0: (0.2 * 255, 0.2 * 255, 0.2 * 255),  # 深灰色
        1: (0.0 * 255, 1.0 * 255, 0.0 * 255),  # 绿色
        2: (0.0 * 255, 0.0 * 255, 1.0 * 255),  # 蓝色
        3: (1.0 * 255, 1.0 * 255, 0.0 * 255),  # 黄色
        4: (1.0 * 255, 0.0 * 255, 1.0 * 255),  # 品红
        5: (0.0 * 255, 1.0 * 255, 1.0 * 255),  # 青色
        6: (0.5 * 255, 0.5 * 255, 0.5 * 255),  # 灰色
        7: (1.0 * 255, 0.5 * 255, 0.0 * 255),  # 橙色
        8: (0.5 * 255, 0.0 * 255, 0.5 * 255),  # 紫色
        9: (0.0 * 255, 0.5 * 255, 0.5 * 255),  # 青绿色
        10: (0.5 * 255, 0.5 * 255, 0.0 * 255),  # 橄榄色
        11: (0.8 * 255, 0.2 * 255, 0.2 * 255),  # 浅红色
        12: (0.2 * 255, 0.8 * 255, 0.2 * 255),  # 浅绿色
        13: (0.2 * 255, 0.2 * 255, 0.8 * 255),  # 浅蓝色
        14: (0.8 * 255, 0.8 * 255, 0.2 * 255),  # 浅黄色
        15: (0.8 * 255, 0.2 * 255, 0.8 * 255),  # 浅品红
        16: (0.2 * 255, 0.8 * 255, 0.8 * 255),  # 浅青色
        17: (0.6 * 255, 0.4 * 255, 0.2 * 255),  # 棕色
        18: (0.2 * 255, 0.6 * 255, 0.4 * 255),  # 深青色
    }


def analyze_dataset(dataset_name):
    args = Args()
    args.dataset = dataset_name
    X, y = build_data_loader(args)
    
    # 获取类别信息
    unique_labels = np.unique(y)
    # 移除背景类（通常标记为0）
    if 0 in unique_labels:
        unique_labels = unique_labels[1:]
    
    print(f"\n{'-'*50}")
    print(f"数据集: {dataset_name}")
    print(f"总类别数: {len(unique_labels)}")
    print(f"数据维度: {X.shape}")
    print(f"标签维度: {y.shape}")
    print(f"\n类别详情:")
    print(f"{'类别ID':<10}{'样本数量':<15}")
    print(f"{'-'*25}")
    
    total_samples = 0
    for label in unique_labels:
        samples = np.sum(y == label)
        total_samples += samples
        print(f"{label:<10}{samples:<15}")
    
    print(f"\n总样本数: {total_samples}")

def main():
    print("高光谱数据集类别分析")
    print("=" * 50)
    
    datasets = [ 'Houston', 'Salinas', 'WHU_Hi_LongKou']
    
    for dataset in datasets:
        analyze_dataset(dataset)

if __name__ == "__main__":
    main()