import numpy as np
import cv2
from osgeo import gdal
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def visualize_labels(labels=None, output_path='classification_map.png'):
    """
    将标签数据可视化并保存为图片
    参数:
        labels: 标签数据，如果为None则尝试读取TIFF文件
        output_path: 输出文件路径
    """
    # 如果没有传入labels，尝试读取TIFF文件
    if labels is None:
        possible_paths = [
'./results/simpoolformer/WHU_Hi_LongKou/Label_PRED.npy'

        ]
        
        for tif_path in possible_paths:
            try:
                dataset = gdal.Open(tif_path)
                if dataset:
                    labels = dataset.ReadAsArray()
                    print(f"成功读取TIFF文件: {tif_path}")
                    break
            except Exception as e:
                continue
        
        if labels is None:
            print("未能找到或读取任何预测结果TIFF文件")
            return

    # 定义颜色列表，按照类别顺序排列
    colors = [
        [0,0,0],
        [1.0, 0.0, 0.0],  # 0: 红色
        [238/255, 154/255, 0.0],  # 1: 橙色
        [1.0, 1.0, 0.0],  # 2: 黄色
        [0.0, 1.0, 0.0],  # 3: 绿色
        [0.0, 1.0, 1.0],  # 4: 青色
        [0.0, 139/255, 139/255],  # 5: 藏绿色
        [0.0, 0.0, 1.0],  # 6: 蓝色
        [1.0, 1.0, 1.0],  # 7: 白色
        [160/255, 32/255, 240/255],  # 8: 紫色
    ]
    
    # 创建颜色映射
    cmap = ListedColormap(colors)
    
    # 使用matplotlib生成可视化图像
    plt.figure(figsize=(10, 10))
    plt.imshow(labels, cmap=cmap, interpolation='nearest')
    plt.axis('off')  # 不显示坐标轴
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化结果已保存至: {os.path.abspath(output_path)}")

    # 如果需要使用OpenCV保存图像，可以将matplotlib图像转换为OpenCV格式
    # 以下是使用OpenCV的替代方法
    '''
    map_H, map_W = labels.shape
    segmented_image = np.zeros((map_H, map_W, 3), dtype=np.uint8)
    
    # 将ListedColormap应用到图像上
    for i in range(len(colors)):
        mask = (labels == i)
        segmented_image[mask] = [int(c*255) for c in colors[i]]
    
    try:
        cv2.imwrite(output_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))  # OpenCV使用BGR格式
        print(f"可视化结果已保存至: {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"保存图像时出错: {e}")
    '''

if __name__ == '__main__':
    visualize_labels()
