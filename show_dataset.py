"""
展示MNIST数据集的脚本
"""
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

def load_mnist_data():
    """
    加载MNIST数据集
    """
    # 定义数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载训练集
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    return train_dataset

def show_mnist_samples(dataset, num_samples=10):
    """
    展示MNIST数据集的样本
    Args:
        dataset: MNIST数据集
        num_samples: 要展示的样本数量（默认为100，即10x10的网格）
    """
    # 创建10x10的子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    # fig.suptitle('MNIST Dataset Samples', fontsize=16)
    
    # 随机选择10个样本
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx, ax in enumerate(axes.flat):
        if idx < num_samples:
            # 获取图像和标签
            image, label = dataset[indices[idx]]
            
            # 将图像转换为numpy数组并去除通道维度
            image = image.squeeze().numpy()
            
            # 显示图像
            ax.imshow(image, cmap='gray')
            
            # 设置标题为标签
            ax.set_title(f'True Label: {label}', fontsize=25)
            
            # 移除坐标轴
            ax.axis('off')
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 创建保存目录
    os.makedirs('results', exist_ok=True)
    
    # 保存图像
    plt.savefig('results/mnist_samples_2x2.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    主函数
    """
    print("正在加载MNIST数据集...")
    dataset = load_mnist_data()
    
    print("正在展示MNIST数据集样本...")
    show_mnist_samples(dataset)
    
    print("样本展示已完成，结果已保存到 results/mnist_samples.png")

if __name__ == "__main__":
    main() 