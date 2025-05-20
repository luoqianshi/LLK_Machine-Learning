# 使用PyTorch加载MNIST数据集，并使用自定义的MLP类进行训练和分类
import numpy as np
import torch
from torchvision import datasets, transforms
from MLP_Numpy import MLP
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可复现
np.random.seed(42)
torch.manual_seed(42)

# 定义数据变换和加载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载训练集和测试集
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 将PyTorch数据集转换为NumPy数组
def convert_to_numpy(loader):
    data_list = []
    targets_list = []
    for data, targets in loader:
        # 将数据从torch张量转换为numpy数组，并展平为向量
        data = data.numpy().reshape(data.shape[0], -1)
        targets = targets.numpy()
        
        data_list.append(data)
        targets_list.append(targets)
    
    # 合并所有批次的数据
    data = np.vstack(data_list)
    targets = np.hstack(targets_list)
    
    return data, targets

# 将训练集和测试集转换为NumPy数组
print("正在转换数据集...")
X_train, y_train = convert_to_numpy(train_loader)
X_test, y_test = convert_to_numpy(test_loader)

print(f"训练集形状: {X_train.shape}, 标签形状: {y_train.shape}")
print(f"测试集形状: {X_test.shape}, 标签形状: {y_test.shape}")

# 将标签转换为one-hot编码
def one_hot_encode(y, num_classes):
    y_one_hot = np.zeros((y.shape[0], num_classes))
    y_one_hot[np.arange(y.shape[0]), y] = 1
    return y_one_hot

y_train_one_hot = one_hot_encode(y_train, 10)
y_test_one_hot = one_hot_encode(y_test, 10)

# 创建并训练MLP
print("初始化MLP模型...")
mlp = MLP(layer_sizes=[784, 128, 64, 10], 
          activation_functions=['relu', 'relu', 'softmax'],
          learning_rate=0.01)

print("开始训练模型...")
# 为了演示，减少训练轮数
mlp.train(X_train, y_train_one_hot, epochs=100, batch_size=64, verbose=True)

# 评估模型
print("评估模型...")
accuracy = mlp.evaluate(X_test, y_test_one_hot)
print(f"测试集准确率: {accuracy:.4f}")

# 获取预测结果
predictions = mlp.predict(X_test)

# 打印一些预测结果进行比较
def print_predictions(X, y_true, predictions, num_samples=5):
    indices = np.random.choice(len(X), num_samples, replace=False)
    print("\n预测结果示例:")
    print("-" * 30)
    print(f"{'索引':<10}{'真实标签':<10}{'预测标签':<10}")
    print("-" * 30)
    
    for i, idx in enumerate(indices):
        true_label = np.argmax(y_true[idx]) if len(y_true.shape) > 1 else y_true[idx]
        pred_label = predictions[idx]
        print(f"{idx:<10}{true_label:<10}{pred_label:<10}")

# 打印一些结果
print_predictions(X_test, y_test, predictions)

# 根据需要启用可视化
ENABLE_VISUALIZATION = False

if ENABLE_VISUALIZATION:
    try:
        # 可视化一些预测结果
        def visualize_predictions(X, y_true, predictions, num_samples=5):
            fig, axes = plt.subplots(1, num_samples, figsize=(12, 3))
            indices = np.random.choice(len(X), num_samples, replace=False)
            
            for i, idx in enumerate(indices):
                # 重塑图像数据用于显示
                img = X[idx].reshape(28, 28)
                true_label = np.argmax(y_true[idx]) if len(y_true.shape) > 1 else y_true[idx]
                pred_label = predictions[idx]
                
                axes[i].imshow(img, cmap='gray')
                axes[i].set_title(f'真实: {true_label}\n预测: {pred_label}')
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig('mnist_predictions.png')
            plt.show()
        
        visualize_predictions(X_test, y_test, predictions)
    except Exception as e:
        print(f"\n可视化过程中出现错误: {e}")
        print("跳过可视化步骤，请检查matplotlib版本兼容性。")