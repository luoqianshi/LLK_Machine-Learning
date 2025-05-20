"""
对比测试文件：比较自定义MLP和基于sklearn的MLP实现
"""
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
from MLP_Numpy import MLP
from MLP_sklearn import MLP_sklearn

# 设置随机种子以确保结果可复现
np.random.seed(42)
torch.manual_seed(42)

def load_mnist_data():
    """
    加载MNIST数据集并转换为NumPy数组
    """
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
    
    return X_train, y_train, y_train_one_hot, X_test, y_test, y_test_one_hot

def compare_models(X_train, y_train, y_train_one_hot, X_test, y_test, y_test_one_hot, epochs=10):
    """
    比较自定义MLP和sklearn MLP的性能
    """
    # 使用小一点的数据子集进行训练，以加快速度
    subset_size = 5000
    X_train_subset = X_train[:subset_size]
    y_train_subset = y_train[:subset_size]
    y_train_one_hot_subset = y_train_one_hot[:subset_size]
    
    # 共同参数
    layer_sizes = [784, 128, 64, 10]
    activation_functions = ['relu', 'relu', 'softmax']
    learning_rate = 0.01
    
    # 结果字典
    results = {
        'custom_mlp': {'train_time': 0, 'accuracy': 0, 'epochs': []},
        'sklearn_mlp': {'train_time': 0, 'accuracy': 0, 'epochs': []}
    }
    
    # ==== 自定义MLP ====
    print("\n" + "="*50)
    print("训练自定义MLP模型...")
    print("="*50)
    
    # 创建自定义MLP模型
    custom_mlp = MLP(
        layer_sizes=layer_sizes,
        activation_functions=activation_functions,
        learning_rate=learning_rate
    )
    
    # 记录开始时间
    start_time = time.time()
    
    # 训练模型
    for epoch in range(epochs):
        # 每次迭代随机选择一个小批量
        indices = np.random.choice(subset_size, 64, replace=False)
        batch_X = X_train_subset[indices]
        batch_y = y_train_one_hot_subset[indices]
        
        # 前向传播
        y_pred = custom_mlp.forward(batch_X)
        
        # 计算损失
        loss = custom_mlp.calculate_loss(y_pred, batch_y)
        
        # 反向传播和更新参数
        weight_gradients, bias_gradients = custom_mlp.backward(batch_X, batch_y)
        custom_mlp.update_parameters(weight_gradients, bias_gradients)
        
        # 记录每个epoch的损失
        results['custom_mlp']['epochs'].append(loss)
        
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
    
    # 记录训练时间
    results['custom_mlp']['train_time'] = time.time() - start_time
    
    # 评估模型
    custom_accuracy = custom_mlp.evaluate(X_test, y_test_one_hot)
    results['custom_mlp']['accuracy'] = custom_accuracy
    print(f"自定义MLP在测试集上的准确率: {custom_accuracy:.4f}")
    print(f"训练时间: {results['custom_mlp']['train_time']:.2f}秒")
    
    # ==== Sklearn MLP ====
    print("\n" + "="*50)
    print("训练Sklearn MLP模型...")
    print("="*50)
    
    # 创建sklearn MLP模型
    sklearn_mlp = MLP_sklearn(
        layer_sizes=layer_sizes,
        activation_functions=activation_functions,
        learning_rate=learning_rate
    )
    
    # 记录开始时间
    start_time = time.time()
    
    # 训练模型
    sklearn_mlp.train(X_train_subset, y_train_subset, epochs=epochs, batch_size=64, verbose=True)
    
    # 记录训练时间
    results['sklearn_mlp']['train_time'] = time.time() - start_time
    
    # 评估模型
    sklearn_accuracy = sklearn_mlp.evaluate(X_test, y_test)
    results['sklearn_mlp']['accuracy'] = sklearn_accuracy
    print(f"Sklearn MLP在测试集上的准确率: {sklearn_accuracy:.4f}")
    print(f"训练时间: {results['sklearn_mlp']['train_time']:.2f}秒")
    
    return results

def visualize_results(results):
    """
    可视化两个模型的性能比较
    """
    try:
        # 1. 准确率比较
        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.bar(models, accuracies, color=['blue', 'green'])
        plt.title('模型准确率比较')
        plt.ylabel('准确率')
        plt.ylim(0, 1)
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
        
        # 2. 训练时间比较
        train_times = [results[model]['train_time'] for model in models]
        
        plt.subplot(1, 2, 2)
        plt.bar(models, train_times, color=['blue', 'green'])
        plt.title('训练时间比较(秒)')
        plt.ylabel('时间(秒)')
        for i, v in enumerate(train_times):
            plt.text(i, v + 0.1, f"{v:.2f}s", ha='center')
        
        plt.tight_layout()
        plt.savefig('mlp_comparison.png')
        print("\n结果已保存为'mlp_comparison.png'")
    except Exception as e:
        print(f"\n可视化过程中出现错误: {e}")
        print("跳过可视化步骤，请检查matplotlib版本兼容性。")

def print_results_summary(results):
    """
    以表格形式打印结果摘要
    """
    print("\n" + "="*60)
    print("性能比较摘要")
    print("="*60)
    print(f"{'模型':<20}{'准确率':<15}{'训练时间(秒)':<15}")
    print("-"*60)
    
    for model_name, model_results in results.items():
        accuracy = model_results['accuracy']
        train_time = model_results['train_time']
        print(f"{model_name:<20}{accuracy:<15.4f}{train_time:<15.2f}")
    
    print("="*60)
    
    # 计算性能差异
    custom_acc = results['custom_mlp']['accuracy']
    sklearn_acc = results['sklearn_mlp']['accuracy']
    custom_time = results['custom_mlp']['train_time']
    sklearn_time = results['sklearn_mlp']['train_time']
    
    acc_diff = abs(custom_acc - sklearn_acc)
    time_ratio = max(custom_time, sklearn_time) / min(custom_time, sklearn_time)
    
    print(f"准确率差异: {acc_diff:.4f}")
    print(f"训练时间比率: {time_ratio:.2f}倍")
    
    if custom_acc > sklearn_acc:
        print("自定义MLP模型在准确率上更优。")
    elif sklearn_acc > custom_acc:
        print("Sklearn MLP模型在准确率上更优。")
    else:
        print("两个模型在准确率上表现相当。")
    
    if custom_time < sklearn_time:
        print("自定义MLP模型在训练速度上更快。")
    elif sklearn_time < custom_time:
        print("Sklearn MLP模型在训练速度上更快。")
    else:
        print("两个模型在训练速度上表现相当。")

def main():
    """
    主函数
    """
    print("加载MNIST数据集...")
    X_train, y_train, y_train_one_hot, X_test, y_test, y_test_one_hot = load_mnist_data()
    
    print("\n开始比较两个MLP实现的性能...")
    results = compare_models(X_train, y_train, y_train_one_hot, X_test, y_test, y_test_one_hot, epochs=10)
    
    # 打印结果摘要
    print_results_summary(results)
    
    # 可视化结果
    try:
        visualize_results(results)
    except Exception as e:
        print(f"可视化结果时出错: {e}")

if __name__ == "__main__":
    main()
