"""
对比测试文件：比较自定义MLP和基于sklearn的MLP实现
"""
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')  # 设置后端为Agg，避免GUI相关问题
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
import time
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from MLP_Numpy import MLP
from MLP_sklearn import MLP_sklearn

# 设置随机种子以确保结果可复现
np.random.seed(42)
torch.manual_seed(42)

def create_result_dirs():
    """
    创建结果保存目录
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("results", current_time)
    numpy_dir = os.path.join(base_dir, "mlp_numpy")
    sklearn_dir = os.path.join(base_dir, "mlp_sklearn")
    
    # 创建目录
    os.makedirs(numpy_dir, exist_ok=True)
    os.makedirs(sklearn_dir, exist_ok=True)
    
    return current_time, base_dir, numpy_dir, sklearn_dir

def load_mnist_data():
    """
    加载MNIST数据集并转换为NumPy数组
    - 其中0.1307和0.3081是MNIST数据集的均值(mean)和标准差(standard deviation)
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

def evaluate_model(model, X_test, y_test, y_test_one_hot, model_type):
    """
    评估模型性能，计算多个指标
    """
    # 记录推理开始时间
    inference_start = time.time()
    
    # 获取预测结果
    if model_type == "numpy":
        predictions = model.predict(X_test)
    else:
        predictions = model.predict(X_test)
    
    # 记录推理结束时间
    inference_time = time.time() - inference_start
    
    # 计算各项指标
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    conf_matrix = confusion_matrix(y_test, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'inference_time': inference_time
    }

def save_results(results, save_dir, model_name):
    """
    保存评估结果和可视化图表
    """
    # 保存数值结果
    metrics = {
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1_score': results['f1_score'],
        'inference_time': results['inference_time']
    }
    
    # 保存指标到文本文件
    with open(os.path.join(save_dir, 'metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{model_name} 评估结果:\n")
        f.write("="*50 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    try:
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.set(font_scale=1.2)  # 设置seaborn的字体大小
        
        # 设置标题和标签
        title = f'{model_name} Confusion Matrix'  # 混淆矩阵
        xlabel = 'Predicted Label'  # 预测标签
        ylabel = 'True Label'  # 真实标签
        
        sns.heatmap(results['confusion_matrix'], 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   cbar_kws={'label': 'Sample Count'}) # 样本数量
        plt.title(title, pad=20)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), 
                   dpi=300, 
                   bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"保存混淆矩阵时出错: {e}")

def compare_models(X_train, y_train, y_train_one_hot, X_test, y_test, y_test_one_hot, epochs=10):
    """
    比较自定义MLP和sklearn MLP的性能
    """
    # 创建结果保存目录
    current_time, base_dir, numpy_dir, sklearn_dir = create_result_dirs()
    
    # 使用完整的训练集
    X_train_full = X_train
    y_train_full = y_train
    y_train_one_hot_full = y_train_one_hot
    
    # 共同参数
    layer_sizes = [784, 128, 64, 10]
    activation_functions = ['relu', 'relu', 'softmax']
    learning_rate = 0.001  # 保持较小的学习率以确保稳定性
    
    # 结果字典
    results = {
        'custom_mlp': {'train_time': 0, 'metrics': None, 'history': None},
        'sklearn_mlp': {'train_time': 0, 'metrics': None, 'history': None}
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
    
    # 训练模型并获取历史记录
    history = custom_mlp.train(X_train_full, y_train_one_hot_full, epochs=epochs, batch_size=256, verbose=True)
    results['custom_mlp']['history'] = history
    
    # 记录训练时间
    results['custom_mlp']['train_time'] = time.time() - start_time
    
    # 评估模型
    numpy_metrics = evaluate_model(custom_mlp, X_test, y_test, y_test_one_hot, "numpy")
    results['custom_mlp']['metrics'] = numpy_metrics
    
    # 保存结果
    save_results(numpy_metrics, numpy_dir, "NumPy MLP")
    
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
    
    # 训练模型并获取历史记录
    history = sklearn_mlp.train(X_train_full, y_train_full, epochs=epochs, batch_size=256, verbose=True)
    results['sklearn_mlp']['history'] = history
    
    # 记录训练时间
    results['sklearn_mlp']['train_time'] = time.time() - start_time
    
    # 评估模型
    sklearn_metrics = evaluate_model(sklearn_mlp, X_test, y_test, y_test_one_hot, "sklearn")
    results['sklearn_mlp']['metrics'] = sklearn_metrics
    
    # 保存结果
    save_results(sklearn_metrics, sklearn_dir, "Sklearn MLP")
    
    return results, current_time

def visualize_results(results, current_time):
    """
    可视化两个模型的性能比较
    """
    try:
        base_dir = os.path.join("results", current_time)
        
        # 1. 准确率比较
        models = list(results.keys())
        accuracies = [results[model]['metrics']['accuracy'] for model in models]
        precisions = [results[model]['metrics']['precision'] for model in models]
        recalls = [results[model]['metrics']['recall'] for model in models]
        f1_scores = [results[model]['metrics']['f1_score'] for model in models]
        train_times = [results[model]['train_time'] for model in models]
        inference_times = [results[model]['metrics']['inference_time'] for model in models]
        
        # 创建性能指标对比图
        plt.figure(figsize=(15, 10))
        
        # 设置标签
        accuracy_label = 'Accuracy' # 准确率
        precision_label = 'Precision' # 精确率
        recall_label = 'Recall' # 召回率
        f1_label = 'F1 Score' # F1分数
        time_label = 'Time (s)' # 时间(秒)
        
        # 性能指标对比（柱状图）
        plt.subplot(2, 2, 1)
        x = np.arange(len(models))
        width = 0.2
        
        # 绘制柱状图
        plt.bar(x - width*1.5, accuracies, width, label=accuracy_label, color='#2ecc71')
        plt.bar(x - width*0.5, precisions, width, label=precision_label, color='#3498db')
        plt.bar(x + width*0.5, recalls, width, label=recall_label, color='#e74c3c')
        plt.bar(x + width*1.5, f1_scores, width, label=f1_label, color='#f1c40f')
        
        # 添加数值标签
        for i, v in enumerate(accuracies):
            plt.text(i - width*1.5, v, f'{v:.4f}', ha='center', va='bottom')
        for i, v in enumerate(precisions):
            plt.text(i - width*0.5, v, f'{v:.4f}', ha='center', va='bottom')
        for i, v in enumerate(recalls):
            plt.text(i + width*0.5, v, f'{v:.4f}', ha='center', va='bottom')
        for i, v in enumerate(f1_scores):
            plt.text(i + width*1.5, v, f'{v:.4f}', ha='center', va='bottom')
        
        plt.title('Model Performance Metrics', pad=20) # 模型性能指标对比
        plt.xticks(x, models)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 训练时间对比
        plt.subplot(2, 2, 2)
        plt.bar(models, train_times, color=['#2ecc71', '#3498db'])
        plt.title('Training Time Comparison', pad=20) # 训练时间比较
        plt.ylabel(time_label)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 推理时间对比
        plt.subplot(2, 2, 3)
        plt.bar(models, inference_times, color=['#2ecc71', '#3498db'])
        plt.title('Inference Time Comparison', pad=20) # 推理时间比较
        plt.ylabel(time_label)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, 'performance_comparison.png'), 
                   dpi=300, 
                   bbox_inches='tight')
        plt.close()
        
        # 绘制训练历史曲线
        plt.figure(figsize=(15, 5))
        
        # 损失函数曲线
        plt.subplot(1, 2, 1)
        for model in models:
            history = results[model]['history']
            epochs = range(0, len(history['loss']) * 2, 2)  # 每2个epoch记录一次
            plt.plot(epochs, history['loss'], 
                    label=f'{model} Loss',
                    linewidth=2)
        plt.title('Training Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 准确率曲线
        plt.subplot(1, 2, 2)
        for model in models:
            history = results[model]['history']
            epochs = range(0, len(history['accuracy']) * 2, 2)  # 每2个epoch记录一次
            plt.plot(epochs, history['accuracy'], 
                    label=f'{model} Accuracy',
                    linewidth=2)
        plt.title('Training Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, 'training_history.png'), 
                   dpi=300, 
                   bbox_inches='tight')
        plt.close()
        
        print(f"\n结果已保存到目录: {base_dir}")
        
    except Exception as e:
        print(f"\n可视化过程中出现错误: {e}")
        print("跳过可视化步骤，请检查matplotlib版本兼容性。")

def print_results_summary(results):
    """
    以表格形式打印结果摘要
    """
    print("\n" + "="*80)
    print("性能比较摘要")
    print("="*80)
    print(f"{'模型':<20}{'准确率':<10}{'精确率':<10}{'召回率':<10}{'F1分数':<10}{'训练时间(秒)':<15}{'推理时间(秒)':<15}")
    print("-"*80)
    
    for model_name, model_results in results.items():
        metrics = model_results['metrics']
        print(f"{model_name:<20}"
              f"{metrics['accuracy']:<10.4f}"
              f"{metrics['precision']:<10.4f}"
              f"{metrics['recall']:<10.4f}"
              f"{metrics['f1_score']:<10.4f}"
              f"{model_results['train_time']:<15.2f}"
              f"{metrics['inference_time']:<15.2f}")
    
    print("="*80)

def plot_confusion_matrix(y_true, y_pred, save_dir):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def plot_learning_curves(history, save_dir):
    """绘制学习曲线"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curves.png'))
    plt.close()

def plot_performance_comparison(results, save_dir):
    """绘制性能对比图"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    models = ['NumPy MLP', 'Sklearn MLP']
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    for i, model in enumerate(models):
        values = [results[model][metric] for metric in metrics]
        plt.bar(x + i*width, values, width, label=model)
    
    plt.title('Model Performance Comparison')
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.xticks(x + width/2, metrics)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_comparison.png'))
    plt.close()

def main():
    """
    主函数
    """
    print("加载MNIST数据集...")
    X_train, y_train, y_train_one_hot, X_test, y_test, y_test_one_hot = load_mnist_data()
    
    print("\n开始比较两个MLP实现的性能...")
    results, current_time = compare_models(X_train, y_train, y_train_one_hot, X_test, y_test, y_test_one_hot, epochs=10)
    
    # 打印结果摘要
    print_results_summary(results)
    
    # 可视化结果
    try:
        visualize_results(results, current_time)
    except Exception as e:
        print(f"可视化结果时出错: {e}")

if __name__ == "__main__":
    main()