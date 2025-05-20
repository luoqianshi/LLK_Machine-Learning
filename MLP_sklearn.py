"""
基于scikit-learn实现的多层感知机，参数配置尽可能与自定义MLP相似
"""
import numpy as np
from sklearn.neural_network import MLPClassifier

class MLP_sklearn:
    def __init__(self, layer_sizes, activation_functions, learning_rate=0.01):
        """
        初始化基于sklearn的多层感知机
        
        参数:
        layer_sizes: 一个列表，包含每层的神经元数量，例如[784, 128, 64, 10]表示输入层784个神经元，两个隐藏层分别有128和64个神经元，输出层10个神经元
        activation_functions: 一个列表，包含每层的激活函数，例如['relu', 'relu', 'softmax']
        learning_rate: 学习率
        """
        self.layer_sizes = layer_sizes
        self.hidden_layer_sizes = layer_sizes[1:-1]  # sklearn只需要隐藏层大小
        self.activation_functions = activation_functions
        self.learning_rate = learning_rate
        
        # sklearn MLP 只支持所有隐藏层使用相同的激活函数
        # 我们取第一个激活函数作为所有隐藏层的激活函数
        self.activation = activation_functions[0]
        
        # 如果激活函数是softmax，在sklearn中使用identity，因为输出层会自动处理
        if self.activation == 'softmax':
            self.activation = 'relu'  # 默认使用relu
        
        # 初始化sklearn MLP
        self.mlp = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver='sgd',  # 使用随机梯度下降，类似于我们的实现
            alpha=0.0001,  # L2正则化参数，默认值
            batch_size=32, # 默认批大小
            learning_rate_init=learning_rate,
            max_iter=1,    # 每次调用fit时只训练一轮，让我们可以控制总训练轮数
            warm_start=True,  # 允许继续训练
            random_state=42  # 固定随机种子
        )
        
        self.classes_ = None  # 用于存储类别
    
    def forward(self, X):
        """
        前向传播
        
        参数:
        X: 输入数据，形状为(batch_size, input_size)
        
        返回:
        output: 网络的输出
        """
        # 确保模型已经至少训练过一次
        if not hasattr(self.mlp, 'classes_'):
            raise ValueError("模型尚未训练，无法执行前向传播")
        
        # 使用sklearn的predict_proba来获取概率输出
        return self.mlp.predict_proba(X)
    
    def train(self, X, y, epochs=100, batch_size=32, verbose=True):
        """
        训练网络
        
        参数:
        X: 输入数据，形状为(num_samples, input_size)
        y: 目标标签，形状为(num_samples, output_size)
        epochs: 训练轮数
        batch_size: 批大小
        verbose: 是否打印训练进度
        """
        # 如果y是one-hot编码的，转换为类别索引
        if len(y.shape) > 1 and y.shape[1] > 1:
            y = np.argmax(y, axis=1)
        
        # 更新批大小
        self.mlp.batch_size = batch_size
        
        # 保存类别信息
        self.classes_ = np.unique(y)
        
        # 训练模型
        for epoch in range(epochs):
            self.mlp.fit(X, y)
            
            if verbose and (epoch + 1) % 10 == 0:
                # 计算并打印损失
                y_pred = self.mlp.predict_proba(X)
                # 转换为one-hot编码以计算损失
                y_one_hot = np.zeros((y.shape[0], len(self.classes_)))
                y_one_hot[np.arange(y.shape[0]), y] = 1
                loss = self.calculate_loss(y_pred, y_one_hot)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
    
    def calculate_loss(self, y_pred, y_true):
        """
        计算损失函数
        
        参数:
        y_pred: 预测值，形状为(batch_size, output_size)
        y_true: 真实值，形状为(batch_size, output_size)
        
        返回:
        loss: 损失值
        """
        # 交叉熵损失
        epsilon = 1e-15  # 避免log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def predict(self, X):
        """
        使用训练好的网络进行预测
        
        参数:
        X: 输入数据，形状为(num_samples, input_size)
        
        返回:
        predictions: 预测结果，对于分类问题返回类别索引
        """
        return self.mlp.predict(X)
    
    def evaluate(self, X, y):
        """
        评估模型性能
        
        参数:
        X: 输入数据，形状为(num_samples, input_size)
        y: 目标标签，形状为(num_samples, output_size) 或 (num_samples,)
        
        返回:
        accuracy: 准确率
        """
        # 如果y是one-hot编码的，转换为类别索引
        if len(y.shape) > 1 and y.shape[1] > 1:
            y = np.argmax(y, axis=1)
        
        # 获取预测结果
        predictions = self.predict(X)
        
        # 计算准确率
        accuracy = np.mean(predictions == y)
        return accuracy
