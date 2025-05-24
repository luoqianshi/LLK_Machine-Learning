import numpy as np

class MLP:
    def __init__(self, layer_sizes, activation_functions, learning_rate=0.01, init_method='he'):
        """
        初始化多层感知机
        
        参数:
        layer_sizes: 一个列表，包含每层的神经元数量，例如[784, 128, 64, 10]表示输入层784个神经元，两个隐藏层分别有128和64个神经元，输出层10个神经元
        activation_functions: 一个列表，包含每层的激活函数，例如['relu', 'relu', 'softmax']
        learning_rate: 学习率
        init_method: 初始化方法，可选 'he' 或 'random'
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1  # 不包括输入层
        self.activation_functions = activation_functions
        self.learning_rate = learning_rate
        self.init_method = init_method
        
        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers):
            if self.init_method == 'he':
                # 使用He初始化权重（由何凯明提出）
                weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            else:  # random
                # 使用随机初始化
                weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            
            bias = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(weight)
            self.biases.append(bias)
        
        # 用于存储前向传播中的中间结果，用于反向传播
        self.z_values = []  # 线性组合的结果
        self.activations = []  # 激活函数的输出
    
    def forward(self, X):
        """
        前向传播
        
        参数:
        X: 输入数据，形状为(batch_size, input_size)
        
        返回:
        output: 网络的输出
        """
        self.z_values = []
        self.activations = [X]  # 输入作为第一层的激活值
        
        for i in range(self.num_layers):
            # 线性组合: z = a * W + b
            z = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            # 应用激活函数
            activation = self._apply_activation(z, self.activation_functions[i])
            self.activations.append(activation)
        
        return self.activations[-1]
    
    def _apply_activation(self, z, activation_name):
        """
        应用激活函数
        
        参数:
        z: 线性组合的结果
        activation_name: 激活函数的名称
        
        返回:
        activation: 激活函数的输出
        """
        if activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif activation_name == 'relu':
            return np.maximum(0, z)
        elif activation_name == 'tanh':
            return np.tanh(z)
        elif activation_name == 'softmax':
            # 为了数值稳定性，减去最大值
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        else:
            raise ValueError(f"不支持的激活函数: {activation_name}")
    
    def _apply_activation_derivative(self, z, activation_name):
        """
        计算激活函数的导数
        
        参数:
        z: 线性组合的结果
        activation_name: 激活函数的名称
        
        返回:
        derivative: 激活函数的导数
        """
        if activation_name == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-z))
            return sigmoid * (1 - sigmoid)
        elif activation_name == 'relu':
            return (z > 0).astype(float)
        elif activation_name == 'tanh':
            return 1 - np.tanh(z)**2
        elif activation_name == 'softmax':
            # softmax的导数在反向传播中会特殊处理
            return 1
        else:
            raise ValueError(f"不支持的激活函数: {activation_name}")
    
    def backward(self, X, y):
        """
        反向传播
        
        参数:
        X: 输入数据，形状为(batch_size, input_size)
        y: 目标标签，形状为(batch_size, output_size)
        
        返回:
        gradients: 包含所有权重和偏置的梯度
        """
        batch_size = X.shape[0]
        
        # 初始化梯度
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]
        
        # 对于softmax+交叉熵，输出层的误差计算可以简化（特殊！）
        output_error = self.activations[-1] - y
        
        # 从最后一层开始，反向计算每一层的误差和梯度
        for l in range(self.num_layers - 1, -1, -1):
            if l == self.num_layers - 1:
                # 输出层的误差
                delta = output_error
            else:
                # 隐藏层的误差 = (下一层的误差 dot 下一层的权重转置) * 激活函数的导数
                delta = np.dot(delta, self.weights[l+1].T) * self._apply_activation_derivative(self.z_values[l], self.activation_functions[l])
            
            # 计算权重和偏置的梯度
            weight_gradients[l] = np.dot(self.activations[l].T, delta) / batch_size
            bias_gradients[l] = np.sum(delta, axis=0, keepdims=True) / batch_size
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients, bias_gradients):
        """
        更新网络参数
        
        参数:
        weight_gradients: 权重的梯度
        bias_gradients: 偏置的梯度
        """
        for l in range(self.num_layers):
            self.weights[l] -= self.learning_rate * weight_gradients[l]
            self.biases[l] -= self.learning_rate * bias_gradients[l]
    
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
        # clip为[epsilon, 1-epsilon]之间, 避免log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def train(self, X, y, epochs=100, batch_size=32, verbose=True):
        """
        训练网络
        
        参数:
        X: 输入数据，形状为(num_samples, input_size)
        y: 目标标签，形状为(num_samples, output_size)
        epochs: 训练轮数
        batch_size: 批大小
        verbose: 是否打印训练进度
        
        返回:
        history: 包含训练过程中的损失和准确率的字典
        """
        num_samples = X.shape[0]
        # ceil 向上取整; 计算每个epoch需要迭代的次数
        iterations = int(np.ceil(num_samples / batch_size))
        
        # 初始化历史记录
        history = {
            'loss': [],
            'accuracy': []
        }
        
        # 根据epochs大小决定打印频率
        print_freq = 1 if epochs <= 100 else 10
        
        for epoch in range(epochs):
            # 随机打乱数据
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            for i in range(iterations):
                # 获取当前批次的数据
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # 前向传播
                y_pred = self.forward(X_batch)
                
                # 计算损失
                batch_loss = self.calculate_loss(y_pred, y_batch)
                epoch_loss += batch_loss * (end_idx - start_idx)
                
                # 反向传播
                weight_gradients, bias_gradients = self.backward(X_batch, y_batch)
                
                # 更新参数
                self.update_parameters(weight_gradients, bias_gradients)
            
            epoch_loss /= num_samples
            
            # 计算准确率
            predictions = self.predict(X)
            if len(y.shape) > 1 and y.shape[1] > 1:
                y_labels = np.argmax(y, axis=1)
            else:
                y_labels = y
            accuracy = np.mean(predictions == y_labels)
            
            # 记录历史
            history['loss'].append(epoch_loss)
            history['accuracy'].append(accuracy)
            
            if verbose and (epoch + 1) % print_freq == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.8f}, Accuracy: {accuracy:.8f}")
        
        return history
    
    def predict(self, X):
        """
        使用训练好的网络进行预测
        
        参数:
        X: 输入数据，形状为(num_samples, input_size)
        
        返回:
        predictions: 预测结果，对于分类问题返回类别索引
        """
        # 前向传播
        y_pred = self.forward(X)
        
        # 对于分类问题，返回概率最高的类别索引
        return np.argmax(y_pred, axis=1)
    
    def evaluate(self, X, y):
        """
        评估模型性能
        
        参数:
        X: 输入数据，形状为(num_samples, input_size)
        y: 目标标签，形状为(num_samples, output_size) 或 (num_samples,)
        
        返回:
        accuracy: 准确率
        """
        predictions = self.predict(X)
        
        # 如果y是one-hot编码的，转换为类别索引
        if len(y.shape) > 1 and y.shape[1] > 1:
            y = np.argmax(y, axis=1)
        
        # 计算准确率
        accuracy = np.mean(predictions == y)
        return accuracy