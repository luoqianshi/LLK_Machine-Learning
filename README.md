# 多层感知机（MLP）实现与比较

这个项目实现了两种多层感知机（MLP）模型，并对其性能进行了比较：
1. 使用NumPy从头实现的自定义MLP
2. 基于scikit-learn的MLP实现

## 项目结构

```
.
├── MLP_Numpy.py          # 使用NumPy实现的自定义MLP
├── MLP_sklearn.py        # 基于scikit-learn的MLP实现
├── compare_mlp.py        # 模型比较和性能评估
├── test.py              # 测试文件
└── data/                # 数据目录（MNIST数据集）
```

## 功能特点

### 自定义MLP实现（MLP_Numpy.py）
- 使用NumPy从头实现多层感知机
- 支持多种激活函数：sigmoid、ReLU、tanh、softmax
- 实现了完整的前向传播和反向传播算法
- 支持批量训练和随机梯度下降
- 使用He初始化方法初始化权重
- 实现了交叉熵损失函数

### Sklearn MLP实现（MLP_sklearn.py）
- 基于scikit-learn的MLPClassifier实现
- 保持与自定义MLP相似的接口和参数配置
- 支持相同的激活函数和训练参数

### 模型比较（compare_mlp.py）
- 在MNIST数据集上进行模型性能比较
- 比较指标包括：
  - 模型准确率
  - 训练时间
  - 损失函数收敛情况
- 提供可视化比较结果
- 生成详细的性能报告

## 环境要求

- Python 3.6+
- NumPy
- scikit-learn
- PyTorch
- torchvision
- matplotlib

## 安装依赖

```bash
pip install numpy scikit-learn torch torchvision matplotlib
```

## 使用方法

1. 运行模型比较：
```bash
python compare_mlp.py
```

2. 使用自定义MLP：
```python
from MLP_Numpy import MLP

# 创建模型
model = MLP(
    layer_sizes=[784, 128, 64, 10],
    activation_functions=['relu', 'relu', 'softmax'],
    learning_rate=0.01
)

# 训练模型
model.train(X_train, y_train, epochs=100, batch_size=32)

# 预测
predictions = model.predict(X_test)
```

3. 使用Sklearn MLP：
```python
from MLP_sklearn import MLP_sklearn

# 创建模型
model = MLP_sklearn(
    layer_sizes=[784, 128, 64, 10],
    activation_functions=['relu', 'relu', 'softmax'],
    learning_rate=0.01
)

# 训练模型
model.train(X_train, y_train, epochs=100, batch_size=32)

# 预测
predictions = model.predict(X_test)
```

## 性能比较

项目会自动生成性能比较图表（mlp_comparison.png），包括：
- 模型准确率对比
- 训练时间对比
- 详细的性能指标报告

## 注意事项

1. 首次运行时会自动下载MNIST数据集到data目录
2. 为了加快训练速度，比较脚本默认使用5000个样本进行训练
3. 可以通过修改参数来调整模型结构和训练过程

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 许可证

MIT License 