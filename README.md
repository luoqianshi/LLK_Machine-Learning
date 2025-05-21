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
├── results/              # 实验结果目录（模型性能对比）
└── data/                 # 数据目录（MNIST数据集）
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
  - 准确率（Accuracy）
  - 精确率（Precision）
  - 召回率（Recall）
  - F1分数（F1 Score）
  - 训练时间
  - 推理时间
  - 混淆矩阵
- 提供可视化比较结果：
  - 性能指标对比图
  - 训练时间对比图
  - 推理时间对比图
  - 混淆矩阵可视化
- 生成详细的性能报告和评估指标

## 环境要求

- Python 3.6+
- NumPy
- scikit-learn
- PyTorch
- torchvision
- matplotlib
- seaborn

## 安装依赖

```bash
pip install numpy scikit-learn torch torchvision matplotlib seaborn
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

项目会自动生成性能比较图表和报告，包括：
- 模型性能指标对比（准确率、精确率、召回率、F1分数）
- 训练时间对比
- 推理时间对比
- 混淆矩阵可视化
- 详细的性能指标报告（保存为metrics.txt）

所有结果将保存在results目录下，按时间戳分类存储。

## 注意事项

1. 首次运行时会自动下载MNIST数据集到data目录
2. 为了加快训练速度，比较脚本默认使用5000个样本进行训练
3. 可以通过修改参数来调整模型结构和训练过程
4. 数据集使用标准化处理，均值为0.1307，标准差为0.3081

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 许可证

MIT License 