根据给定的MLP结构和MBGD算法，基于MNIST数据集的训练数学原理如下：

---

### **1. 前向传播**

#### **输入层 → hidden1**  
- 输入数据：（N为batch_size，输入展平为784维）
  $$
  X \in \mathbb{R}^{N \times 784}
  $$

- 权重：
  $$
  W_1 \in \mathbb{R}^{784 \times 128}
  $$

- 偏置：
  $$
  b_1 \in \mathbb{R}^{128}
  $$

- 线性变换：

$$
Z_1 = X W_1 + b_1
$$

- 激活函数（ReLU）：

$$
A_1 = \text{ReLU}(Z_1) = \max(0, Z_1)
$$

#### **hidden1 → hidden2**  
- 权重：

$$
W_2 \in \mathbb{R}^{128 \times 64}
$$

- 偏置：

$$
b_2 \in \mathbb{R}^{64}
$$

- 线性变换：

$$
Z_2 = A_1 W_2 + b_2
$$

- 激活函数（ReLU）：

$$
A_2 = \text{ReLU}(Z_2)
$$

#### **hidden2 → 输出层**  
- 权重：

$$
W_3 \in \mathbb{R}^{64 \times 10}
$$

- 偏置：

$$
b_3 \in \mathbb{R}^{10}
$$

- 线性变换：

$$
Z_3 = A_2 W_3 + b_3
$$

- 激活函数（Softmax）：  
  $$
  Y_{\text{hat}} = \text{Softmax}(Z_3) = \frac{e^{Z_3}}{\sum_{k=1}^{10} e^{Z_3^{(k)}}}
  $$

---

### **2. 损失函数（交叉熵损失）**  
- 真实标签：（One-Hot编码）  

$$
Y \in \mathbb{R}^{N \times 10}
$$

- 损失值： 

$$
L = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^{10} Y^{(i,k)} \log\left(Y_{\text{hat}}^{(i,k)}\right)
$$

---

### **3. 反向传播（梯度计算）**
#### **输出层梯度**  
- 损失对输出层输入的梯度：

$$
\frac{\partial L}{\partial Z_3} = Y_{\text{hat}} - Y \quad \in \mathbb{R}^{N \times 10}
$$

- 参数梯度： 

$$
\frac{\partial L}{\partial W_3} = \frac{1}{N} A_2^\top \frac{\partial L}{\partial Z_3}, \quad \frac{\partial L}{\partial b_3} = \frac{1}{N} \sum_{i=1}^N \frac{\partial L}{\partial Z_3^{(i)}}
$$

#### **hidden2梯度**  
- 损失对hidden2输出的梯度：

$$
\frac{\partial L}{\partial A_2} = \frac{\partial L}{\partial Z_3} W_3^\top \quad \in \mathbb{R}^{N \times 64}
$$

- ReLU导数：

$$
\frac{\partial L}{\partial Z_2} = \frac{\partial L}{\partial A_2} \odot \mathbb{I}(Z_2 > 0)
（逐元素乘，\mathbb{I}为指示函数）
$$

- 参数梯度：

$$
\frac{\partial L}{\partial W_2} = \frac{1}{N} A_1^\top \frac{\partial L}{\partial Z_2}, \quad \frac{\partial L}{\partial b_2} = \frac{1}{N} \sum_{i=1}^N \frac{\partial L}{\partial Z_2^{(i)}}
$$

#### **hidden1梯度**  
- 损失对hidden1输出的梯度：

$$
\frac{\partial L}{\partial A_1} = \frac{\partial L}{\partial Z_2} W_2^\top \quad \in \mathbb{R}^{N \times 128}
$$

- ReLU导数：

$$
\frac{\partial L}{\partial Z_1} = \frac{\partial L}{\partial A_1} \odot \mathbb{I}(Z_1 > 0)
$$

- 参数梯度：

$$
\frac{\partial L}{\partial W_1} = \frac{1}{N} X^\top \frac{\partial L}{\partial Z_1}, \quad \frac{\partial L}{\partial b_1} = \frac{1}{N} \sum_{i=1}^N \frac{\partial L}{\partial Z_1^{(i)}}
$$

---

### **4. MBGD参数更新**
更新规则：  
$$
\alpha为学习率
$$

$$
\begin{aligned}
W_1 &\leftarrow W_1 - \alpha \frac{\partial L}{\partial W_1}, \quad b_1 \leftarrow b_1 - \alpha \frac{\partial L}{\partial b_1}, \\
W_2 &\leftarrow W_2 - \alpha \frac{\partial L}{\partial W_2}, \quad b_2 \leftarrow b_2 - \alpha \frac{\partial L}{\partial b_2}, \\
W_3 &\leftarrow W_3 - \alpha \frac{\partial L}{\partial W_3}, \quad b_3 \leftarrow b_3 - \alpha \frac{\partial L}{\partial b_3}.
\end{aligned}
$$

---

### **5. 流程总结**
1. 从MNIST中采样一个batch（N个样本）；  
2. 前向传播计算预测值；

$$
Y_{\text{hat}}
$$

1. 计算交叉熵损失；  
2. 反向传播计算各参数梯度；  
3. 用MBGD更新参数；  
4. 重复直到收敛。

---

**关键点**  
- **MBGD特点**：使用小批量样本计算梯度，平衡了SGD（噪声大）和BGD（计算慢）的优缺点。  
- **激活函数选择**：隐藏层使用ReLU加速收敛，输出层使用Softmax生成概率分布。  
- **梯度公式**：交叉熵损失与Softmax结合时，梯度

$$
\frac{\partial L}{\partial Z_3}
$$

- 形式简化为：

$$
Y_{\text{hat}} - Y
$$


---

