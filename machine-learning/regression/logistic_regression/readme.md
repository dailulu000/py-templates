# 逻辑回归算法实现项目

本项目是机器学习算法系列中的逻辑回归（Logistic Regression）算法的完整代码实现，包含了基础理论和多种应用场景的实践案例。

## 项目目录结构

```
logistic_regression/
├── NonLinearBoundary.py          # 非线性边界分类示例（微芯片测试数据）
├── __pycache__/                  # Python编译缓存目录
│   ├── NonLinearBoundary.cpython-36.pyc
│   ├── logistic_regression.cpython-311.pyc
│   └── logistic_regression.cpython-36.pyc
├── data/                         # 数据集目录
│   ├── fashion-mnist-demo.csv    # Fashion MNIST演示数据集
│   ├── iris.csv                  # 鸢尾花数据集
│   ├── microchips-tests.csv      # 微芯片测试数据集
│   ├── mnist-demo.csv            # MNIST手写数字演示数据集
│   ├── non-linear-regression-x-y.csv  # 非线性回归数据集
│   ├── server-operational-params.csv  # 服务器运行参数数据集
│   └── world-happiness-report-2017.csv  # 世界幸福报告数据集
├── logistic_regression.py        # 逻辑回归算法核心实现
├── logistic_regression_with_linear_boundary.py  # 线性边界分类示例（鸢尾花数据）
├── mnist.py                      # MNIST手写数字识别示例
├── readme.md                     # 项目说明文档
└── utils/                        # 工具函数目录
    ├── __init__.py               # 包初始化文件
    ├── __pycache__/              # 工具函数编译缓存
    │   ├── __init__.cpython-311.pyc
    │   └── __init__.cpython-36.pyc
    ├── features/                 # 特征处理模块
    │   ├── __init__.py
    │   ├── __pycache__/
    │   ├── generate_polynomials.py  # 生成多项式特征
    │   ├── generate_sinusoids.py    # 生成正弦特征
    │   ├── normalize.py             # 数据归一化
    │   └── prepare_for_training.py  # 数据预处理主函数
    └── hypothesis/               # 假设函数模块
        ├── __init__.py
        ├── __pycache__/
        ├── sigmoid.py             # Sigmoid激活函数
        └── sigmoid_gradient.py    # Sigmoid梯度函数
```

## 核心功能

### 1. 逻辑回归算法实现

**LogisticRegression** 类是项目的核心，实现了完整的逻辑回归算法，支持：

- 多分类问题处理（一对多方法）
- 特征预处理（归一化、多项式特征、正弦特征）
- 梯度下降优化训练
- 模型预测和评估

主要方法：

- `__init__()`: 初始化模型并预处理数据
- `train()`: 使用梯度下降训练模型
- `predict()`: 使用训练好的模型进行预测
- 静态方法：`cost_function()`, `hypothesis()`, `gradient_step()` 等算法核心组件

### 2. 特征处理模块

`prepare_for_training.py` 提供了完整的数据预处理功能：

- 数据归一化
- 生成多项式特征（用于非线性边界建模）
- 生成正弦特征
- 添加偏置项

### 3. 示例应用

项目提供了三个典型的应用示例，展示逻辑回归在不同场景下的应用：

#### 鸢尾花分类（线性边界）

`logistic_regression_with_linear_boundary.py` 使用线性逻辑回归对鸢尾花数据集进行分类，展示了：

- 数据可视化
- 模型训练和成本函数监控
- 决策边界可视化
- 模型精度评估

#### 微芯片测试分类（非线性边界）

`NonLinearBoundary.py` 演示了如何使用多项式特征扩展来处理非线性分类问题，包括：

- 非线性特征变换
- 复杂决策边界的可视化
- 模型性能评估

#### MNIST手写数字识别

`mnist.py` 实现了基于逻辑回归的手写数字识别，展示了：

- 高维数据处理
- 多分类问题求解
- 模型参数可视化
- 测试集评估和结果可视化

## 技术栈

- **Python 3**：主要编程语言
- **NumPy**：数值计算和矩阵操作
- **Pandas**：数据读取和处理
- **Matplotlib**：数据可视化
- **SciPy**：优化算法（如TNC优化器）

## 使用方法

### 环境配置

确保安装了以下Python库：

```bash
pip install numpy pandas matplotlib scipy
```

### 运行示例

直接运行对应的Python文件即可查看不同示例的运行结果：

```bash
# 运行鸢尾花分类示例（线性边界）
python logistic_regression_with_linear_boundary.py

# 运行微芯片测试分类示例（非线性边界）
python NonLinearBoundary.py

# 运行MNIST手写数字识别示例
python mnist.py
```

## 项目特点

1. **理论与实践结合**：不仅实现了算法理论，还提供了多个实际应用场景
2. **模块化设计**：代码结构清晰，便于理解和扩展
3. **可视化丰富**：每个示例都包含详细的数据和结果可视化
4. **教学价值高**：适合机器学习初学者学习逻辑回归算法的原理和实现

## 数据集说明

项目使用的数据集存放在 `/data/` 目录下：

- **iris.csv**：经典的鸢尾花分类数据集，包含3个品种的鸢尾花数据
- **microchips-tests.csv**：微芯片测试数据集，用于展示非线性分类问题
- **mnist-demo.csv**：MNIST手写数字数据集的简化版本，用于数字识别示例

## 学习目标

通过学习和运行本项目，您将能够：

1. 理解逻辑回归算法的基本原理
2. 掌握多分类问题的处理方法
3. 学习特征工程在分类问题中的应用
4. 了解如何评估和可视化分类模型的性能
