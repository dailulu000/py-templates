import numpy as np
from ml_utils.features.prepare_for_training import prepare_for_training


class LinearRegression:

    def __init__(
        self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True
    ):
        """
        初始化线性回归模型

        参数:
        - data: 训练数据，形状为 [样本数, 特征数]
        - labels: 标签值，形状为 [样本数, 1]
        - polynomial_degree: 多项式特征的最高次数，默认为0（不使用多项式特征）
        - sinusoid_degree: 正弦特征的最高次数，默认为0（不使用正弦特征）
        - normalize_data: 是否对数据进行归一化处理，默认为True

        功能:
        1. 对数据进行预处理（特征工程和归一化）
        2. 保存处理后的训练数据和原始参数
        3. 初始化参数矩阵theta
        """
        # 调用特征预处理函数，得到处理后的数据、特征均值和标准差
        (data_processed, features_mean, features_deviation) = prepare_for_training(
            data, polynomial_degree, sinusoid_degree, normalize_data=True
        )

        # 保存处理后的数据和参数
        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        # 初始化参数矩阵theta，形状为 [特征数, 1]，初始值全为0
        num_features = self.data.shape[1]  # 特征数为数据矩阵的列数
        self.theta = np.zeros(
            (num_features, 1)
        )  # 初始化参数矩阵theta，形状为 [特征数, 1]，初始值全为0

    def train(self, alpha, num_iterations=500):
        """
        训练线性回归模型，执行梯度下降算法

        参数:
        - alpha: 学习率，控制梯度下降的步长
        - num_iterations: 迭代次数，默认为500次

        返回值:
        - self.theta: 训练得到的最优参数矩阵
        - cost_history: 每次迭代的损失函数值列表
        """
        # 调用梯度下降方法进行训练，并获取损失历史
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iterations):
        """
        梯度下降算法的迭代实现

        参数:
        - alpha: 学习率
        - num_iterations: 迭代次数

        返回值:
        - cost_history: 记录每次迭代的损失值的列表
        """
        # 初始化损失历史记录列表
        cost_history = []

        # 执行指定次数的迭代
        for _ in range(num_iterations):
            # 执行单步梯度更新
            self.gradient_step(alpha)
            # 计算当前的损失值并记录
            cost_history.append(self.cost_function(self.data, self.labels))

        return cost_history

    def gradient_step(self, alpha):
        """
        执行单步梯度下降更新

        参数:
        - alpha: 学习率

        功能:
        使用矩阵运算实现梯度下降的参数更新
        theta = theta - alpha * (1/m) * X^T*(X*theta - y)
        其中m为样本数量
        """
        # 获取样本数量
        num_examples = self.data.shape[0]

        # 计算预测值，使用假设函数h(X) = X*theta
        prediction = LinearRegression.hypothesis(self.data, self.theta)

        # 计算预测值与实际值的差异
        delta = prediction - self.labels

        # 保存当前的theta值
        theta = self.theta

        # 执行梯度下降更新：theta = theta - alpha*(1/m)*(X^T*delta)
        # np.dot(delta.T, self.data) 计算X^T*delta
        # .T 转置结果以匹配theta的维度并更新参数
        self.theta = theta - alpha * (1 / num_examples) * (np.dot(delta.T, self.data)).T

    def cost_function(self, data, labels):
        """
        计算损失函数（均方误差）

        参数:
        - data: 输入数据
        - labels: 实际标签值

        返回值:
        - cost: 计算得到的损失值，即 (1/2m)*sum((h(X)-y)^2)
        """
        # 获取样本数量
        num_examples = data.shape[0]

        # 计算预测值与实际值的差异
        delta = LinearRegression.hypothesis(self.data, self.theta) - labels

        # 计算均方误差: (1/2m)*sum((h(X)-y)^2)
        # np.dot(delta.T, delta) 计算向量点积，相当于sum(delta^2)
        cost = (1 / 2) * np.dot(delta.T, delta) / num_examples

        # 返回标量损失值
        return cost[0][0]

    @staticmethod
    def hypothesis(data, theta):
        """
        线性回归的假设函数 h(X) = X*theta

        参数:
        - data: 输入数据矩阵
        - theta: 参数矩阵

        返回值:
        - predictions: 预测结果向量

        注意：这是一个静态方法，可以直接通过类名调用
        """
        # 计算矩阵乘法 X*theta，得到预测值
        predictions = np.dot(data, theta)
        return predictions

    def get_cost(self, data, labels):
        """
        计算给定数据和标签的损失值

        参数:
        - data: 输入数据
        - labels: 实际标签值

        返回值:
        - cost: 计算得到的损失值

        功能:
        先对输入数据进行与训练时相同的预处理，然后计算损失
        """
        # 对输入数据进行预处理（使用与训练时相同的参数）
        data_processed = prepare_for_training(
            data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data
        )[0]

        # 调用损失函数计算损失值
        return self.cost_function(data_processed, labels)

    def predict(self, data):
        """
        使用训练好的模型对新数据进行预测

        参数:
        - data: 待预测的输入数据

        返回值:
        - predictions: 预测结果向量

        功能:
        1. 对输入数据进行预处理（特征工程和归一化）
        2. 使用训练好的参数theta计算预测值
        """
        # 对输入数据进行预处理（使用与训练时相同的参数）
        data_processed = prepare_for_training(
            data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data
        )[0]

        # 使用假设函数计算预测值
        predictions = LinearRegression.hypothesis(data_processed, self.theta)

        return predictions
