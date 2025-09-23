import numpy as np  # 导入NumPy库，用于科学计算和数组操作
import pandas as pd  # 导入Pandas库，用于数据处理和分析
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于数据可视化

from linear_regression import LinearRegression  # 导入自定义的LinearRegression类

# 读取CSV格式的世界幸福报告2017年数据
data = pd.read_csv("data/world-happiness-report-2017.csv")

# 划分训练数据和测试数据，训练数据占80%
train_data = data.sample(frac=0.8)  # 随机选择80%的数据作为训练集
test_data = data.drop(train_data.index)  # 删除训练集数据，剩余的20%作为测试集

# 定义输入特征和输出目标的列名
input_param_name = "Economy..GDP.per.Capita."  # 输入特征：人均GDP
output_param_name = "Happiness.Score"  # 输出目标：幸福分数

# 准备训练数据，提取特征和标签并转换为NumPy数组
x_train = train_data[[input_param_name]].values  # 训练特征，双括号保持二维数组结构
y_train = train_data[[output_param_name]].values  # 训练标签，双括号保持二维数组结构

# 准备测试数据
x_test = test_data[input_param_name].values  # 测试特征
y_test = test_data[output_param_name].values  # 测试标签

# 绘制训练数据和测试数据的散点图
plt.scatter(x_train, y_train, label="Train data")  # 绘制训练数据点
plt.scatter(x_test, y_test, label="test data")  # 绘制测试数据点
plt.xlabel(input_param_name)  # 设置x轴标签
plt.ylabel(output_param_name)  # 设置y轴标签
plt.title("Happy")  # 设置图表标题
plt.legend()  # 显示图例
plt.show()  # 显示图表

# 设置梯度下降的超参数
num_iterations = 500  # 迭代次数
learning_rate = 0.01  # 学习率，控制参数更新步长

# 创建线性回归模型实例并进行训练
linear_regression = LinearRegression(x_train, y_train)  # 初始化模型
(theta, cost_history) = linear_regression.train(
    learning_rate, num_iterations
)  # 训练模型，返回参数和损失历史

# 打印训练开始时和结束时的损失值
print("开始时的损失：", cost_history[0])  # 第一次迭代的损失
print("训练后的损失：", cost_history[-1])  # 最后一次迭代的损失

# 绘制损失函数随迭代次数变化的曲线
plt.plot(range(num_iterations), cost_history)  # x轴：迭代次数，y轴：损失值
plt.xlabel("Iter")  # 设置x轴标签为迭代次数
plt.ylabel("cost")  # 设置y轴标签为损失值
plt.title("GD")  # 设置图表标题为梯度下降(Gradient Descent)
plt.show()  # 显示图表

# 使用训练好的模型进行预测
predictions_num = 100  # 生成100个预测点
x_predictions = np.linspace(x_train.min(), x_train.max(), predictions_num).reshape(
    predictions_num, 1
)  # 在训练数据范围内生成均匀分布的预测点
y_predictions = linear_regression.predict(x_predictions)  # 对预测点进行预测

# 绘制训练数据、测试数据和预测曲线
plt.scatter(x_train, y_train, label="Train data")  # 绘制训练数据点
plt.scatter(x_test, y_test, label="test data")  # 绘制测试数据点
plt.plot(x_predictions, y_predictions, "r", label="Prediction")  # 绘制红色预测曲线
plt.xlabel(input_param_name)  # 设置x轴标签
plt.ylabel(output_param_name)  # 设置y轴标签
plt.title("Happy")  # 设置图表标题
plt.legend()  # 显示图例
plt.show()  # 显示图表
