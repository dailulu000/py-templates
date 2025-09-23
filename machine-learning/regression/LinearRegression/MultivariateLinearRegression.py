import numpy as np  # 导入NumPy库，用于科学计算和数组操作
import pandas as pd  # 导入Pandas库，用于数据处理和分析
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于基本数据可视化
import plotly  # 导入Plotly库，用于交互式3D可视化
import plotly.graph_objs as go  # 导入Plotly的图形对象模块

plotly.offline.init_notebook_mode()  # 初始化Plotly的离线模式，用于在Jupyter Notebook中显示图形
from linear_regression import LinearRegression  # 导入自定义的LinearRegression类

# 读取CSV格式的世界幸福报告2017年数据
data = pd.read_csv("data/world-happiness-report-2017.csv")

# 划分训练数据和测试数据，训练数据占80%
train_data = data.sample(frac=0.8)  # 随机选择80%的数据作为训练集
test_data = data.drop(train_data.index)  # 删除训练集数据，剩余的20%作为测试集

# 定义两个输入特征和输出目标的列名
input_param_name_1 = "Economy..GDP.per.Capita."  # 第一个输入特征：人均GDP
input_param_name_2 = "Freedom"  # 第二个输入特征：自由度
output_param_name = "Happiness.Score"  # 输出目标：幸福分数

# 准备训练数据，提取两个特征和标签并转换为NumPy数组
x_train = train_data[
    [input_param_name_1, input_param_name_2]
].values  # 训练特征，包含两个特征列
y_train = train_data[[output_param_name]].values  # 训练标签

# 准备测试数据
x_test = test_data[[input_param_name_1, input_param_name_2]].values  # 测试特征
y_test = test_data[[output_param_name]].values  # 测试标签

# 配置训练数据集的3D散点图
plot_training_trace = go.Scatter3d(
    x=x_train[:, 0].flatten(),  # x轴数据：第一个特征(GDP)的所有值
    y=x_train[:, 1].flatten(),  # y轴数据：第二个特征(自由度)的所有值
    z=y_train.flatten(),  # z轴数据：目标值(幸福分数)的所有值
    name="Training Set",  # 数据集名称：训练集
    mode="markers",  # 图表模式：散点图
    marker={
        "size": 10,  # 点的大小
        "opacity": 1,  # 点的透明度
        "line": {
            "color": "rgb(255, 255, 255)",  # 点的边框颜色
            "width": 1,  # 点的边框宽度
        },
    },
)

# 配置测试数据集的3D散点图
plot_test_trace = go.Scatter3d(
    x=x_test[:, 0].flatten(),  # x轴数据：第一个特征(GDP)的所有值
    y=x_test[:, 1].flatten(),  # y轴数据：第二个特征(自由度)的所有值
    z=y_test.flatten(),  # z轴数据：目标值(幸福分数)的所有值
    name="Test Set",  # 数据集名称：测试集
    mode="markers",  # 图表模式：散点图
    marker={
        "size": 10,  # 点的大小
        "opacity": 1,  # 点的透明度
        "line": {
            "color": "rgb(255, 255, 255)",  # 点的边框颜色
            "width": 1,  # 点的边框宽度
        },
    },
)

# 配置3D图表的布局
plot_layout = go.Layout(
    title="Date Sets",  # 图表标题
    scene={
        "xaxis": {"title": input_param_name_1},  # x轴标题：第一个特征名称
        "yaxis": {"title": input_param_name_2},  # y轴标题：第二个特征名称
        "zaxis": {"title": output_param_name},  # z轴标题：目标值名称
    },
    margin={"l": 0, "r": 0, "b": 0, "t": 0},  # 图表边距设置为0
)

# 组合训练集和测试集的可视化数据
plot_data = [plot_training_trace, plot_test_trace]

# 创建3D图形对象
plot_figure = go.Figure(data=plot_data, layout=plot_layout)

# 在浏览器中显示3D交互式图形
plotly.offline.plot(plot_figure)

# 设置梯度下降的超参数
num_iterations = 500  # 迭代次数
learning_rate = 0.01  # 学习率，控制参数更新步长
polynomial_degree = 0  # 多项式特征的阶数，0表示不使用多项式特征
sinusoid_degree = 0  # 正弦特征的阶数，0表示不使用正弦特征

# 创建线性回归模型实例并进行训练
linear_regression = LinearRegression(
    x_train, y_train, polynomial_degree, sinusoid_degree
)  # 初始化模型，传入训练数据和特征转换参数

(theta, cost_history) = linear_regression.train(
    learning_rate, num_iterations
)  # 训练模型，返回参数和损失历史

# 打印训练开始时和结束时的损失值
print("开始损失", cost_history[0])  # 第一次迭代的损失
print("结束损失", cost_history[-1])  # 最后一次迭代的损失

# 绘制损失函数随迭代次数变化的曲线
plt.plot(range(num_iterations), cost_history)  # x轴：迭代次数，y轴：损失值
plt.xlabel("Iterations")  # 设置x轴标签
plt.ylabel("Cost")  # 设置y轴标签
plt.title("Gradient Descent Progress")  # 设置图表标题：梯度下降进展
plt.show()  # 显示图表

# 生成预测点以可视化回归平面
predictions_num = 10  # 在每个特征维度上生成10个点，总共100个预测点

# 获取训练数据中两个特征的最小值和最大值
x_min = x_train[
    :, 0
].min()  # 第一个特征的最小值\ nx_max = x_train[:, 0].max()  # 第一个特征的最大值

x_max = x_train[:, 0].max()  # 第一个特征的最大值

y_min = x_train[:, 1].min()  # 第二个特征的最小值
y_max = x_train[:, 1].max()  # 第二个特征的最大值

# 在每个特征的取值范围内生成均匀分布的预测点
x_axis = np.linspace(x_min, x_max, predictions_num)  # 第一个特征的预测点
y_axis = np.linspace(y_min, y_max, predictions_num)  # 第二个特征的预测点

# 创建存储预测点的数组
x_predictions = np.zeros(
    (predictions_num * predictions_num, 1)
)  # 存储第一个特征的预测点
y_predictions = np.zeros(
    (predictions_num * predictions_num, 1)
)  # 存储第二个特征的预测点

# 填充预测点数组，生成网格点
x_y_index = 0
for x_index, x_value in enumerate(x_axis):
    for y_index, y_value in enumerate(y_axis):
        x_predictions[x_y_index] = x_value  # 设置第一个特征值
        y_predictions[x_y_index] = y_value  # 设置第二个特征值
        x_y_index += 1

# 使用训练好的模型对所有预测点进行预测
z_predictions = linear_regression.predict(
    np.hstack((x_predictions, y_predictions))
)  # 合并两个特征列并进行预测

# 配置预测平面的3D可视化
plot_predictions_trace = go.Scatter3d(
    x=x_predictions.flatten(),  # x轴数据：第一个特征的预测点
    y=y_predictions.flatten(),  # y轴数据：第二个特征的预测点
    z=z_predictions.flatten(),  # z轴数据：预测的目标值
    name="Prediction Plane",  # 图表名称：预测平面
    mode="markers",  # 图表模式：散点图
    marker={
        "size": 1,  # 点的大小
    },
    opacity=0.8,  # 透明度
    surfaceaxis=2,  # 将点连接成平面，2表示z轴方向
)

# 组合训练集、测试集和预测平面的可视化数据
plot_data = [plot_training_trace, plot_test_trace, plot_predictions_trace]
plot_figure = go.Figure(data=plot_data, layout=plot_layout)  # 创建新的3D图形对象
plotly.offline.plot(plot_figure)  # 在浏览器中显示包含预测平面的3D交互式图形
