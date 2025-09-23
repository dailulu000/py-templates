import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # 添加numpy库，用于数学计算和数组操作

from linear_regression import LinearRegression  # 导入自定义的线性回归类

# 读取数据集
# 从CSV文件中加载非线性回归的训练数据
# 数据格式应该包含x（自变量）和y（因变量）两列
# 文件路径：data/non-linear-regression-x-y.csv
print("正在加载数据集...")
data = pd.read_csv('data/non-linear-regression-x-y.csv')

# 数据预处理
# 将DataFrame中的数据转换为numpy数组，并重塑为列向量形式
# x是特征矩阵，形状为(样本数, 1)
x = data['x'].values.reshape((data.shape[0], 1))
# y是目标变量，形状为(样本数, 1)
y = data['y'].values.reshape((data.shape[0], 1))

# 查看前10行数据，用于快速检查数据格式和内容
data.head(10)

# 可视化原始数据分布
# 绘制散点图，观察x和y之间的关系
print("正在绘制原始数据分布图...")
plt.plot(x, y)
plt.xlabel('x值')
plt.ylabel('y值')
plt.title('原始数据关系图')
plt.grid(True)
plt.show()

# 模型参数设置
num_iterations = 50000  # 梯度下降的迭代次数
learning_rate = 0.02   # 学习率，控制参数更新步长
polynomial_degree = 15  # 多项式特征的最高次数，用于非线性转换
sinusoid_degree = 15    # 正弦函数特征的最高次数，用于非线性转换
normalize_data = True   # 是否对数据进行标准化处理

# 创建线性回归模型实例
# LinearRegression类内部会自动处理非线性特征转换
linear_regression = LinearRegression(x, y, polynomial_degree, sinusoid_degree, normalize_data)

# 训练模型
# 使用梯度下降算法优化模型参数
# 返回训练后的模型参数theta和每次迭代的损失值
print("开始训练模型...")
(theta, cost_history) = linear_regression.train(
    learning_rate,
    num_iterations
)

# 打印训练结果
# 输出初始损失值和最终损失值，评估模型训练效果
print('开始损失: {:.2f}'.format(cost_history[0]))
print('结束损失: {:.2f}'.format(cost_history[-1]))

# 创建模型参数的DataFrame表格，方便查看
theta_table = pd.DataFrame({'Model Parameters': theta.flatten()})
print("模型参数表格预览:")
print(theta_table.head())  # 打印前几行参数

# 可视化训练过程中的损失函数下降曲线
# 观察梯度下降是否收敛
print("正在绘制损失函数下降曲线...")
plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iterations')  # 迭代次数
plt.ylabel('Cost')       # 损失值
plt.title('Gradient Descent Progress')  # 梯度下降训练过程
plt.grid(True)
plt.show()

# 生成预测数据点
# 创建一组均匀分布的x值，用于生成预测结果曲线
predictions_num = 1000  # 预测点的数量
x_predictions = np.linspace(x.min(), x.max(), predictions_num).reshape(predictions_num, 1);

# 使用训练好的模型进行预测
# 注意：模型内部会自动对输入数据进行与训练时相同的特征转换和标准化
print("正在进行预测...")
y_predictions = linear_regression.predict(x_predictions)

# 可视化预测结果
# 对比原始数据点和模型预测的曲线
print("正在绘制预测结果图...")
plt.scatter(x, y, label='Training Dataset')  # 原始训练数据散点图
plt.plot(x_predictions, y_predictions, 'r', label='Prediction')  # 预测结果曲线（红色）
plt.xlabel('x值')
plt.ylabel('y值')
plt.title('非线性回归预测结果')
plt.legend()  # 显示图例
plt.grid(True)
plt.show()

print("程序执行完毕！")