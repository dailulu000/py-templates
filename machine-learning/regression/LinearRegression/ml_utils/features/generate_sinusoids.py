"""向特征集添加正弦函数特征"""

import numpy as np


def generate_sinusoids(dataset, sinusoid_degree):
    """
    生成基于正弦函数的特征变换。
    对于数据集中的每个特征x，生成sin(x), sin(2x), sin(3x), ..., sin(nx)等特征。
    
    参数:
    ----------
    dataset : numpy.ndarray
        输入数据集，形状为 (样本数, 特征数)
    sinusoid_degree : int
        要生成的正弦函数的最高频率系数
        
    返回:
    ----------
    numpy.ndarray
        包含新生成的正弦函数特征的数据集
    """
    
    # 获取数据集的样本数量
    num_examples = dataset.shape[0]
    
    # 创建一个空数组，用于存储生成的正弦函数特征
    # 初始时列数为0，后续会逐步添加新特征
    sinusoids = np.empty((num_examples, 0))
    
    # 为每个频率系数生成正弦函数特征
    for degree in range(1, sinusoid_degree + 1):
        # 计算当前频率系数下的正弦函数值：sin(degree * x)
        # 这将数据映射到不同频率的正弦曲线上，有助于捕获数据中的周期性模式
        sinusoid_features = np.sin(degree * dataset)
        # 将新生成的正弦函数特征追加到结果数组中
        sinusoids = np.concatenate((sinusoids, sinusoid_features), axis=1)
        
    return sinusoids
