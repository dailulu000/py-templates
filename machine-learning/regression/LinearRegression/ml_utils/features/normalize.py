"""对特征进行标准化处理"""

import numpy as np


def normalize(features):
    """
    对输入特征进行标准化处理，使每个特征的均值为0，标准差为1。
    标准化处理可以加速梯度下降等优化算法的收敛过程，提高模型训练效果。
    
    参数:
    ----------
    features : numpy.ndarray
        输入特征矩阵，形状为 (样本数, 特征数)
        
    返回:
    ----------
    tuple
        返回一个包含三个元素的元组：
        - 标准化后的特征矩阵
        - 每个特征的均值
        - 每个特征的标准差
    """
    
    # 创建特征矩阵的副本，并转换为浮点型以避免整数运算的精度问题
    features_normalized = np.copy(features).astype(float)
    
    # 计算每个特征的均值
    # axis=0 表示沿着列方向计算，即对每个特征计算所有样本的均值
    features_mean = np.mean(features, 0)
    
    # 计算每个特征的标准差
    features_deviation = np.std(features, 0)
    
    # 标准化操作第一步：减去均值，使特征均值为0
    # 注意：只有当样本数大于1时才进行此操作，否则会导致除以0的错误
    if features.shape[0] > 1:
        features_normalized -= features_mean
    
    # 防止除以0：将标准差为0的特征的标准差设置为1
    # 标准差为0表示该特征在所有样本中取值相同，无需标准化
    features_deviation[features_deviation == 0] = 1
    
    # 标准化操作第二步：除以标准差，使特征的标准差为1
    features_normalized /= features_deviation
    
    return features_normalized, features_mean, features_deviation
