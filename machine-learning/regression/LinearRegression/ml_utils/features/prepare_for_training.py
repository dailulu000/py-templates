"""准备用于训练的数据集"""

import numpy as np
from .normalize import normalize  # 导入数据标准化函数
from .generate_sinusoids import generate_sinusoids  # 导入正弦函数特征生成函数
from .generate_polynomials import generate_polynomials  # 导入多项式特征生成函数


def prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
    """
    对训练数据进行预处理，包括标准化和特征变换（多项式特征和正弦函数特征）。
    这是一个综合的特征预处理函数，通常在模型训练前调用。
    
    参数:
    ----------
    data : numpy.ndarray
        原始训练数据，形状为 (样本数, 特征数)
    polynomial_degree : int, 可选
        要生成的多项式特征的最高次数，默认为0（不生成多项式特征）
    sinusoid_degree : int, 可选
        要生成的正弦函数特征的最高频率系数，默认为0（不生成正弦函数特征）
    normalize_data : bool, 可选
        是否对数据进行标准化处理，默认为True
        
    返回:
    ----------
    tuple
        返回一个包含三个元素的元组：
        - 预处理后的训练数据（已添加偏置项）
        - 原始数据的特征均值（用于后续对测试数据进行相同的标准化）
        - 原始数据的特征标准差（用于后续对测试数据进行相同的标准化）
    """
    
    # 计算样本总数
    num_examples = data.shape[0]
    
    # 创建数据的副本，避免修改原始数据
    data_processed = np.copy(data)
    
    # 初始化特征均值和标准差为0
    features_mean = 0
    features_deviation = 0
    
    # 如果需要，对数据进行标准化处理
    if normalize_data:
        # 调用normalize函数进行标准化
        (data_normalized, features_mean, features_deviation) = normalize(data_processed)
        # 更新处理后的数据为标准化后的数据
        data_processed = data_normalized
    
    # 如果指定了sinusoid_degree > 0，则生成正弦函数特征并添加到数据中
    if sinusoid_degree > 0:
        # 生成正弦函数特征
        sinusoids = generate_sinusoids(data_normalized, sinusoid_degree)
        # 将正弦函数特征与现有数据水平拼接
        data_processed = np.concatenate((data_processed, sinusoids), axis=1)
    
    # 如果指定了polynomial_degree > 0，则生成多项式特征并添加到数据中
    if polynomial_degree > 0:
        # 生成多项式特征
        polynomials = generate_polynomials(data_normalized, polynomial_degree, normalize_data)
        # 将多项式特征与现有数据水平拼接
        data_processed = np.concatenate((data_processed, polynomials), axis=1)
    
    # 添加偏置项（bias term）：在数据前添加一列全为1的特征
    # 这是为了使线性回归模型能够拟合截距项
    data_processed = np.hstack((np.ones((num_examples, 1)), data_processed))
    
    # 返回预处理后的数据、特征均值和标准差
    return data_processed, features_mean, features_deviation
