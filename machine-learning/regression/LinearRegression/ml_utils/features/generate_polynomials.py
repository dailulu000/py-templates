"""向特征集添加多项式特征"""

import numpy as np
from .normalize import normalize  # 导入数据标准化函数


def generate_polynomials(dataset, polynomial_degree, normalize_data=False):
    """
    生成多项式特征，通过以下方式变换数据：
    x1, x2, x1^2, x2^2, x1*x2, x1*x2^2, 等等。
    
    参数:
    ----------
    dataset : numpy.ndarray
        输入数据集，形状为 (样本数, 特征数)
    polynomial_degree : int
        要生成的多项式的最高次数
    normalize_data : bool, 可选
        是否对生成的多项式特征进行标准化处理，默认为False
        
    返回:
    ----------
    numpy.ndarray
        包含原始特征和新生成的多项式特征的扩展数据集
    """
    
    # 将数据集沿着列方向（特征方向）分成两部分
    # 这是为了能够生成交叉项特征（如x1*x2）
    features_split = np.array_split(dataset, 2, axis=1)
    dataset_1 = features_split[0]  # 第一部分数据集
    dataset_2 = features_split[1]  # 第二部分数据集
    
    # 获取两部分数据集的形状信息
    (num_examples_1, num_features_1) = dataset_1.shape  # 第一部分样本数和特征数
    (num_examples_2, num_features_2) = dataset_2.shape  # 第二部分样本数和特征数
    
    # 数据校验：确保两部分数据集具有相同的样本数
    if num_examples_1 != num_examples_2:
        raise ValueError('无法为行数不同的两个数据集生成多项式特征')
    
    # 数据校验：确保至少有一个数据集包含特征
    if num_features_1 == 0 and num_features_2 == 0:
        raise ValueError('无法为两列都为空的数据集生成多项式特征')
    
    # 处理特殊情况：如果其中一个数据集没有特征，则使用另一个数据集
    if num_features_1 == 0:
        dataset_1 = dataset_2
    elif num_features_2 == 0:
        dataset_2 = dataset_1
    
    # 确保两部分数据集具有相同的特征数
    # 选择较小的特征数作为共同特征数
    num_features = num_features_1 if num_features_1 < num_examples_2 else num_features_2
    # 截取数据集，仅保留相同数量的特征列
    dataset_1 = dataset_1[:, :num_features]
    dataset_2 = dataset_2[:, :num_features]
    
    # 创建一个空数组，用于存储生成的多项式特征
    polynomials = np.empty((num_examples_1, 0))
    
    # 生成多项式特征
    # i表示当前生成的多项式次数
    for i in range(1, polynomial_degree + 1):
        # j表示当前次数中dataset_2的指数
        for j in range(i + 1):
            # 计算多项式特征：dataset_1^(i-j) * dataset_2^j
            # 这样可以生成所有可能的组合，如x1^2, x1*x2, x2^2等
            polynomial_feature = (dataset_1 ** (i - j)) * (dataset_2 ** j)
            # 将新生成的多项式特征追加到结果数组中
            polynomials = np.concatenate((polynomials, polynomial_feature), axis=1)
    
    # 如果需要，对生成的多项式特征进行标准化处理
    if normalize_data:
        polynomials = normalize(polynomials)[0]  # 只需要返回的标准化后的特征，不需要均值和标准差
    
    return polynomials
