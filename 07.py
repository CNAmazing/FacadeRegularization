# import numpy as np

# # 示例二维数组
# data = np.array([
#     [1, 2, 3, 4],
#     [5, 6, 7, 8],
#     [9, 10, 11, 12]
# ])
# def x_error(data):
#     # 计算第1列（索引0）和第3列（索引2）的均值
#     col1_mean = data[:, 0].mean()  # 第1列均值
#     col3_mean = data[:, 2].mean()  # 第3列均值
#     # print("第1列均值:", col1_mean)  # 输出: 5.0
#     # print("第3列均值:", col3_mean)  # 输出: 7.0
#     # 计算每个元素与第1列均值的平方误差（仅第1列）
#     sq_error_col1 = (data[:, 0] - col1_mean) ** 2  # 仅计算第1列
#     # 计算每个元素与第3列均值的平方误差（仅第3列）
#     sq_error_col3 = (data[:, 2] - col3_mean) ** 2  # 仅计算第3列
#     # 计算平方误差和
#     total_sq_error_col1 = sq_error_col1.sum()  # 第1列平方误差和
#     total_sq_error_col3 = sq_error_col3.sum()  # 第3列平方误差和
#     return total_sq_error_col1, total_sq_error_col3
#     # print("第1列平方误差和:", total_sq_error_col1)  # 输出: 32.0
#     # print("第3列平方误差和:", total_sq_error_col3)  # 输出: 20.0

import numpy as np
from scipy.optimize import minimize

# 目标函数（示例：最小化 x0² + x1² + x2² + x3² + x4²）
def objective(x):
    return np.sum(x**2-6)

# 约束：x0 = x1 = x2 = x3 = x4
A_eq = np.array([
    [1, -1,  0,  0,  0],  # x0 - x1 = 0
    [0,  1, -1,  0,  0],  # x1 - x2 = 0
    [0,  0,  1, -1,  0],  # x2 - x3 = 0
])
b_eq = np.array([0, 0, 0, ])

# 初始猜测（任意值，这里设为 [1, 2, 3, 4, 5]）
x0 = np.array([1, 2, 3, 4, 5])

# 优化求解
result = minimize(
    objective,
    x0,
    constraints={'type': 'eq', 'fun': lambda x: A_eq @ x - b_eq},
    method='SLSQP'
)

print("最优解:", result.x)  # 输出应为 [c, c, c, c, c]，其中 c 是一个常数