# import numpy as np

# # 输入 Labels 数组
# Labels = np.array([0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 3, 4, 3, 4, 3, 4, 3, 4, 0, 1, 5, 6, 5, 6, 5, 6, 3, 4, 2, 3, 5, 6, 2, 3, 5, 6, 2, 3, 3, 4, 5, 6, 2, 3, 0, 1, 0, 1])

# # 定义要查找的值
# x = 3

# # 获取值为 x 的元素的索引集合
# indices = np.where(Labels == x)[0]

# # 输出结果
# print(f"值为 {x} 的元素的索引集合:", indices)
import numpy as np

# 原始数据
data = np.array([
    [1, 1, 1, 5, 3],
    [1, 1, 0, 5, 3],
    [1, 1, 1, 5, 0],
    [1,1,1,1,1],
    [1, 0, 1, 5, 3]
])

# 标记缺失值的位置
missing_mask = (data == 0)
print("Missing values mask:\n", missing_mask)
# 初始化均值和方差
# 数据中所有可能的取值
unique_values = np.unique(data[data != 0])  # 忽略缺失值
n_values = len(unique_values)

# 初始化多项式分布的概率（均匀分布）
prob = np.ones(n_values) / n_values
print("Initial probabilities:", prob)
max_iter = 100
tolerance = 1e-6
for iteration in range(max_iter):
    # E步：估计缺失值的概率分布
    data_imputed = data.copy()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if missing_mask[i, j]:
                # 使用多项式分布的概率补全缺失值
                data_imputed[i, j] = np.random.choice(unique_values, p=prob)

    # M步：更新多项式分布的概率
    value_counts = np.zeros(n_values)
    for idx, value in enumerate(unique_values):
        value_counts[idx] = np.sum(data_imputed == value)
    new_prob = value_counts / np.sum(value_counts)

    # 判断是否收敛
    if np.max(np.abs(new_prob - prob)) < tolerance:
        break

    prob = new_prob

print("Final probabilities:", prob)
print("Imputed data:\n", data_imputed)