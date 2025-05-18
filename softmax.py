import numpy as np

def softmax(z):
    exp_z = np.exp(z - np.max(z))  # 减去最大值以提高数值稳定性
    return exp_z / np.sum(exp_z)

# 示例输入
z = np.array([1, 2, 3])
# 计算Softmax输出
output = softmax(z)
print("Softmax输出:", output)
print("概率和:", np.sum(output))