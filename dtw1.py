from dtw import dtw
import numpy as np
import matplotlib.pyplot as plt
def L2_Norm(a, b):
    return np.sqrt(np.sum((a - b) ** 2))
# 示例序列
a = np.array([1, 3, 4, 5]).reshape(-1, 1)  # 必须为2D数组
b = np.array([1, 5]).reshape(-1, 1)
# 计算 DTW
alignment = dtw(a, b,dist=L2_Norm)

print("DTW 距离:", alignment)  # DTW 距离
# 可视化对齐路径
# alignment.plot(type="twoway")

cost_matrix = alignment[1]
accumulated_cost = alignment[2]
path_x, path_y = alignment[3]  # 规整路径的索引

# 假设原始序列（根据成本矩阵形状推断）
x = a # 长度=4（成本矩阵行数）
y = b            # 长度=2（成本矩阵列数）

# ====================== 1. 绘制成本矩阵和规整路径 ======================
plt.figure(figsize=(12, 5))

# 成本矩阵 + 规整路径
plt.subplot(1, 2, 1)
plt.imshow(cost_matrix.T, origin='lower', cmap='viridis', interpolation='nearest')
plt.plot(path_x, path_y, 'r-', linewidth=2)  # 红色路径
plt.colorbar(label="Cost")
plt.title("Cost Matrix with Warp Path")
plt.xlabel("Sequence X (index)")
plt.ylabel("Sequence Y (index)")

# ====================== 2. 绘制序列对齐情况 ======================
plt.subplot(1, 2, 2)
plt.plot(x, '-o', label='Sequence X')
plt.plot(y, '-o', label='Sequence Y')
for i, j in zip(path_x, path_y):
    plt.plot([i, j], [x[i], y[j]], 'r--', alpha=0.3)  # 虚线连接匹配点
plt.title("Sequence Alignment")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()