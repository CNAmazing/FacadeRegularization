import os
from typing import List, Tuple
import numpy as np
import json
from sklearn.cluster import MeanShift
import cvxpy as cp
import matplotlib.patches as patches
import matplotlib.pyplot as plt
def xyxy_to_xywh(bboxes):
    array=[]
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        array.append([x1, y1, x2 - x1, y2 - y1])
    return np.array(array)
def get_xywh(windows, x, y, w, h):
    for item in windows:
        x.append(item[0])
        y.append(item[1])
        w.append(item[2])
        h.append(item[3])
def read_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误：{e}")
        return None
    
def pre_cluster(x: list[float], delta: float) -> list[float]:
    """
    使用MeanShift对一维数据x进行聚类，返回聚类中心
    参数:
        x: 原始一维数据列表
        delta: 带宽参数（控制聚类半径）
    返回:
        聚类中心列表X
    """
    if not x:  # 如果 points 是空列表
        return [], []
    # 将一维数据转为二维 [(x1,0), (x2,0), ...]
    points = np.array([[xi, 0] for xi in x])
    
    # 初始化MeanShift模型（带宽=delta）
    ms = MeanShift(bandwidth=delta)
    ms.fit(points)
    
    # 提取聚类中心（取第一维，忽略补0的维度）
    X = [center[0] for center in ms.cluster_centers_]
    return X
def get_result(r, X, Y, W, H, N, m, n, m_, n_):
    """
    从优化结果中提取每个矩形的最终参数 (x, y, w, h)
    
    参数:
        r: 优化结果向量（整数形式）
        X, Y, W, H: 聚类中心列表
        N: 数据点数量
        m, n, m_, n_: 聚类中心数量（x, y, w, h方向）
    
    返回:
        result: 格式化字符串，每行表示一个矩形的优化后参数
    """
    # 初始化存储分配索引的列表
    r_x, r_y, r_w, r_h = [], [], [], []
    
    # 提取x方向的分配
    for i in range(N):
        for k in range(m):
            if r[i * m + k] != 0:
                r_x.append(k)
    
    # 提取y方向的分配
    for i in range(N):
        for k in range(n):
            if r[N * m + i * n + k] != 0:
                r_y.append(k)
    
    # 提取w方向的分配
    for i in range(N):
        for k in range(m_):
            if r[N * m + N * n + i * m_ + k] != 0:
                r_w.append(k)
    
    # 提取h方向的分配
    for i in range(N):
        for k in range(n_):
            if r[N * m + N * n + N * m_ + i * n_ + k] != 0:
                r_h.append(k)
    
    # 生成结果字符串
    result = []
    for i in range(N):
        x = X[r_x[i]]
        y = Y[r_y[i]]
        w = W[r_w[i]]
        h = H[r_h[i]]
        result.append([x, y, w, h])
    
    return result
def regularize(x, y, w, h, X, Y, W, H, alpha_x, alpha_y, alpha_w, alpha_h):
    N = len(x)
    m, n, m_, n_ = len(X), len(Y), len(W), len(H)
    idx= N + m + n + m_ + n_
    count =N * (m + n + m_ + n_) + (m + n + m_ + n_)
    a1 = []
    for xi in x:
        a1.extend([(xi - Xk)**2 for Xk in X])
    for yi in y:
        a1.extend([(yi - Yk)**2 for Yk in Y])
    for wi in w:
        a1.extend([(wi - Wk)**2 for Wk in W])
    for hi in h:
        a1.extend([(hi - Hk)**2 for Hk in H])
    a1.extend([alpha_x] * m + [alpha_y] * n + [alpha_w] * m_ + [alpha_h] * n_)
    
    a2 = []
    for i in range(N):
        v = np.zeros(count)
        for k in range(m):
            v[i * m + k] = 1.0
        a2.append(v)
    
    for i in range(N):
        v = np.zeros(count)
        for k in range(n):
            v[N * m + i * n + k] = 1.0
        a2.append(v)

    for i in range(N):
        v = np.zeros(count)
        for k in range(m_):
            v[N * (m + n) + i * m_ + k] = 1.0
        a2.append(v)

    for i in range(N):
        v = np.zeros(count)
        for k in range(n_):
            v[N * (m + n + m_) + i * n_ + k] = 1.0
        a2.append(v)
    
    a3 = []
    
    # 1. X方向平衡约束
    for i in range(m):
        v = np.zeros(count)
        v[idx + i] = -1.0  # 特殊位置的负权重
        for k in range(N):
            v[k * m + i] = 1.0  # 每个数据点对该X中心的分配
        a3.append(v)
    
    # 2. Y方向平衡约束
    for i in range(n):
        v = np.zeros(count)
        v[idx + m + i] = -1.0
        for k in range(N):
            v[N * m + k * n + i] = 1.0
        a3.append(v)
    
    # 3. W方向平衡约束
    for i in range(m_):
        v = np.zeros(count)
        v[idx + m + n + i] = -1.0
        for k in range(N):
            v[N * (m + n) + k * m_ + i] = 1.0
        a3.append(v)
    
    # 4. H方向平衡约束
    for i in range(n_):
        v = np.zeros(count)
        v[idx + m + n + m_ + i] = -1.0
        for k in range(N):
            v[N * (m + n + m_) + k * n_ + i] = 1.0
        a3.append(v)
    a4 = []
    
    # 1. X方向分配约束
    for i in range(m):
        for k in range(N):
            v = np.zeros(count)
            v[k * m + i] = 1.0    # 数据点k分配到X中心i
            v[idx + i] = -1.0      # 对应X中心i的容量
            a4.append(v)
    
    # 2. Y方向分配约束
    for i in range(n):
        for k in range(N):
            v = np.zeros(count)
            v[N * m + k * n + i] = 1.0  # 数据点k分配到Y中心i
            v[idx + m + i] = -1.0        # 对应Y中心i的容量
            a4.append(v)
    
    # 3. W方向分配约束
    for i in range(m_):
        for k in range(N):
            v = np.zeros(count)
            v[N * (m + n) + k * m_ + i] = 1.0  # 数据点k分配到W中心i
            v[idx + m + n + i] = -1.0           # 对应W中心i的容量
            a4.append(v)
    
    # 4. H方向分配约束
    for i in range(n_):
        for k in range(N):
            v = np.zeros(count)
            v[N * (m + n + m_) + k * n_ + i] = 1.0  # 数据点k分配到H中心i
            v[idx + m + n + m_ + i] = -1.0           # 对应H中心i的容量
            a4.append(v)
    r = cp.Variable((N, m + n + m_ + n_), boolean=True)
    # 辅助变量（对应原C++中的x_变量）
    x_ = cp.Variable(N * (m + n + m_ + n_) + (m + n + m_ + n_), integer=True)
    
    # 2. 构建约束
    constraints = [x_ >= 0,x_<=1]  # 非负约束
    
    # 分配约束（a2）：每个数据点必须分配到恰好一个聚类中心
    for i in range(len(a2)):
        constraints.append(a2[i] @ x_ == 1)
    
    # 容量下限约束（a3）：聚类中心使用量不能太低
    for i in range(len(a3)):
        constraints.append(a3[i] @ x_ >= 0)
    
    # 容量上限约束（a4）：聚类中心使用量不能太高
    # for i in range(len(a4)):
    #     constraints.append(a4[i] @ x_ <= 0)
    
    # 3. 设置目标函数
    objective = cp.Minimize(a1 @ x_)
    
    # 4. 求解问题
    problem = cp.Problem(objective, constraints)
    problem.solve(solver='MOSEK',
                   verbose=True) # 使用整数规划求解器
    
    if problem.status == cp.OPTIMAL:
        return x_.value
    else:
        raise ValueError(f"求解失败，状态: {problem.status}")

def get_center(rect):
        x1, y1, w, h = rect
        cx = x1 + w / 2
        cy = y1 + h / 2
        return cx, cy
def rectangle_compare(rect1, rect2):
     # 创建画布
    fig, ax = plt.subplots(figsize=(12, 10))
    # 绘制rect1（蓝色边框+蓝色编号）
    for i, rect in enumerate(rect1):
        x1, y1, width, height = rect
        rect_patch = patches.Rectangle(
            (x1, y1), width, height, 
            linewidth=1, edgecolor='blue', 
            facecolor='none', linestyle='-'
        )
        ax.add_patch(rect_patch)
        cx, cy = get_center(rect)
        ax.text(cx, cy, f"1-{i}", fontsize=10, 
                ha='center', va='center', color='blue')

    # 绘制rect2（红色边框+红色编号）
    for i, rect in enumerate(rect2):
        x1, y1, width, height = rect
        rect_patch = patches.Rectangle(
            (x1, y1), width, height, 
            linewidth=1, edgecolor='red', 
            facecolor='none', linestyle='--'
        )
        ax.add_patch(rect_patch)
        cx, cy = get_center(rect)
        ax.text(cx, cy, f"2-{i}", fontsize=10, 
                ha='center', va='center', color='red')

    # 设置坐标轴等比例+显示网格
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.title('Rectangles Comparison (Blue: rect1, Red: rect2)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # 自动调整坐标范围
    all_rects = np.vstack([rect1, rect2])
    x_margin = max(all_rects[:, 2]) * 0.2
    y_margin = max(all_rects[:, 3]) * 0.2
    ax.set_xlim([np.min(all_rects[:, 0]) - x_margin, 
                np.max(all_rects[:, 0] + all_rects[:, 2]) + x_margin])
    ax.set_ylim([np.min(all_rects[:, 1]) - y_margin, 
                np.max(all_rects[:, 1] + all_rects[:, 3]) + y_margin])

    plt.tight_layout()
    plt.show()
def show_rectangles(rect1,rect2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 绘制rect1（蓝色色块+蓝色编号）
    for i, rect in enumerate(rect1):
        x1, y1, width, height = rect
        rect_patch = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=1, edgecolor='blue',
            facecolor='lightblue', alpha=0.7,  # 使用浅蓝色填充
            linestyle='-'
        )
        ax1.add_patch(rect_patch)
        cx, cy = get_center(rect)
        ax1.text(cx, cy, f"1-{i}", fontsize=10,
                ha='center', va='center', color='darkblue')

    # 绘制rect2（红色色块+红色编号）
    for i, rect in enumerate(rect2):
        x1, y1, width, height = rect
        rect_patch = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=1, edgecolor='red',
            facecolor='lightcoral', alpha=0.7,  # 使用浅红色填充
            linestyle='--'
        )
        ax2.add_patch(rect_patch)
        cx, cy = get_center(rect)
        ax2.text(cx, cy, f"2-{i}", fontsize=10,
                ha='center', va='center', color='darkred')

    # 设置子图1
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.set_title('Rectangles Group 1 (Blue)')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')

    # 设置子图2
    ax2.set_aspect('equal')
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.set_title('Rectangles Group 2 (Red)')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')

    # 自动调整坐标范围
    all_rects = np.vstack([rect1, rect2])
    x_margin = max(all_rects[:, 2]) * 0.2
    y_margin = max(all_rects[:, 3]) * 0.2

    # 统一两个子图的坐标范围
    xlim = [np.min(all_rects[:, 0]) - x_margin, 
            np.max(all_rects[:, 0] + all_rects[:, 2]) + x_margin]
    ylim = [np.min(all_rects[:, 1]) - y_margin, 
            np.max(all_rects[:, 1] + all_rects[:, 3]) + y_margin]

    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)

    plt.tight_layout()
    plt.show()
        
def BIP(rectangles):
    """
    对矩形进行正则化处理
    参数:
        rectangles: 矩形列表，每个矩形格式为 (x, y, w, h)
    返回:
        处理后的矩形列表
    """
    x: List[float] = []
    y: List[float] = []
    w: List[float] = []
    h: List[float] = []
    X: List[float] = []
    Y: List[float] = []
    W: List[float] = []
    H: List[float] = []
    delta_x = 5                   # 聚类参数
    delta_y = 5
    delta_w = 3
    delta_h = 3
    # rectangles = xyxy_to_xywh(rectangles)
    get_xywh(rectangles, x, y, w, h)
    X = pre_cluster(x, delta_x)
    Y = pre_cluster(y, delta_y)
    W = pre_cluster(w, delta_w)
    H = pre_cluster(h, delta_h)
    alpha_x, alpha_y, alpha_w, alpha_h = 1, 1 ,1, 1
    r =regularize(x, y, w, h, X, Y, W, H, alpha_x, alpha_y, alpha_w, alpha_h)
    result= get_result(r, X, Y, W, H, len(x), len(X), len(Y), len(W), len(H))
    return result
def main():

    data = read_json('data2.json')
    # 矩形表示: (x1, y1, w, h)
    rect1 = data['window']
    rect1 = xyxy_to_xywh(rect1)
    rect2=BIP(rect1)
    rectangle_compare(rect1, rect2)
   


if __name__ == "__main__":
    main()