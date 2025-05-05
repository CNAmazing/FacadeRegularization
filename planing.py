import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import json
import numpy as np
from typing import Literal
import random
from dtw import dtw
def read_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误：{e}")
        return None
def xyxy_to_xywh(bboxes):
    array=[]
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        array.append([x1, y1, x2 - x1, y2 - y1])
    return np.array(array)

# 计算矩形中心点
def get_center(rect):
    x1, y1, w, h = rect
    cx = x1 + w / 2
    cy = y1 + h / 2
    return cx, cy


# 可视化矩形和标签
def visualize_rectangles(rectangles):
    """
    可视化矩形及其相邻关系，并将坐标原点设为左上角（y 轴朝下，x 轴朝右）。
    
    参数:
        rectangles: 矩形列表，格式为 [(x1, y1, width, height), ...]
        nearest_rectangles: 相邻矩形信息，格式为 [(i, {'up': ..., 'down': ..., 'left': ..., 'right': ...}), ...]
        data: 一维数据点，用于绘制散点图（可选）
        labels: 数据点的标签，用于着色（可选）
        x_threshold: x 轴阈值（可选）
        y_threshold: y 轴阈值（可选）
        cls: 类别信息（可选）
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制矩形
    for i, rect in enumerate(rectangles):
        x1, y1, width, height = rect
        rect_patch = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect_patch)
        
        # 标注矩形编号
        cx, cy = get_center(rect)
        ax.text(cx, cy, str(i), fontsize=12, ha='center', va='center', color='blue')
    

    ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1))  # 修改：将图例放置在绘图区域
    # 设置坐标轴
    ax.set_xlim(0, 1000)
    ax.set_ylim(1000, 0)  # 将 y 轴反转，使原点在左上角
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.title("Rectangles and Their Nearest Neighbors")
    plt.show()


def remove_random_element(rectangles,delete_count=1):
    if rectangles.size == 0:
        return rectangles  # 如果数组为空，直接返回
    # 生成随机索引

       
    random_index = np.random.randint(0, len(rectangles), size=delete_count)
    # 删除随机索引处的元素
    result = np.delete(rectangles, random_index, axis=0)
    
    return result
def ray_intersection_detection(start,end,rectangles,i,j):
    for idx,rect in enumerate(rectangles):
        if idx == i or idx == j:
            continue
        x1, y1, w, h = rect
        # 计算矩形的四个顶点
        A = (x1, y1)
        B = (x1 + w, y1)
        C = (x1 + w, y1 + h)
        D = (x1, y1 + h)
        # 检查射线是否与矩形相交
        if segments_intersect(start, end, A, B) or segments_intersect(start, end, B, C) or \
           segments_intersect(start, end, C, D) or segments_intersect(start, end, D, A):
            return False
    return True 
def rectangle_neiborhood(rectangles):
    relations = []
    for i in range(len(rectangles)):
        neighbor=[]
        vectorGroup = []
        rect1 = rectangles[i]
        w,h=rect1[2],rect1[3]
        for j in range(len(rectangles)):
            if j == i:
                continue
            rect2 = rectangles[j]
            # 计算矩形的中心点
            start,end=get_StartAndEnd(rect1,rect2)
            # 计算向量
            vec_ = tuple([end[0] - start[0], end[1] - start[1],w,h])
            # 检查射线是否与其他矩形相交
            if ray_intersection_detection(start,end,rectangles,i,j):
                neighbor.append(j)
                vectorGroup.append(tuple(vec_))
            
        relations.append((i, tuple(neighbor), tuple(vectorGroup))) 
    return relations       
def segments_intersect(A, B, C, D):
    # Unpack points
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C
    x4, y4 = D
    
    # Compute bounding boxes
    def bounding_box_intersect():
        return (max(x1, x2) >= min(x3, x4) and
                min(x1, x2) <= max(x3, x4) and
                max(y1, y2) >= min(y3, y4) and
                min(y1, y2) <= max(y3, y4))
    
    if not bounding_box_intersect():
        return False
    
    # Compute cross products
    def cross_product(a, b, c):
        return (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])
    
    d1 = cross_product(A, B, C)
    d2 = cross_product(A, B, D)
    d3 = cross_product(C, D, A)
    d4 = cross_product(C, D, B)
    
    # Check straddling
    if (d1 * d2 < 0) and (d3 * d4 < 0):
        return True
    
    # Check collinear overlapping
    if d1 == 0 and d2 == 0 and d3 == 0 and d4 == 0:
        # Check overlap in x or y
        def on_segment(a, b, c):
            return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and
                    min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))
        
        return (on_segment(A, B, C) or on_segment(A, B, D) or
                on_segment(C, D, A) or on_segment(C, D, B))
    
    return False
def get_StartAndEnd(rect1,rect2):
    """
    计算两个矩形的邻域关系
    :param rect1: 第一个矩形 (x1, y1, w, h)
    :param rect2: 第二个矩形 (x1, y1, w, h)
    :return: 邻域关系字典
    """
    x1_1, y1_1, w1_1, h1_1 = rect1
    x2_2, y2_2, w2_2, h2_2 = rect2
    
    # 计算中心点坐标
    cx1 = x1_1 + w1_1 / 2
    cy1 = y1_1 + h1_1 / 2
    cx2 = x2_2 + w2_2 / 2
    cy2 = y2_2 + h2_2 / 2
    
    start=(cx1, cy1)
    end=(cx2, cy2)
    return start,end
def pairwise_L2_distance(A, B):
    """计算 A 和 B 之间的 L2 距离矩阵"""
    v1= np.array(A)
    v2= np.array(B)
    
    return  np.linalg.norm(v1 - v2)
def L2_And_Size_distance(A, B):
    """计算 A 和 B 之间的 L2 距离矩阵"""
    v1= np.array(A)
    v2= np.array(B)
    vector_error = np.linalg.norm(v1[:2] - v2[:2])
    size_error = abs(v1[2]*v1[3] - v2[2]*v2[3])
    return  vector_error + size_error
# 读取 JSON 文件
data = read_json(r'E:\WorkSpace\FacadeRegularization\data2.json')

# 矩形表示: (x1, y1, w, h)
rectangles = data['window']
rectangles = xyxy_to_xywh(rectangles)
rectangles=remove_random_element(rectangles,delete_count=2)
relations = rectangle_neiborhood(rectangles)
for r in relations:
    print(r)
    print("\n") 
# print("邻域关系：", relations)
k = 2

for i in range(len(relations)):
    idx=-1
    min_error = math.inf
    min_indices_Group = None
    
    for j in range(len(relations)):
        if i == j:
            continue
        align = dtw(relations[i][2], relations[j][2], dist=L2_And_Size_distance)
        # print(f"DTW 距离 between {relations[i][0]} and {relations[j][0]}: {align}")
        # 计算局部成本矩阵
        local_cost_matrix = align[1]
        cost_matrix = align[2]
        cost_array=[]
        idx_array=[]
        for r,c in zip(align[-1][0],align[-1][1]):
            cost=cost_matrix[r][c]
            cost_array.append(cost)
            idx_array.append(tuple([relations[i][1][r],relations[j][1][c]]))
        cost_array=np.array(cost_array)
        idx_array=np.array(idx_array)
        for idx_ in range(len(cost_array)):
            if idx_ >=1:
                cost_array[idx_] = cost_array[idx_] - cost_array[idx_-1]
        min_k_indices = np.argpartition(cost_array, k)[:k]
        # print(min_k_indices)
        # flat_indices = np.argpartition(local_cost_matrix.flatten(), k)[:k]
        
        # error = np.sum(local_cost_matrix.flatten()[min_k_indices])
        error = align[0]
        # top_k_matches = np.unravel_index(flat_indices, local_cost_matrix.shape)
        
        if error < min_error:
            min_error = error
            idx=j
            min_indices_Group = min_k_indices
    print(f"最小距离的邻域关系为：{relations[i][0]} and {relations[idx][0]}")
    print(f"最小距离为：{min_error}")
    # for min_idx in min_indices_Group:
    #     print(f'{relations[]}')
        
visualize_rectangles(rectangles)