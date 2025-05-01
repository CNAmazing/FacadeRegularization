import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import json
import numpy as np
from typing import Literal
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
# 读取 JSON 文件
data = read_json(r'E:\WorkSpace\RegularFacade\data1.json')

# 矩形表示: (x1, y1, w, h)
rectangles = data['window']
rectangles = xyxy_to_xywh(rectangles)
def calculate_score(direction,distance, angle,w1=0.5,w2=0.5):
    threshold = 60
    match direction:
        case 'up':
            if abs(angle-90)>threshold:
                return float('inf')
            return w1 * distance + w2 * abs(angle-90)
        case 'down':
            if abs(angle-270)>threshold:
                return float('inf')
            return w1 * distance + w2 * abs(angle-270)
        case 'left':
            if abs(angle-180)>threshold:
                return float('inf')
            return w1 * distance + w2 * abs(angle-180)
        case 'right':
            if min(abs(angle),abs(angle-360))>threshold:
                return float('inf')
            return w1 * distance + w2 * min(abs(angle),abs(angle-360))
        


# 计算矩形中心点
def get_center(rect):
    x1, y1, w, h = rect
    cx = x1 + w / 2
    cy = y1 + h / 2
    return cx, cy

# 计算两个矩形中心点的向量和距离
def calculate_vector_and_distance(rect1, rect2):
    cx1, cy1 = get_center(rect1)
    cx2, cy2 = get_center(rect2)
    dx = cx2 - cx1
    dy = cy2 - cy1
    distance = math.sqrt(dx**2 + dy**2)
    return dx, dy, distance
# 找到每个矩形在不同方向上最近的矩形
def find_nearest_rectangles(rectangles):
    result = []
    for i, rect in enumerate(rectangles):
        nearest = {'up': None, 'down': None, 'left': None, 'right': None}
        min_score = {'up': float('inf'), 'down': float('inf'), 'left': float('inf'), 'right': float('inf')}
        
        for j, other_rect in enumerate(rectangles):
            if i == j:
                continue  # 跳过自身
            dx, dy, distance = calculate_vector_and_distance(rect, other_rect)
            angle = math.degrees(math.atan2(dy, dx))

            # 转换为 0 到 360 度
            if angle < 0:
                angle += 360

            
            # 计算各方向的得分（向量分量乘以距离）
            score_up = calculate_score('up',distance,angle) # 上
            score_down = calculate_score('down',distance,angle)  # 下
            score_left = calculate_score('left',distance,angle)  # 左
            score_right = calculate_score('right',distance,angle)   # 右
            
            # 更新最小得分
            if score_up < min_score['up']:
                nearest['up'] = j
                min_score['up'] = score_up
            if score_down < min_score['down']:
                nearest['down'] = j
                min_score['down'] = score_down
            if score_left < min_score['left']:
                nearest['left'] = j
                min_score['left'] = score_left
            if score_right < min_score['right']:
                nearest['right'] = j
                min_score['right'] = score_right
            
        # 去重
        # for key in nearest.keys():
        #     for key2 in nearest.keys():
        #         if nearest[key] == nearest[key2] and key != key2:
        #             if min_score[key] < min_score[key2]:
        #                 nearest[key2] = None
        #             else:
        #                 nearest[key] = None
        
        result.append((i, nearest))
    return result

# 可视化矩形和标签
def visualize_rectangles(rectangles, nearest_rectangles, x_data=None, y_data=None,x_labels=None,y_labels=None, x_threshold=None, y_threshold=None,):
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
    
    # 标注相邻矩形
    for i, nearest in nearest_rectangles:
        rect = rectangles[i]
        cx, cy = get_center(rect)
        
        # 标注上下左右相邻的矩形
        if nearest['down'] is not None:
            ax.text(cx, cy + 5, f"↑{nearest['down']}", fontsize=10, ha='center', va='bottom', color='red')
        if nearest['up'] is not None:
            ax.text(cx, cy - 5, f"↓{nearest['up']}", fontsize=10, ha='center', va='top', color='red')
        if nearest['left'] is not None:
            ax.text(cx - 5, cy, f"←{nearest['left']}", fontsize=10, ha='right', va='center', color='red')
        if nearest['right'] is not None:
            ax.text(cx + 5, cy, f"→{nearest['right']}", fontsize=10, ha='left', va='center', color='red')
    
    # 绘制散点图（如果提供了 data 和 labels）
    if x_data is not None and x_labels is not None:
        colors = plt.cm.tab20(x_labels)  # 使用 tab20 颜色映射
        scatter = ax.scatter(x_data, np.zeros_like(x_data), c=colors, s=100, alpha=0.8)
        
        # # 添加数据点标签
        # for i, (xi, label) in enumerate(zip(x_data, x_labels)):
        #     ax.text(xi, 0, f'{xi} ({label})', fontsize=9, ha='center', va='bottom')
        
        # 添加图例
        unique_labels = np.unique(x_labels)
        for label in unique_labels:
            ax.scatter([], [], c=plt.cm.tab20(label), label=f'Cluster {label}')
    if y_data is not None and y_labels is not None:
        colors = plt.cm.tab20(y_labels)  # 使用 tab10 颜色映射
        scatter = ax.scatter(np.zeros_like(y_data),y_data, c=colors, s=100, alpha=0.8)

        # # 添加数据点标签
        # for i, (xi, label) in enumerate(zip(y_data, y_labels)):
        #     ax.text(xi, 0, f'{xi} ({label})', fontsize=9, ha='center', va='bottom')
        
        # 添加图例
        unique_labels = np.unique(y_labels)
        for label in unique_labels:
            ax.scatter([], [], c=plt.cm.tab20(label), label=f'Cluster {label}')
    
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1))  # 修改：将图例放置在绘图区域
    # 设置坐标轴
    ax.set_xlim(0, 1000)
    ax.set_ylim(1000, 0)  # 将 y 轴反转，使原点在左上角
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.title("Rectangles and Their Nearest Neighbors")
    plt.show()
def find_min_horizontal_vertical_spacing(rectangles, nearest_rectangles):
    min_horizontal_spacing = float('inf')
    min_vertical_spacing = float('inf')
    
    for i, nearest in nearest_rectangles:
        rect1 = rectangles[i]
        x1, y1, w1, h1 = rect1
        if min_horizontal_spacing > w1:
            min_horizontal_spacing = w1
            print(f'最小水平间距已更新为第{i}个矩形的宽度：{w1}')
        if min_vertical_spacing > h1:
            min_vertical_spacing = h1
            print(f'最小垂直间距已更新为第{i}个矩形的高度：{h1}')
        for direction, neighbor_idx in nearest.items():
            if neighbor_idx is not None:
                rect2 = rectangles[neighbor_idx]
                x2, y2, w2, h2 = rect2
                
                # 计算水平间距
                if x1 + w1 < x2:  # rect1 在 rect2 的左侧
                    horizontal_spacing = x2 - (x1 + w1)
                elif x2 + w2 < x1:  # rect1 在 rect2 的右侧
                    horizontal_spacing = x1 - (x2 + w2)
                else:  # 矩形在水平方向上有重叠
                    horizontal_spacing = float('inf')
                
                # 计算垂直间距
                if y1 + h1 < y2:  # rect1 在 rect2 的下方
                    vertical_spacing = y2 - (y1 + h1)
                elif y2 + h2 < y1:  # rect1 在 rect2 的上方
                    vertical_spacing = y1 - (y2 + h2)
                else:  # 矩形在垂直方向上有重叠
                    vertical_spacing = float('inf')
                
                # 更新最小水平间距
                
                if horizontal_spacing < min_horizontal_spacing:
                    min_horizontal_spacing = horizontal_spacing
                    print('最小水平间距已更新，索引为：',i,neighbor_idx)
                # 更新最小垂直间距
                if vertical_spacing < min_vertical_spacing:
                    min_vertical_spacing = vertical_spacing
                    print('最小垂直间距已更新，索引为：',i,neighbor_idx)
    return min_horizontal_spacing, min_vertical_spacing
def get_labels_by_Projection_point_array(point_array,threshold,rectangles_relationship):
    labels = -np.ones(len(point_array), dtype=int)
    current_label = 0
    for i in range(len(point_array)):
        if labels[i] == -1:  # 如果该点未分类
            while True:
                # 找到距离该点小于等于阈值的点
                neighbors = np.where(np.abs(point_array - point_array[i]) <= threshold)[0]
                neighbors_idx=neighbors//2
                # 检查是否存在当前点和邻居点在 rectangles_relationship 中
                conflict = False
                print('当前索引:',i//2)
                print('neighbors_idx:',neighbors_idx)
                for neighbor in neighbors_idx:
                    if (i//2, neighbor) in rectangles_relationship or (neighbor, i//2) in rectangles_relationship:
                        conflict = True
                        break
                
                if not conflict:
                    # 如果没有冲突，标记这些点为当前类别
                    
                    labels[neighbors] = current_label
                    break
                else:
                    # 如果有冲突，缩小阈值
                    threshold *= 0.9  # 缩小阈值为原来的 90%
                    print(f"缩小阈值至 {threshold}，重新检查点 {i}")
            current_label += 1
    return labels

def get_rectangles_relationship(nearest_rectangles,
                                direction:Literal['Horizontal','Vertical']):
    """
    Horizontal表示水平投影
    Vertical表示垂直投影
    """
    rectangles_relationship=set()
    match direction:
        case 'Vertical':
            for rect in nearest_rectangles:
                for key,value in rect[1].items():
                    if value is not None:
                        if key=='left' or key=='right':
                            rectangles_relationship.add((value,rect[0]))
        case 'Horizontal':
            for rect in nearest_rectangles:
                for key,value in rect[1].items():
                    if value is not None:
                        if key=='up' or key=='down':
                            rectangles_relationship.add((value,rect[0]))            
    return rectangles_relationship
# 运行并输出结果
nearest_rectangles = find_nearest_rectangles(rectangles)
min_horizontal_spacing, min_vertical_spacing = find_min_horizontal_vertical_spacing(rectangles, nearest_rectangles)
print("最小水平间距:", min_horizontal_spacing)
print("最小垂直间距:", min_vertical_spacing)
x_projection=[]
y_projection=[]
for box in rectangles:
    x_projection.append(box[0])
    x_projection.append(box[0]+box[2])
    y_projection.append(box[1])
    y_projection.append(box[1]+box[3])


x_projection=np.array(x_projection)
y_projection=np.array(y_projection)

x_rectangles_relationship=get_rectangles_relationship(nearest_rectangles,direction='Vertical')
y_rectangles_relationship=get_rectangles_relationship(nearest_rectangles,'Horizontal')

print("矩形关系:",x_rectangles_relationship)
x_labels=get_labels_by_Projection_point_array(x_projection,
                                              min_vertical_spacing,
                                              x_rectangles_relationship,
                                              
                                              )
y_labels=get_labels_by_Projection_point_array(y_projection,
                                              min_horizontal_spacing,
                                              y_rectangles_relationship,
                                                )

print("nearest_rectangles:",nearest_rectangles)
print("x_labels:", x_labels)
print("y_labels:", y_labels)

visualize_rectangles(rectangles, 
                     nearest_rectangles,
                     x_data=x_projection,
                     y_data=y_projection,
                     x_threshold=min_vertical_spacing,
                     y_threshold=min_horizontal_spacing,
                     x_labels=x_labels,
                     y_labels=y_labels)