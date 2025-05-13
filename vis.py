import json 
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

def xyxy_to_xywh(bboxes):
    array = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        array.append([x1, y1, x2 - x1, y2 - y1])
    return np.array(array)

def read_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误：{e}")
        return None

# 读取数据
data1 = read_json(r'E:\WorkSpace\FacadeRegularization\data2.json')
data2 = read_json(r'E:\WorkSpace\FacadeRegularization\data7.json')

# 转换为xywh格式
rect1 = xyxy_to_xywh(data1['window'])
rect2 = xyxy_to_xywh(data2['window'])

# 创建画布
fig, ax = plt.subplots(figsize=(12, 10))

def get_center(rect):
    x1, y1, w, h = rect
    cx = x1 + w / 2
    cy = y1 + h / 2
    return cx, cy

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